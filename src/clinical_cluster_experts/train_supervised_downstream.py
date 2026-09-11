""" Supervised training on downstream tasks. 

The script reads the target names dynamically from tokenizer_metadata.pt: "target_columns"
For each patient it builds:
    target_binary: [T] float tensor with 0/1 labels
    mask_binary:   [T] bool tensor, False for missing labels

Missing labels do not contribute to the loss or metrics. Supervised optimization
uses Transformer-style AdamW parameter groups: weight decay is applied to regular
weight matrices but not to biases, normalization parameters, embeddings, CLS
tokens, one-dimensional parameters, or parameters that remain frozen. Optional
linear warmup followed by cosine learning-rate decay scales all learning-rate
groups while preserving their ratio.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.tokenizer.dataset import TokenizedTabularDataset
from src.clinical_cluster_experts.model import ClinicalClusterEncoder, DownstreamPredictionHead
from src.clinical_cluster_experts.utils import (
    resolve_device,
    move_batch_to_device,
    load_cluster_assignments,
    load_embeddings,
    load_pretrained_branches,
)
from src.clinical_cluster_experts.summary_diagnostics import (
    SummaryTokenDiagnosticsAccumulator,
    append_summary_diagnostics_csv,
    log_summary_diagnostics_tensorboard,
)
from src.clinical_cluster_experts.summary_auxiliary_losses import (
    SUMMARY_AUX_LOSS_CHOICES,
    compute_summary_auxiliary_loss,
)

LOSS_CHOICES = ("weighted_bce", "focal")


class SupervisedTokenizedDataset(Dataset):
    """TokenizedTabularDataset plus dynamic binary targets from metadata['target_columns']."""

    def __init__(self, token_dir: Path, split: str, target_columns: list[str] | None = None):
        self.base = TokenizedTabularDataset.from_dir(token_dir, split=split)
        self.metadata = self.base.metadata
        self.split = split
        self.target_columns = list(target_columns or self.metadata.get("target_columns", []))
        if not self.target_columns:
            raise ValueError("No target columns found. Expected metadata['target_columns'] or --target-columns.")
        self.targets = torch.load(token_dir / f"{split}_targets.pt", map_location="cpu", weights_only=True)
        if not isinstance(self.targets, dict):
            raise ValueError(f"Expected {split}_targets.pt to be a dict, got {type(self.targets)}")
        for name in self.target_columns:
            if name not in self.targets:
                raise KeyError(f"Missing target {name!r} in {split}_targets.pt")
            if len(self.targets[name]) != len(self.base):
                raise ValueError(f"Target {name!r} has len={len(self.targets[name])}, dataset has len={len(self.base)}")

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _label_value_and_mask(x: Any) -> tuple[float, bool]:
        if x is None:
            return 0.0, False
        try:
            if np.isnan(x):
                return 0.0, False
        except TypeError:
            pass
        return float(x), True

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = dict(self.base[idx])
        values, masks = [], []
        for name in self.target_columns:
            value, mask = self._label_value_and_mask(self.targets[name][idx])
            values.append(value); masks.append(mask)
        item["target_binary"] = torch.tensor(values, dtype=torch.float32)
        item["mask_binary"] = torch.tensor(masks, dtype=torch.bool)
        return item


# use the same seed for every Optuna trial so that LR/weight decay/cluster dropout max are compared under the same initialization, batch order, and feature/cluster-dropout random sequence.
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _nanmean(values: list[float]) -> float:
    """Compute mean of finite values, return NaN if no finite values exist."""
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def binary_auroc_auprc(y_true: torch.Tensor | np.ndarray, y_score: torch.Tensor | np.ndarray) -> tuple[float, float]:
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_score):
        y_score = y_score.detach().cpu().numpy()
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    keep = np.isfinite(y_score)
    y_true, y_score = y_true[keep], y_score[keep]
    if y_true.size == 0:
        return float("nan"), float("nan")
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos > 0 and n_neg > 0:
        ranks = _average_ranks(y_score)
        auroc = (float(ranks[y_true == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    else:
        auroc = float("nan")
    if n_pos > 0:
        order = np.argsort(-y_score, kind="mergesort")
        y_sorted = y_true[order]
        tp_cum = np.cumsum(y_sorted == 1)
        fp_cum = np.cumsum(y_sorted == 0)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        auprc = float(precision[y_sorted == 1].sum() / n_pos)
    else:
        auprc = float("nan")
    return float(auroc), float(auprc)




def build_adamw_param_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    group_name: str,
) -> list[dict]:
    """Build Transformer-style AdamW parameter groups.

    - Weight decay is applied to regular weight matrices only. Biases, normalization parameters, embedding-like parameters, 
        CLS tokens, one-dimensional parameters, and frozen parameters are excluded from decay.
    - The explicit learning rate allows the expert bank and fusion/head modules to use different rates during end-to-end fine-tuning.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        no_decay = (
            name_lower.endswith(".bias")
            or "norm" in name_lower
            or "layernorm" in name_lower
            or "embedding" in name_lower
            or "cls_token" in name_lower
            or param.ndim < 2
        )
        if no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict] = []
    if decay_params:
        groups.append(
            {
                "params": decay_params,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "group_name": f"{group_name}_decay",
            }
        )
    if no_decay_params:
        groups.append(
            {
                "params": no_decay_params,
                "lr": float(learning_rate),
                "weight_decay": 0.0,
                "group_name": f"{group_name}_no_decay",
            }
        )
    return groups


def get_optimizer_group_value(
    optimizer: torch.optim.Optimizer,
    group_prefix: str,
    key: str,
) -> float:
    """Read a value from the first optimizer group with the requested prefix."""
    for group in optimizer.param_groups:
        if str(group.get("group_name", "")).startswith(group_prefix):
            return float(group[key])
    return float("nan")


def build_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float, min_lr_ratio: float):
    """Create a scheduler where args.lr is the peak LR.

    The learning rate increases linearly during warmup and then follows cosine
    decay down to min_lr_ratio * peak_lr. The scheduler is stepped once after
    each optimizer update.
    """
    total_steps = max(int(total_steps), 1)
    warmup_frac = float(warmup_frac)
    min_lr_ratio = float(min_lr_ratio)
    warmup_steps = max(1, int(warmup_frac * total_steps)) if warmup_frac > 0.0 else 0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), warmup_steps


def run_epoch(
    *,
    encoder: ClinicalClusterEncoder,
    head: DownstreamPredictionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    device: torch.device,
    active_clusters: list[int],
    ignored_feature_mask: torch.Tensor | None = None,
    feature_dropout_prob_max: float = 0.0,
    cluster_dropout_prob_max: float = 0.0,
    target_columns: list[str],
    binary_loss_type: str,
    binary_pos_weights: torch.Tensor | None,
    focal_gamma: float,
    train_experts: bool,
    trainable_parameters: list[nn.Parameter],
    train: bool,
    max_batches: int = 0,
    collect_summary_diagnostics: bool = False,
    summary_aux_loss: str = "none",
    summary_aux_weight: float = 0.0,
    summary_attention_margin: float = 0.9,
    summary_output_margin: float = 0.9,
    summary_mi_beta: float = 1.0,
    summary_mi_temperature: float = 1.0,
) -> dict[str, Any]:
    # Experts use dropout only when they are being fine-tuned. Validation always
    # runs the complete model in evaluation mode.
    encoder.expert_bank.train(train and train_experts)
    encoder.fusion.train(train)
    head.train(train)

    if collect_summary_diagnostics and train:
        raise ValueError(
            "Summary-token diagnostics are evaluation-only. Collect them on "
            "validation/test data where dropout is disabled."
        )
    summary_accumulator = (
        SummaryTokenDiagnosticsAccumulator(
            num_clusters=encoder.expert_bank.n_clusters,
            num_summary_tokens=encoder.num_summary_tokens,
            d_model=encoder.expert_bank.d_model,
        )
        if collect_summary_diagnostics
        else None
    )

    T = len(target_columns)
    short_names = [name.removeprefix("label_") for name in target_columns]

    totals = {
        "loss_total": 0.0,
        "loss_optimization": 0.0,
        "summary_aux_raw": 0.0,
        "summary_aux_weighted": 0.0,
        "summary_aux_ratio": 0.0,
        "summary_pairwise_similarity": 0.0,
        "summary_conditional_entropy": 0.0,
        "summary_marginal_entropy": 0.0,
        "grad_norm": 0.0,
    }
    for name in short_names:
        totals[f"loss_{name}"] = 0.0; totals[f"n_{name}"] = 0.0

    n_batches = 0

    num_active_clusters = len(active_clusters)
    cluster_count_sum = 0.0
    cluster_count_sq_sum = 0.0
    cluster_count_hist = [0] * (num_active_clusters + 1)
    cluster_keep_counts = {c: 0 for c in active_clusters}
    sampled_p_cluster_sum = 0.0

    score_chunks: list[list[torch.Tensor]] = [[] for _ in range(T)]
    target_chunks: list[list[torch.Tensor]] = [[] for _ in range(T)]
    tp = torch.zeros(T, dtype=torch.float64); tn = torch.zeros(T, dtype=torch.float64)
    fp = torch.zeros(T, dtype=torch.float64); fn = torch.zeros(T, dtype=torch.float64)

    iterable = islice(loader, max_batches) if max_batches else loader
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    iterator = tqdm(enumerate(iterable), total=total_batches, desc="train" if train else "val", leave=False)

    for _, batch in iterator:
        batch = move_batch_to_device(batch, device)
        B, N = batch["numeric_values"].shape

        if "observed_mask" in batch:
            feature_available_mask = batch["observed_mask"].bool()
        else:
            feature_available_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Explicitly remove ignored-cluster features from schema evidence.
        if ignored_feature_mask is not None and ignored_feature_mask.numel() > 0:
            ignored_mask = ignored_feature_mask.to(device=device, dtype=torch.bool)
            if ignored_mask.numel() != N:
                raise ValueError(f"ignored_feature_mask has length {ignored_mask.numel()}, expected {N}.")
            feature_available_mask = feature_available_mask.clone()
            feature_available_mask[:, ignored_mask] = False

        # randomly drop features during training for robustness (to simulate feature missingness)
        if train and feature_dropout_prob_max > 0.0:
            p_feat = float(torch.rand((), device=device) * feature_dropout_prob_max)
            feature_available_mask = feature_available_mask & (torch.rand(B, N, device=device) >= p_feat)

        # randomly drop clusters during training for robustness (to simulate cluster missingness)
        p_cluster = 0.0
        batch_active_clusters = list(active_clusters)
        if train and cluster_dropout_prob_max > 0.0 and len(batch_active_clusters) > 1:
            max_clusters_to_drop = min(
                int(math.floor(cluster_dropout_prob_max * len(batch_active_clusters))),
                len(batch_active_clusters) - 1,
            )
            if max_clusters_to_drop > 0:
                n_drop = int(torch.randint(max_clusters_to_drop + 1, (1,), device=device).item())
                p_cluster = n_drop / float(len(batch_active_clusters))
                if n_drop > 0:
                    drop_indices = torch.randperm(len(batch_active_clusters), device=device)[:n_drop]
                    drop_mask = torch.zeros(len(batch_active_clusters), dtype=torch.bool, device=device)
                    drop_mask[drop_indices] = True
                    batch_active_clusters = [c for c, drop_c in zip(batch_active_clusters, drop_mask.detach().cpu().tolist()) if not drop_c]
        
        # Record the realized cluster availability after dropout.
        n_clusters_used = len(batch_active_clusters)
        cluster_count_sum += float(n_clusters_used)
        cluster_count_sq_sum += float(n_clusters_used**2)
        cluster_count_hist[n_clusters_used] += 1
        sampled_p_cluster_sum += p_cluster
        for c in batch_active_clusters:
            cluster_keep_counts[c] += 1

        with torch.set_grad_enabled(train):
            need_summary_attention = collect_summary_diagnostics or summary_aux_loss in {
                "attention_margin", "attention_mi"
            }
            enc_out = encoder.forward_active_clusters(
                batch=batch,
                active_clusters=batch_active_clusters,
                feature_available_mask=feature_available_mask,
                return_summary_attention=need_summary_attention,
            )
            if summary_accumulator is not None:
                summary_accumulator.update(
                    group_tokens=enc_out["group_tokens"],
                    cluster_available_mask=enc_out["cluster_available_mask"],
                    attention_overlap=enc_out.get("summary_attention_overlap"),
                )
            outputs = head(enc_out["patient_embedding"])
            logits = outputs["binary_logits"]
            targets = batch["target_binary"].float()
            masks = batch["mask_binary"].bool()
            losses, loss_logs = [], {}
            loss_type = binary_loss_type
            for j, name in enumerate(short_names):
                m = masks[:, j]
                if not m.any():
                    loss_j = logits[:, j].sum() * 0.0
                else:
                    if loss_type == "weighted_bce":
                        pos_weight = None if binary_pos_weights is None else binary_pos_weights[j].to(device).view(())
                        loss_j = F.binary_cross_entropy_with_logits(logits[m, j], targets[m, j], reduction="mean", pos_weight=pos_weight)
                    elif loss_type == "focal":
                        bce = F.binary_cross_entropy_with_logits(logits[m, j], targets[m, j], reduction="none", pos_weight=None,)
                        probs = torch.sigmoid(logits[m, j])
                        pt = torch.where(targets[m, j] > 0.5, probs, 1.0 - probs).clamp(min=1e-8, max=1.0)
                        loss_j = ((1.0 - pt).pow(float(focal_gamma)) * bce).mean()
                    else:
                        raise ValueError(f"Unknown binary loss type: {loss_type}")
                losses.append(loss_j); loss_logs[f"loss_{name}"] = float(loss_j.detach().cpu()); loss_logs[f"n_{name}"] = float(m.sum().detach().cpu())
            
            # Keep the supervised objective separate from the optimization objective
            # so checkpoint selection remains comparable across loss ablations.
            supervised_loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0
            aux = compute_summary_auxiliary_loss(
                summary_aux_loss,
                group_tokens=enc_out["group_tokens"],
                cluster_available_mask=enc_out["cluster_available_mask"],
                attention_by_cluster=enc_out.get("summary_feature_attention"),
                feature_mask_by_cluster=enc_out.get("summary_feature_available_mask"),
                attention_margin=summary_attention_margin,
                output_margin=summary_output_margin,
                mi_beta=summary_mi_beta,
                mi_temperature=summary_mi_temperature,
            )
            weighted_aux = float(summary_aux_weight) * aux["loss"]
            optimization_loss = supervised_loss + weighted_aux

            grad_norm = 0.0
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                optimization_loss.backward()
                # Clip every parameter that is currently trainable. This includes
                # the expert bank only when --train-experts is enabled.
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        1.0,
                    ).detach().cpu()
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits.detach())
            pred = probs >= 0.5
            target_bool = targets.bool()
            for j in range(T):
                m = masks[:, j]
                if not m.any():
                    continue
                pred_j, true_j = pred[:, j][m], target_bool[:, j][m]
                tp[j] += float((pred_j & true_j).sum().detach().cpu())
                tn[j] += float(((~pred_j) & (~true_j)).sum().detach().cpu())
                fp[j] += float((pred_j & (~true_j)).sum().detach().cpu())
                fn[j] += float(((~pred_j) & true_j).sum().detach().cpu())
                score_chunks[j].append(probs[:, j][m].cpu())
                target_chunks[j].append(targets[:, j][m].cpu())

        supervised_value = float(supervised_loss.detach().cpu())
        aux_raw_value = float(aux["loss"].detach().cpu())
        aux_weighted_value = float(weighted_aux.detach().cpu())
        totals["loss_total"] += supervised_value
        totals["loss_optimization"] += float(optimization_loss.detach().cpu())
        totals["summary_aux_raw"] += aux_raw_value
        totals["summary_aux_weighted"] += aux_weighted_value
        totals["summary_aux_ratio"] += abs(aux_weighted_value) / max(abs(supervised_value), 1e-12)
        totals["summary_pairwise_similarity"] += float(aux["pairwise_similarity"].detach().cpu())
        totals["summary_conditional_entropy"] += float(aux["conditional_entropy"].detach().cpu())
        totals["summary_marginal_entropy"] += float(aux["marginal_entropy"].detach().cpu())
        totals["grad_norm"] += grad_norm
        for k, v in loss_logs.items():
            totals[k] += float(v)
        n_batches += 1
        iterator.set_postfix(sup=f"{supervised_value:.4f}", aux=f"{aux_weighted_value:.4f}")

    denom = max(n_batches, 1)
    out = {k: (v if k.startswith("n_") else v / denom) for k, v in totals.items()}

    if n_batches > 0:
        clusters_used_mean = cluster_count_sum / n_batches
        #clusters_used_second_moment = cluster_count_sq_sum / n_batches
        #clusters_used_variance = max(clusters_used_second_moment - clusters_used_mean**2, 0.0,)

        out["num_clusters_used_mean"] = clusters_used_mean
        # out["num_clusters_used_std"] = math.sqrt(clusters_used_variance)
        # out["cluster_fraction_used_mean"] = (clusters_used_mean / num_active_clusters)
        # out["cluster_fraction_dropped_mean"] = (1.0 - out["cluster_fraction_used_mean"])
        out["sampled_cluster_dropout_prob_mean"] = (sampled_p_cluster_sum / n_batches)
        for k in range(1, num_active_clusters + 1):
            out[f"num_clusters_used_{k}_fraction"] = (cluster_count_hist[k] / n_batches)
        for c in active_clusters:
            out[f"cluster_{c}_keep_fraction"] = (cluster_keep_counts[c] / n_batches)
    else:
        out["num_clusters_used_mean"] = float("nan")
        # out["num_clusters_used_std"] = float("nan")
        # out["cluster_fraction_used_mean"] = float("nan")
        # out["cluster_fraction_dropped_mean"] = float("nan")
        out["sampled_cluster_dropout_prob_mean"] = float("nan")

        for k in range(1, num_active_clusters + 1):
            out[f"num_clusters_used_{k}_fraction"] = float("nan")

        for c in active_clusters:
            out[f"cluster_{c}_keep_fraction"] = float("nan")
            
    aurocs, auprcs, baccs, f1s = [], [], [], []
    for j, name in enumerate(short_names):
        if score_chunks[j]:
            y_score, y_true = torch.cat(score_chunks[j]), torch.cat(target_chunks[j])
            auroc, auprc = binary_auroc_auprc(y_true, y_score)
        else:
            auroc, auprc = float("nan"), float("nan")
        pos_support, neg_support = tp[j] + fn[j], tn[j] + fp[j]
        recall_pos = tp[j] / pos_support.clamp_min(1.0) if pos_support > 0 else torch.tensor(float("nan"))
        recall_neg = tn[j] / neg_support.clamp_min(1.0) if neg_support > 0 else torch.tensor(float("nan"))
        bacc = _nanmean([float(recall_pos), float(recall_neg)])
        precision = tp[j] / (tp[j] + fp[j]).clamp_min(1.0)
        recall = tp[j] / (tp[j] + fn[j]).clamp_min(1.0)
        f1 = float((2 * precision * recall / (precision + recall).clamp_min(1e-12)).item()) if (tp[j] + fp[j] + fn[j]) > 0 else float("nan")
        out[f"{name}_auroc"] = auroc; out[f"{name}_auprc"] = auprc; out[f"{name}_balanced_acc"] = bacc; out[f"{name}_f1"] = f1
        aurocs.append(auroc); auprcs.append(auprc); baccs.append(bacc); f1s.append(f1)
    out["macro_auroc"] = _nanmean(aurocs); out["macro_auprc"] = _nanmean(auprcs)
    out["macro_balanced_acc"] = _nanmean(baccs); out["macro_f1"] = _nanmean(f1s)
    if summary_accumulator is not None:
        out["summary_diagnostics"] = summary_accumulator.compute()
    return out


def parse_int_list(text: str | None) -> list[int]:
    """Parse a comma-separated integer list. Empty/None returns an empty list."""
    if text is None:
        return []
    text = str(text).strip()
    if text == "" or text.lower() == "none":
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def metric_is_better(metric_name: str, new_value: float, best_value: float) -> bool:
    """Return whether new_value improves over best_value for the selected metric."""
    if not np.isfinite(new_value):
        return False
    if metric_name == "loss_total":
        return new_value < best_value
    return new_value > best_value


def sanitize_metric(metric_name: str, value: float) -> float:
    """Make metric safe for checkpoint selection and Optuna reporting."""
    value = float(value)
    if np.isfinite(value):
        return value
    return float("inf") if metric_name == "loss_total" else -float("inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised SGMET training with frozen or trainable pretrained experts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token-dir", type=str, required=True)
    parser.add_argument("--cluster-csv", type=str, required=True)
    parser.add_argument("--num-clusters", type=int, required=True)
    parser.add_argument("--active-clusters", type=str, default="all", help="Comma-separated clusters, e.g. '1', '0,1,2', or 'all'.")
    parser.add_argument("--ignore-clusters", type=str, default="none", help=("Comma-separated cluster ids to remove from active clusters and from schema evidence"))
    parser.add_argument("--feature-dropout-prob-max", type=float, default=0.0,
        help=("Upper bound for supervised feature/schema dropout. During training, each batch samples p ~ Uniform(0, max) and removes features from encoder evidence. Validation uses no random dropout."))
    parser.add_argument("--cluster-dropout-prob-max", type=float, default=0.0,
        help=("Upper bound for supervised cluster dropout during training. For K active clusters, each batch samples n_drop uniformly from 0..floor(max * K), then drops exactly n_drop randomly selected whole expert clusters before fusion. At least one active cluster is kept."))
    parser.add_argument("--branch-checkpoint", action="append", default=[], help="Pretrained branch checkpoint as cluster_id=path. Can be repeated.")
    parser.add_argument("--strict-branch-load", action="store_true")
    parser.add_argument("--allow-random-active-branches", action="store_true",
        help="Allow active clusters without pretrained checkpoints. Usually leave this off except for debugging.")
    parser.add_argument("--init-supervised-checkpoint", type=str, default=None,
        help=("Optional supervised checkpoint used to initialize the full encoder and prediction head. "
            "Use this for stage-2 end-to-end fine-tuning from a best frozen-expert checkpoint. Optimizer and scheduler states are intentionally not resumed."))
    parser.add_argument("--train-experts", action="store_true", help=("Fine-tune the expert bank together with the fusion Transformer and prediction head. "))
    parser.add_argument("--expert-lr", type=float, default=5e-5,help="Peak/fixed learning rate for trainable experts. Used only with --train-experts.")
    parser.add_argument("--expert-weight-decay", type=float, default=None, help=("Weight decay for regular expert weight matrices. Defaults to --weight-decay. "
            "Biases, normalization parameters, embeddings, CLS tokens, and 1D parameters receive no decay."))
    parser.add_argument("--target-columns", type=str, default=None, help="Optional comma-separated target list. Defaults to metadata['target_columns'].")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-summary-tokens", type=int, default=1, help=("Number of independently initialized CLS summary tokens per expert."))
    parser.add_argument("--summary-aux-loss", type=str, default="none", choices=SUMMARY_AUX_LOSS_CHOICES,
        help="Enable exactly one summary-token auxiliary loss, or none.")
    parser.add_argument("--summary-aux-weight", type=float, default=0.0,
        help="Lambda multiplying the selected summary auxiliary loss.")
    parser.add_argument("--summary-aux-warmup-epochs", type=int, default=5,
        help="Linearly ramp lambda to its full value over this many epochs; 0 disables the ramp.")
    parser.add_argument("--summary-attention-margin", type=float, default=0.9,
        help="Maximum unpenalized cosine overlap for attention_margin.")
    parser.add_argument("--summary-output-margin", type=float, default=0.9,
        help="Maximum unpenalized output cosine similarity for output_margin.")
    parser.add_argument("--summary-mi-beta", type=float, default=1.0,
        help="Weight of marginal slot-usage entropy in attention_mi.")
    parser.add_argument("--summary-mi-temperature", type=float, default=1.0,
        help="Temperature for competitive p(slot|feature) in attention_mi.")
    parser.add_argument("--expert-n-heads", type=int, default=4)
    parser.add_argument("--expert-n-layers", type=int, default=2)
    parser.add_argument("--fusion-n-heads", type=int, default=4)
    parser.add_argument("--fusion-n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate when --lr-schedule=cosine; fixed learning rate when --lr-schedule=constant.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup-frac", type=float, default=0.05, help="Fraction of optimizer steps used for linear LR warmup when --lr-schedule=cosine.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.05, help="Final LR ratio relative to peak LR after cosine decay.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for model initialization, DataLoader shuffling, and feature/cluster dropout. The same seed is used for every HPO trial.")

    parser.add_argument("--binary-loss", type=str, default="focal", choices=LOSS_CHOICES)
    parser.add_argument("--class-weight-smoothing", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=20.0) # the max pos-weight for our targets are currently around 22. 
    parser.add_argument("--focal-gamma", type=float, default=1.0, choices=[0.5, 1.0, 2.0])
    parser.add_argument("--selection-metric", type=str, default="macro_auroc", choices=["macro_auroc", "macro_auprc", "loss_total"])
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="If >0, stop when selection metric does not improve for this many epochs. Mainly useful for final non-HPO training.")

    parser.add_argument("--no-cluster-embedding", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/supervised_trained_with_frozen_experts")
    parser.add_argument("--log-dir", type=str, default="runs/supervised_training_with_frozen_experts")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--summary-diagnostics", action="store_true", help=("Collect summary-token norm, variance, pairwise cosine similarity, and final-layer CLS-to-feature attention overlap on validation data. "
            "Diagnostics are written to separate CSV files and TensorBoard."),
    )

    # Optuna HPO over optimizer hyperparameters and maximum cluster dropout.
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO over lr, weight_decay, and cluster_dropout_prob_max.")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--optuna-study-name", type=str, default=None)
    parser.add_argument("--optuna-storage", type=str, default=None, help="Optuna storage URL. Defaults to sqlite:///<output-dir>/optuna_supervised.db")
    parser.add_argument("--optuna-lr-low", type=float, default=1e-5)
    parser.add_argument("--optuna-lr-high", type=float, default=3e-3)
    parser.add_argument("--optuna-weight-decay-low", type=float, default=1e-7)
    parser.add_argument("--optuna-weight-decay-high", type=float, default=1e-2)
    parser.add_argument("--optuna-cluster-dropout-prob-max-low", type=float, default=0.0)
    parser.add_argument("--optuna-cluster-dropout-prob-max-high", type=float, default=0.9)
    parser.add_argument("--optuna-startup-trials", type=int, default=8)
    parser.add_argument("--optuna-pruner-startup-trials", type=int, default=5)
    parser.add_argument("--optuna-pruner-warmup-steps", type=int, default=15, help="Number of completed epochs before a trial becomes eligible for pruning.")
    parser.add_argument("--optuna-pruner", type=str, default="median", choices=["median", "none"])
    parser.add_argument("--optuna-sampler-seed", type=int, default=42)
    parser.add_argument("--hpo-objective", type=str, default="macro_auroc", choices=["macro_auroc", "macro_auprc"], help="Validation metric maximized by Optuna.")
    args = parser.parse_args()

    for name in ["feature_dropout_prob_max", "cluster_dropout_prob_max", "warmup_frac", "min_lr_ratio"]:
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1], got {value}.")
    if args.num_summary_tokens < 1:
        raise ValueError("--num-summary-tokens must be >= 1.")
    if args.summary_aux_weight < 0:
        raise ValueError("--summary-aux-weight must be >= 0.")
    if args.summary_aux_warmup_epochs < 0:
        raise ValueError("--summary-aux-warmup-epochs must be >= 0.")
    if not (0.0 <= args.summary_attention_margin < 1.0):
        raise ValueError("--summary-attention-margin must be in [0, 1).")
    if not (-1.0 <= args.summary_output_margin < 1.0):
        raise ValueError("--summary-output-margin must be in [-1, 1).")
    if args.summary_mi_beta < 0:
        raise ValueError("--summary-mi-beta must be >= 0.")
    if args.summary_mi_temperature <= 0:
        raise ValueError("--summary-mi-temperature must be > 0.")
    if args.summary_aux_loss != "none":
        if args.num_summary_tokens < 2:
            raise ValueError("A summary auxiliary loss requires --num-summary-tokens >= 2.")
        if not args.train_experts:
            raise ValueError("A summary auxiliary loss requires --train-experts; frozen expert outputs cannot be diversified.")
        if args.summary_aux_weight <= 0:
            raise ValueError("Set --summary-aux-weight > 0 when a summary auxiliary loss is enabled.")
    if args.d_model % args.expert_n_heads != 0:
        raise ValueError("--d-model must be divisible by --expert-n-heads.")
    if args.d_model % args.fusion_n_heads != 0:
        raise ValueError("--d-model must be divisible by --fusion-n-heads.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0.")
    if args.pos_weight_max <= 0:
        raise ValueError("--pos-weight-max must be > 0.")
    if args.class_weight_smoothing < 0:
        raise ValueError("--class-weight-smoothing must be >= 0.")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0.")
    if args.expert_lr <= 0:
        raise ValueError("--expert-lr must be > 0.")
    if args.expert_weight_decay is not None and args.expert_weight_decay < 0:
        raise ValueError("--expert-weight-decay must be >= 0 when provided.")
    if args.optuna_lr_low <= 0 or args.optuna_lr_high <= 0 or args.optuna_lr_low >= args.optuna_lr_high:
        raise ValueError("Optuna lr bounds must be positive and low < high.")
    if args.optuna_weight_decay_low <= 0 or args.optuna_weight_decay_high <= 0 or args.optuna_weight_decay_low >= args.optuna_weight_decay_high:
        raise ValueError("Optuna weight_decay bounds must be positive and low < high.")
    if not (0.0 <= args.optuna_cluster_dropout_prob_max_low < args.optuna_cluster_dropout_prob_max_high <= 1.0):
        raise ValueError("Optuna cluster_dropout_prob_max bounds must satisfy 0 <= low < high <= 1.")
    if args.optuna_trials <= 0:
        raise ValueError("--optuna-trials must be > 0.")
    if args.optuna_startup_trials < 0 or args.optuna_pruner_startup_trials < 0 or args.optuna_pruner_warmup_steps < 0:
        raise ValueError("Optuna startup/warmup values must be >= 0.")
    return args


def run_training(args: argparse.Namespace, device: torch.device, trial: optuna.Trial | None = None) -> float:
    """Run one supervised frozen-expert training job and return the best HPO objective value."""
    trial_number = None if trial is None else int(trial.number)

    # reset all relevant RNGs at the start of every trial. Keeping this seed fixed across trials isolates the effects of LR, weight decay, and cluster dropout max.
    set_seed(args.seed)

    token_dir = Path(args.token_dir)
    metadata = TokenizedTabularDataset.from_dir(token_dir, split="train").metadata
    target_columns = [x.strip() for x in args.target_columns.split(",") if x.strip()] if args.target_columns else list(metadata["target_columns"])
    short_names = [name.removeprefix("label_") for name in target_columns]

    out_dir = Path(args.output_dir) if trial is None else Path(args.output_dir) / "optuna_trials" / f"trial_{trial_number:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SupervisedTokenizedDataset(token_dir, "train", target_columns)
    val_ds = SupervisedTokenizedDataset(token_dir, "val", target_columns)

    feature_names = list(metadata["feature_names"])
    feature_type_ids = torch.as_tensor(metadata["feature_type_ids"]).long()
    categorical_cardinalities = [int(x) for x in metadata["categorical_cardinalities"]]
    cluster_assignments = load_cluster_assignments(args.cluster_csv, feature_names)
    if int(cluster_assignments.max()) >= int(args.num_clusters):
        raise ValueError("cluster CSV contains cluster_id >= --num-clusters")

    ignore_clusters = set(parse_int_list(args.ignore_clusters))
    for c in ignore_clusters:
        if c < 0 or c >= args.num_clusters:
            raise ValueError(f"Invalid ignored cluster {c}; expected 0..{args.num_clusters - 1}.")

    active_text = str(args.active_clusters).strip().lower()
    if active_text == "all":
        active_clusters = [c for c in range(args.num_clusters) if c not in ignore_clusters]
    elif active_text in {"", "none"}:
        active_clusters = []
    else:
        active_clusters = [c for c in parse_int_list(active_text) if c not in ignore_clusters]

    for c in active_clusters:
        if c < 0 or c >= args.num_clusters:
            raise ValueError(f"Invalid active cluster {c}; expected 0..{args.num_clusters - 1}.")
    if not active_clusters:
        raise ValueError("No active clusters selected after applying --ignore-clusters.")

    ignored_feature_mask = torch.zeros(len(feature_names), dtype=torch.bool)
    if ignore_clusters:
        ignored_tensor = torch.tensor(sorted(ignore_clusters), dtype=cluster_assignments.dtype)
        ignored_feature_mask = torch.isin(cluster_assignments.cpu(), ignored_tensor)

    branch_checkpoints: dict[int, Path] = {}
    for item in args.branch_checkpoint:
        if "=" not in item:
            raise ValueError("--branch-checkpoint entries must look like '1=path/to/cluster_1_best.pt'")
        cluster_str, path_str = item.split("=", 1)
        cluster_id = int(cluster_str)
        if cluster_id in ignore_clusters:
            print(f"[WARN] Ignoring checkpoint for ignored cluster {cluster_id}: {path_str}")
            continue
        branch_checkpoints[cluster_id] = Path(path_str)

    # A supervised checkpoint already contains the complete encoder, including
    # every expert branch. Therefore branch checkpoints are required only when no
    # full supervised initialization checkpoint is supplied.
    init_supervised_checkpoint: Path | None = None
    init_supervised_state: dict[str, Any] | None = None
    if args.init_supervised_checkpoint:
        init_supervised_checkpoint = Path(args.init_supervised_checkpoint)
        if not init_supervised_checkpoint.is_file():
            raise FileNotFoundError(
                f"Supervised initialization checkpoint not found: {init_supervised_checkpoint}"
            )
        loaded = torch.load(init_supervised_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError(
                "Expected --init-supervised-checkpoint to contain a checkpoint dictionary."
            )
        if "encoder_state_dict" not in loaded or "head_state_dict" not in loaded:
            raise KeyError(
                "Supervised initialization checkpoint must contain "
                "'encoder_state_dict' and 'head_state_dict'."
            )
        init_supervised_state = loaded

        checkpoint_num_summary_tokens = int(loaded.get("num_summary_tokens", loaded.get("args", {}).get("num_summary_tokens", 1)))
        if checkpoint_num_summary_tokens != int(args.num_summary_tokens):
            raise ValueError("--init-supervised-checkpoint summary-token mismatch: "
                f"checkpoint uses {checkpoint_num_summary_tokens}, current run uses {args.num_summary_tokens}."
            )

        checkpoint_targets = loaded.get("target_columns")
        if checkpoint_targets is not None and list(checkpoint_targets) != list(target_columns):
            raise ValueError(
                "Target columns in --init-supervised-checkpoint do not match the current target columns. "
                f"Checkpoint={list(checkpoint_targets)}, current={target_columns}."
            )

    missing_active = [c for c in active_clusters if c not in branch_checkpoints]
    if (
        init_supervised_state is None
        and missing_active
        and not args.allow_random_active_branches
    ):
        raise ValueError(
            "Active clusters without pretrained checkpoints would be random branches: "
            f"{missing_active}. Pass --branch-checkpoint for each active cluster, provide "
            "--init-supervised-checkpoint, reduce --active-clusters, or use "
            "--allow-random-active-branches only for debugging."
        )

    missing_reason_cardinality = int(metadata["missing_reason_cardinality"])
    for ckpt_path in branch_checkpoints.values():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_missing_reason_cardinality" in ckpt:
            missing_reason_cardinality = max(missing_reason_cardinality, int(ckpt["model_missing_reason_cardinality"]))
    if init_supervised_state is not None and "model_missing_reason_cardinality" in init_supervised_state:
        missing_reason_cardinality = max(
            missing_reason_cardinality,
            int(init_supervised_state["model_missing_reason_cardinality"]),
        )

    name_embeddings, feature_context_offsets, category_embedding_weights = load_embeddings(token_dir, metadata)
    encoder = ClinicalClusterEncoder(
        cluster_assignments=cluster_assignments,
        name_embeddings=name_embeddings,
        feature_context_offsets=feature_context_offsets,
        feature_type_ids=feature_type_ids,
        categorical_cardinalities=categorical_cardinalities,
        continuous_bin_cardinality=int(metadata["continuous_bin_cardinality"]),
        missing_reason_cardinality=missing_reason_cardinality,
        num_clusters=args.num_clusters,
        d_model=args.d_model,
        expert_n_heads=args.expert_n_heads,
        expert_n_layers=args.expert_n_layers,
        fusion_n_heads=args.fusion_n_heads,
        fusion_n_layers=args.fusion_n_layers,
        dropout=args.dropout,
        categorical_embedding_weights=category_embedding_weights,
        use_cluster_embedding=not args.no_cluster_embedding,
        num_summary_tokens=args.num_summary_tokens,
    ).to(device)

    head = DownstreamPredictionHead(
        d_model=args.d_model,
        n_binary_targets=len(target_columns),
        dropout=args.dropout,
    ).to(device)

    # Preserve the model's intended trainability before applying the global
    # freeze/unfreeze choice. In particular, category lookup tables initialized
    # from text embeddings are intentionally frozen by the model definition and
    # remain frozen even when --train-experts is enabled.
    expert_default_requires_grad = {
        name: bool(param.requires_grad)
        for name, param in encoder.expert_bank.named_parameters()
    }

    if init_supervised_state is not None:
        # Stage-2 fine-tuning starts from the complete best supervised model:
        # expert bank + fusion Transformer + prediction head. We intentionally
        # create a fresh optimizer and LR schedule below rather than resuming them.
        encoder.load_state_dict(init_supervised_state["encoder_state_dict"], strict=True)
        head.load_state_dict(init_supervised_state["head_state_dict"], strict=True)
        print(f"[INFO] Loaded supervised initialization checkpoint: {init_supervised_checkpoint}")
        if branch_checkpoints:
            print(
                "[INFO] --branch-checkpoint entries were provided but are not loaded because "
                "--init-supervised-checkpoint supplies the complete encoder state."
            )
    else:
        load_pretrained_branches(encoder, branch_checkpoints, strict=args.strict_branch_load)

    for name, param in encoder.expert_bank.named_parameters():
        param.requires_grad = bool(
            args.train_experts and expert_default_requires_grad[name]
        )

    # Training/evaluation mode is set per epoch in run_epoch().
    encoder.expert_bank.eval()

    pos_weights = []
    print("[INFO] Binary target counts/pos_weight:")
    for name in target_columns:
        pos = neg = 0.0
        for x in train_ds.targets[name]:
            value, ok = SupervisedTokenizedDataset._label_value_and_mask(x)
            if ok:
                pos += float(int(value) == 1)
                neg += float(int(value) == 0)
        pw = min((neg + args.class_weight_smoothing) / (pos + args.class_weight_smoothing), args.pos_weight_max)
        pos_weights.append(pw)
        print(f"  {name}: pos={int(pos)}, neg={int(neg)}, pos_rate={pos / max(pos + neg, 1):.4f}, pos_weight={pw:.4f}")

    binary_pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=device) if args.binary_loss == "weighted_bce" else None

    # explicit generator makes the shuffled batch order reproducible and identical across Optuna trials.
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Fusion/head always use --lr. When experts are trainable they receive their own smaller --expert-lr. 
    # The cosine scheduler applies the same multiplier to every group, so the learning-rate ratio remains constant throughout training.
    fusion_head_model = nn.ModuleDict({"fusion": encoder.fusion, "head": head})
    optimizer_groups = build_adamw_param_groups(
        fusion_head_model,
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        group_name="fusion_head",
    )

    expert_weight_decay = (
        float(args.weight_decay)
        if args.expert_weight_decay is None
        else float(args.expert_weight_decay)
    )
    if args.train_experts:
        optimizer_groups.extend(
            build_adamw_param_groups(
                encoder.expert_bank,
                learning_rate=float(args.expert_lr),
                weight_decay=expert_weight_decay,
                group_name="experts",
            )
        )

    optimizer = torch.optim.AdamW(optimizer_groups)
    trainable_parameters = [
        param
        for param in list(encoder.parameters()) + list(head.parameters())
        if param.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters were selected.")

    steps_per_epoch = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
    total_train_steps = max(1, int(args.epochs) * int(steps_per_epoch))
    scheduler = None
    warmup_steps = 0
    if args.lr_schedule == "cosine":
        scheduler, warmup_steps = build_warmup_cosine_scheduler(optimizer, total_train_steps, args.warmup_frac, args.min_lr_ratio)

    training_mode = "finetuneExperts" if args.train_experts else "frozenExperts"
    expert_lr_tag = f"_expertLR{args.expert_lr:.2e}" if args.train_experts else ""
    run_name = (
        f"{training_mode}_active{','.join(map(str, active_clusters))}_d{args.d_model}_"
        f"summaryTokens{args.num_summary_tokens}_aux{args.summary_aux_loss}_lam{args.summary_aux_weight:g}_"
        f"bin{args.binary_loss}_gamma{args.focal_gamma}_lr{args.lr:.2e}{expert_lr_tag}_"
        f"wd{args.weight_decay:.2e}_{args.lr_schedule}_warmup{args.warmup_frac}_"
        f"minLR{args.min_lr_ratio}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if trial is not None:
        run_name = f"optuna_trial{trial_number:04d}_" + run_name
    writer = None if args.no_tensorboard else SummaryWriter(Path(args.log_dir) / run_name)

    metric_keys = [
        "loss_total", "loss_optimization", "summary_aux_raw",
        "summary_aux_weighted", "summary_aux_ratio",
        "summary_pairwise_similarity", "summary_conditional_entropy",
        "summary_marginal_entropy", "macro_auroc", "macro_auprc",
        "macro_balanced_acc", "macro_f1", "grad_norm",
    ]
    for name in short_names:
        metric_keys += [f"loss_{name}", f"n_{name}", f"{name}_auroc", f"{name}_auprc", f"{name}_balanced_acc", f"{name}_f1"]

    csv_filename = (
        "supervised_finetune_metrics.csv"
        if args.train_experts
        else "supervised_frozen_metrics.csv"
    )
    csv_path = out_dir / csv_filename
    header = [
        "epoch", "lr", "expert_lr", "weight_decay", "expert_weight_decay",
        "train_experts", "init_supervised_checkpoint", "lr_schedule", "warmup_frac",
        "min_lr_ratio", "binary_loss", "focal_gamma",
        "summary_aux_loss", "summary_aux_weight", "summary_aux_effective_weight",
        "summary_attention_margin", "summary_output_margin",
        "summary_mi_beta", "summary_mi_temperature",
        "selection_metric", "selection_score", "hpo_objective", "hpo_score",
        *[f"train_{k}" for k in metric_keys],
        *[f"val_{k}" for k in metric_keys],
    ]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=header).writeheader()

    print(f"[INFO] device={device}")
    print(f"[INFO] targets={target_columns}")
    print(f"[INFO] active_clusters={active_clusters}")
    print(f"[INFO] ignored_clusters={sorted(ignore_clusters)}")
    print(f"[INFO] summary tokens per expert={args.num_summary_tokens}")
    print(
        f"[INFO] summary auxiliary loss={args.summary_aux_loss}, "
        f"lambda={args.summary_aux_weight:g}, warmup_epochs={args.summary_aux_warmup_epochs}, "
        f"attention_margin={args.summary_attention_margin:g}, "
        f"output_margin={args.summary_output_margin:g}, "
        f"mi_beta={args.summary_mi_beta:g}, mi_temperature={args.summary_mi_temperature:g}"
    )
    print(f"[INFO] fusion receives {len(active_clusters) * args.num_summary_tokens} active group-summary tokens before any cluster dropout")
    print(f"[INFO] ignored_features={int(ignored_feature_mask.sum())}")
    print(f"[INFO] branch_checkpoints={branch_checkpoints}")
    print(f"[INFO] init_supervised_checkpoint={init_supervised_checkpoint}")
    print(f"[INFO] binary_loss={args.binary_loss}, focal_gamma={args.focal_gamma}, pos_weight_max={args.pos_weight_max}")
    print(f"[INFO] fusion/head peak_lr={args.lr:.3g}, weight_decay={args.weight_decay:.3g}")
    if args.train_experts:
        print(f"[INFO] expert peak_lr={args.expert_lr:.3g}, weight_decay={expert_weight_decay:.3g}")
    print(f"[INFO] lr_schedule={args.lr_schedule}, warmup_steps={warmup_steps}/{total_train_steps}, warmup_frac={args.warmup_frac}, min_lr_ratio={args.min_lr_ratio}")
    print(f"[INFO] feature_dropout_prob_max={args.feature_dropout_prob_max}")
    print(f"[INFO] cluster_dropout_prob_max={args.cluster_dropout_prob_max}")
    print(f"[INFO] seed={args.seed}")
    if args.no_cluster_embedding:
        print("[INFO] Using NO cluster embeddings in the fusion transformer.")
    else:
        print("[INFO] Using cluster embeddings in the fusion transformer.")
    if init_supervised_state is None and missing_active:
        print(f"[WARN] Active clusters without pretrained checkpoints will be random branches: {missing_active}")
    if args.train_experts:
        print("[INFO] trainable modules: expert bank + fusion Transformer + binary prediction head")
    else:
        print("[INFO] trainable modules: fusion Transformer + binary prediction head; expert bank frozen")
    print(f"[INFO] trainable parameter count={sum(p.numel() for p in trainable_parameters):,}")

    best_selection_score = float("inf") if args.selection_metric == "loss_total" else -float("inf")
    best_hpo_score = -float("inf")
    best_epoch = 0
    epochs_since_improvement = 0

    common = dict(
        encoder=encoder,
        head=head,
        device=device,
        active_clusters=active_clusters,
        ignored_feature_mask=ignored_feature_mask,
        target_columns=target_columns,
        feature_dropout_prob_max=args.feature_dropout_prob_max,
        cluster_dropout_prob_max=args.cluster_dropout_prob_max,
        binary_loss_type=args.binary_loss,
        binary_pos_weights=binary_pos_weights,
        focal_gamma=args.focal_gamma,
        train_experts=bool(args.train_experts),
        trainable_parameters=trainable_parameters,
        summary_aux_loss=args.summary_aux_loss,
        summary_attention_margin=args.summary_attention_margin,
        summary_output_margin=args.summary_output_margin,
        summary_mi_beta=args.summary_mi_beta,
        summary_mi_temperature=args.summary_mi_temperature,
    )

    if init_supervised_state is not None:
        # Treat the loaded best frozen-expert checkpoint as epoch 0. 
        # This prevents a fine-tuning run from replacing it with a worse epoch merely because the new run initially has no best score.
        baseline_val_logs = run_epoch(
            loader=val_loader,
            optimizer=None,
            scheduler=None,
            train=False,
            max_batches=args.max_val_batches,
            collect_summary_diagnostics=bool(args.summary_diagnostics),
            summary_aux_weight=float(args.summary_aux_weight),
            **common,
        )
        best_selection_score = sanitize_metric(
            args.selection_metric,
            baseline_val_logs[args.selection_metric],
        )
        best_hpo_score = sanitize_metric(
            args.hpo_objective,
            baseline_val_logs[args.hpo_objective],
        )
        best_epoch = 0

        # Keep the initialization checkpoint as best.pt unless a fine-tuning epoch improves the selected validation metric.
        baseline_checkpoint = dict(init_supervised_state)
        baseline_checkpoint["num_summary_tokens"] = int(args.num_summary_tokens)
        baseline_checkpoint["summary_aux_loss"] = args.summary_aux_loss
        baseline_checkpoint["summary_aux_weight"] = float(args.summary_aux_weight)
        baseline_checkpoint["summary_attention_margin"] = float(args.summary_attention_margin)
        baseline_checkpoint["summary_output_margin"] = float(args.summary_output_margin)
        baseline_checkpoint["summary_mi_beta"] = float(args.summary_mi_beta)
        baseline_checkpoint["summary_mi_temperature"] = float(args.summary_mi_temperature)
        baseline_checkpoint["finetune_baseline"] = True
        baseline_checkpoint["init_supervised_checkpoint"] = str(init_supervised_checkpoint)
        baseline_checkpoint["selection_metric"] = args.selection_metric
        baseline_checkpoint["selection_score"] = float(best_selection_score)
        baseline_checkpoint["hpo_objective"] = args.hpo_objective
        baseline_checkpoint["hpo_score"] = float(best_hpo_score)
        baseline_checkpoint["val_loss"] = float(baseline_val_logs["loss_total"])
        baseline_checkpoint["val_macro_auroc"] = float(baseline_val_logs["macro_auroc"])
        baseline_checkpoint["val_macro_auprc"] = float(baseline_val_logs["macro_auprc"])
        torch.save(baseline_checkpoint, out_dir / "best.pt")
        if args.summary_diagnostics and "summary_diagnostics" in baseline_val_logs:
            append_summary_diagnostics_csv(
                baseline_val_logs["summary_diagnostics"],
                output_dir=out_dir,
                epoch=0,
                split="val",
            )
            log_summary_diagnostics_tensorboard(
                writer,
                baseline_val_logs["summary_diagnostics"],
                step=0,
                split="val",
            )

        print(
            f"[INFO] fine-tuning baseline (epoch 0) | "
            f"val_loss {baseline_val_logs['loss_total']:.4f} | "
            f"val_macro_auroc {baseline_val_logs['macro_auroc']:.4f} | "
            f"val_macro_auprc {baseline_val_logs['macro_auprc']:.4f} | "
            f"selection {args.selection_metric}={best_selection_score:.4f}"
        )

    for epoch in range(1, args.epochs + 1):
        aux_scale = (
            1.0 if args.summary_aux_warmup_epochs == 0
            else min(1.0, epoch / float(args.summary_aux_warmup_epochs))
        )
        effective_aux_weight = float(args.summary_aux_weight) * aux_scale
        train_logs = run_epoch(
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            max_batches=args.max_train_batches,
            collect_summary_diagnostics=False,
            summary_aux_weight=effective_aux_weight,
            **common,
        )
        val_logs = run_epoch(
            loader=val_loader,
            optimizer=None,
            scheduler=None,
            train=False,
            max_batches=args.max_val_batches,
            collect_summary_diagnostics=bool(args.summary_diagnostics),
            summary_aux_weight=effective_aux_weight,
            **common,
        )

        lr = get_optimizer_group_value(optimizer, "fusion_head", "lr")
        wd = get_optimizer_group_value(optimizer, "fusion_head_decay", "weight_decay")
        current_expert_lr = (
            get_optimizer_group_value(optimizer, "experts", "lr")
            if args.train_experts
            else float("nan")
        )
        current_expert_wd = (
            get_optimizer_group_value(optimizer, "experts_decay", "weight_decay")
            if args.train_experts
            else float("nan")
        )
        selection_score = sanitize_metric(args.selection_metric, val_logs[args.selection_metric])
        hpo_score = sanitize_metric(args.hpo_objective, val_logs[args.hpo_objective])

        if metric_is_better(args.hpo_objective, hpo_score, best_hpo_score):
            best_hpo_score = hpo_score

        expert_lr_text = (
            f" | expert_lr {current_expert_lr:.2e}"
            if args.train_experts
            else ""
        )
        print(
            f"epoch {epoch:03d} | lr {lr:.2e}{expert_lr_text} | train_sup {train_logs['loss_total']:.4f} | "
            f"train_aux {train_logs['summary_aux_weighted']:.4f} | aux_ratio {train_logs['summary_aux_ratio']:.3f} | "
            f"val_loss {val_logs['loss_total']:.4f} | "
            f"val_macro_auroc {val_logs['macro_auroc']:.4f} | "
            f"val_macro_auprc {val_logs['macro_auprc']:.4f} | "
            f"selection {args.selection_metric}={selection_score:.4f}"
        )

        if writer is not None:
            writer.add_scalars("loss/total", {"train": train_logs["loss_total"], "val": val_logs["loss_total"]}, epoch)
            writer.add_scalars("loss/supervised", {"train": train_logs["loss_total"], "val": val_logs["loss_total"]}, epoch)
            writer.add_scalars("loss/optimization", {"train": train_logs["loss_optimization"], "val": val_logs["loss_optimization"]}, epoch)
            writer.add_scalars("summary_aux/raw", {"train": train_logs["summary_aux_raw"], "val": val_logs["summary_aux_raw"]}, epoch)
            writer.add_scalars("summary_aux/weighted", {"train": train_logs["summary_aux_weighted"], "val": val_logs["summary_aux_weighted"]}, epoch)
            writer.add_scalar("summary_aux/train_ratio_to_supervised", train_logs["summary_aux_ratio"], epoch)
            writer.add_scalar("summary_aux/effective_weight", effective_aux_weight, epoch)
            writer.add_scalars("summary_aux/pairwise_similarity", {"train": train_logs["summary_pairwise_similarity"], "val": val_logs["summary_pairwise_similarity"]}, epoch)
            writer.add_scalars("summary_aux/conditional_entropy", {"train": train_logs["summary_conditional_entropy"], "val": val_logs["summary_conditional_entropy"]}, epoch)
            writer.add_scalars("summary_aux/marginal_entropy", {"train": train_logs["summary_marginal_entropy"], "val": val_logs["summary_marginal_entropy"]}, epoch)
            for name in short_names:
                writer.add_scalars(f"loss/{name}", {"train": train_logs[f"loss_{name}"], "val": val_logs[f"loss_{name}"]}, epoch)
                writer.add_scalars(f"metric/{name}_auroc", {"train": train_logs[f"{name}_auroc"], "val": val_logs[f"{name}_auroc"]}, epoch)
                writer.add_scalars(f"metric/{name}_auprc", {"train": train_logs[f"{name}_auprc"], "val": val_logs[f"{name}_auprc"]}, epoch)
                writer.add_scalars(f"metric/{name}_balanced_acc", {"train": train_logs[f"{name}_balanced_acc"], "val": val_logs[f"{name}_balanced_acc"]}, epoch)
                writer.add_scalars(f"metric/{name}_f1", {"train": train_logs[f"{name}_f1"], "val": val_logs[f"{name}_f1"]}, epoch)
                writer.add_scalars(f"availability/{name}", {"train": train_logs[f"n_{name}"], "val": val_logs[f"n_{name}"]}, epoch)
            for key in ["macro_auroc", "macro_auprc", "macro_balanced_acc", "macro_f1"]:
                writer.add_scalars(f"metric/{key}", {"train": train_logs[key], "val": val_logs[key]}, epoch)
            # Keep optim/lr for backward-compatible TensorBoard dashboards.
            writer.add_scalar("optim/lr", lr, epoch)
            writer.add_scalar("optim/fusion_head_lr", lr, epoch)
            writer.add_scalar("optim/weight_decay", wd, epoch)
            if args.train_experts:
                writer.add_scalar("optim/expert_lr", current_expert_lr, epoch)
                writer.add_scalar("optim/expert_weight_decay", current_expert_wd, epoch)
            writer.add_scalar("optim/grad_norm", train_logs["grad_norm"], epoch)
            writer.add_scalar("selection/score", selection_score, epoch)
            writer.add_scalar(f"hpo/{args.hpo_objective}", hpo_score, epoch)

            writer.add_scalars("cluster_dropout/num_clusters_used_mean", {"train": train_logs["num_clusters_used_mean"], "val": val_logs["num_clusters_used_mean"]}, epoch)
            writer.add_scalars("cluster_dropout/sampled_probability_mean", {"train": train_logs["sampled_cluster_dropout_prob_mean"], "val": val_logs["sampled_cluster_dropout_prob_mean"]}, epoch)
            if args.summary_diagnostics and "summary_diagnostics" in val_logs:
                log_summary_diagnostics_tensorboard(
                    writer,
                    val_logs["summary_diagnostics"],
                    step=epoch,
                    split="val",
                )
            writer.flush()

        row = {
            "epoch": epoch,
            "lr": lr,
            "expert_lr": current_expert_lr,
            "weight_decay": wd,
            "expert_weight_decay": current_expert_wd,
            "train_experts": bool(args.train_experts),
            "init_supervised_checkpoint": str(init_supervised_checkpoint or ""),
            "lr_schedule": args.lr_schedule,
            "warmup_frac": float(args.warmup_frac),
            "min_lr_ratio": float(args.min_lr_ratio),
            "binary_loss": args.binary_loss,
            "focal_gamma": float(args.focal_gamma),
            "summary_aux_loss": args.summary_aux_loss,
            "summary_aux_weight": float(args.summary_aux_weight),
            "summary_aux_effective_weight": float(effective_aux_weight),
            "summary_attention_margin": float(args.summary_attention_margin),
            "summary_output_margin": float(args.summary_output_margin),
            "summary_mi_beta": float(args.summary_mi_beta),
            "summary_mi_temperature": float(args.summary_mi_temperature),
            "selection_metric": args.selection_metric,
            "selection_score": float(selection_score),
            "hpo_objective": args.hpo_objective,
            "hpo_score": float(hpo_score),
        }
        for split, logs in [("train", train_logs), ("val", val_logs)]:
            for key in metric_keys:
                row[f"{split}_{key}"] = logs.get(key, float("nan"))
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)

        if args.summary_diagnostics and "summary_diagnostics" in val_logs:
            append_summary_diagnostics_csv(
                val_logs["summary_diagnostics"],
                output_dir=out_dir,
                epoch=epoch,
                split="val",
            )

        ckpt = {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "lr_schedule": args.lr_schedule,
            "warmup_frac": float(args.warmup_frac),
            "min_lr_ratio": float(args.min_lr_ratio),
            "warmup_steps": int(warmup_steps),
            "total_train_steps": int(total_train_steps),
            "model_missing_reason_cardinality": int(missing_reason_cardinality),
            "num_summary_tokens": int(args.num_summary_tokens),
            "summary_aux_loss": args.summary_aux_loss,
            "summary_aux_weight": float(args.summary_aux_weight),
            "summary_attention_margin": float(args.summary_attention_margin),
            "summary_output_margin": float(args.summary_output_margin),
            "summary_mi_beta": float(args.summary_mi_beta),
            "summary_mi_temperature": float(args.summary_mi_temperature),
            "active_clusters": active_clusters,
            "ignored_clusters": sorted(ignore_clusters),
            "ignored_feature_mask": ignored_feature_mask.cpu(),
            "branch_checkpoints": {k: str(v) for k, v in branch_checkpoints.items()},
            "init_supervised_checkpoint": str(init_supervised_checkpoint or ""),
            "train_experts": bool(args.train_experts),
            "expert_lr": float(args.expert_lr),
            "expert_weight_decay": float(expert_weight_decay),
            "target_columns": target_columns,
            "binary_pos_weights": binary_pos_weights.detach().cpu() if binary_pos_weights is not None else None,
            "selection_metric": args.selection_metric,
            "selection_score": float(selection_score),
            "hpo_objective": args.hpo_objective,
            "hpo_score": float(hpo_score),
            "val_loss": float(val_logs["loss_total"]),
            "val_macro_auroc": float(val_logs["macro_auroc"]),
            "val_macro_auprc": float(val_logs["macro_auprc"]),
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if metric_is_better(args.selection_metric, selection_score, best_selection_score):
            best_selection_score = selection_score
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(ckpt, out_dir / "best.pt")
        else:
            epochs_since_improvement += 1

        if trial is not None:
            trial.report(hpo_score, step=epoch)
            if trial.should_prune():
                if writer is not None:
                    writer.close()
                raise RuntimeError("OPTUNA_PRUNED")

        if args.early_stopping_patience > 0 and epochs_since_improvement >= args.early_stopping_patience:
            print(
                f"[INFO] Early stopping at epoch {epoch}: no improvement in "
                f"{args.selection_metric} for {args.early_stopping_patience} epochs. "
                f"Best epoch was {best_epoch}."
            )
            break

    if writer is not None:
        writer.close()

    # Optuna maximizes hpo_objective. For normal training this return value is not used.
    return float(best_hpo_score)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.hpo:
        run_training(args, device)
        return

    def objective(trial: optuna.Trial) -> float:
        trial_args = argparse.Namespace(**vars(args))
        trial_args.lr = trial.suggest_float("lr", args.optuna_lr_low, args.optuna_lr_high, log=True)
        trial_args.weight_decay = trial.suggest_float("weight_decay", args.optuna_weight_decay_low, args.optuna_weight_decay_high, log=True)
        trial_args.cluster_dropout_prob_max = trial.suggest_float(
            "cluster_dropout_prob_max",
            args.optuna_cluster_dropout_prob_max_low,
            args.optuna_cluster_dropout_prob_max_high,
            step=0.1,
        )
        if trial_args.train_experts:
            trial_args.expert_lr = trial_args.lr
        try:
            return run_training(trial_args, device, trial=trial)
        except RuntimeError as exc:
            if str(exc) == "OPTUNA_PRUNED":
                raise optuna.TrialPruned()
            raise

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    storage = args.optuna_storage or f"sqlite:///{Path(args.output_dir) / 'optuna_supervised.db'}"
    active_tag = str(args.active_clusters).replace(",", "_").replace(" ", "")
    mode_tag = "finetuneExperts" if args.train_experts else "frozenExperts"
    study_name = args.optuna_study_name or f"supervised_{mode_tag}_active{active_tag}_{args.binary_loss}_{args.hpo_objective}"
    sampler = optuna.samplers.TPESampler(
        seed=int(args.optuna_sampler_seed),
        multivariate=True,
        n_startup_trials=int(args.optuna_startup_trials),
    )
    if args.optuna_pruner == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(args.optuna_pruner_startup_trials),
            n_warmup_steps=int(args.optuna_pruner_warmup_steps),
            interval_steps=1,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print("[HPO CONFIG]")
    print(f"  fixed epochs per complete trial: {args.epochs}")
    print(f"  early_stopping_patience: {args.early_stopping_patience}")
    print(f"  objective: {args.hpo_objective}")
    print(f"  lr range: [{args.optuna_lr_low}, {args.optuna_lr_high}] log-scale")
    print(f"  weight_decay range: [{args.optuna_weight_decay_low}, {args.optuna_weight_decay_high}] log-scale")
    print(f"  cluster_dropout_prob_max range: [{args.optuna_cluster_dropout_prob_max_low}, {args.optuna_cluster_dropout_prob_max_high}] step=0.1")
    print(f"  pruner: {args.optuna_pruner}, warmup_epochs={args.optuna_pruner_warmup_steps}")
    print(f"  training seed (same for all trials): {args.seed}")
    print(f"  sampler seed: {args.optuna_sampler_seed}")

    study.optimize(objective, n_trials=args.optuna_trials)

    print("Best trial:")
    print(f"  value: {study.best_trial.value}")
    print(f"  params: {study.best_trial.params}")
    print(f"  storage: {storage}")
    print(f"  study_name: {study.study_name}")


if __name__ == "__main__":
    main()
