""" Masked value-reconstruction pretraining for one SGMET expert encoder branch for one semantic feature group .

Masking semantics used here
---------------------------
The script keeps three different concepts separate:

1. Patient-level missingness
   `missing_mask=True` means the value for this patient/feature is naturally
   missing in the tokenized dataset. The feature token is still allowed to be
   attended over because its feature identity and missing-reason code can be
   informative.

2. Schema availability / schema dropout
   `schema_available_mask=False` means the feature is unavailable as model
   evidence. This can come from `observed_mask` in the tokenized data and/or
   from batch-level schema dropout. Schema-dropped features are not attended to
   and are not reconstruction targets.

3. Artificial reconstruction masking
   `artificial_masked_mask=True` means an originally observed value is hidden
   from the encoder and used as a reconstruction target. Only non-missing,
   schema-available values can become reconstruction targets.

Training flow
-------------
original local batch
    -> schema dropout removes whole feature columns from encoder evidence
    -> choose reconstruction targets only among remaining non-missing values
    -> corrupt target value channels
    -> encoder attends to schema-available features except reconstruction targets
       including true patient-missing tokens
    -> decoder predicts only artificially masked numerical/categorical values

Example usage: see scripts/pretrain_expert_reconstruction.sh

tensorboard --logdir=runs/...
optuna-dashboard sqlite:///outputs/...db
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Mapping
try:
    import optuna
except ModuleNotFoundError:  # Optional until --hpo is requested.
    optuna = None  # type: ignore[assignment]

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm  

from src.tokenizer.dataset import TokenizedTabularDataset
from src.tokenizer.schema import FeatureType
from src.clinical_cluster_experts.model import ClusterBranch, TOKEN_BATCH_KEYS
from src.clinical_cluster_experts.utils import (
    load_cluster_assignments,
    load_embeddings,
    move_batch_to_device,
)
from src.clinical_cluster_experts.summary_auxiliary_losses import (
    SUMMARY_AUX_LOSS_CHOICES,
    compute_summary_auxiliary_loss,
)


CAT_LOSS_CHOICES = ("ce", "weighted_ce", "focal", "weighted_focal")


@dataclass(frozen=True)
class MaskingResult:
    """Container for all masks produced for one local cluster batch.

    Attributes:
        corrupted_batch:
            Copy of the local token batch after artificial reconstruction targets
            have been corrupted in their value channels.
        has_value_mask:
            True where the original cell has a real value, i.e. not patient-missing.
        base_schema_available_mask:
            Initial feature-availability mask, usually loaded from observed_mask.
            In the current NHANES tokenized data this is typically all True.
        schema_available_mask:
            Availability mask after optional schema dropout.
        schema_dropped_mask:
            True where schema dropout removed an otherwise available feature.
        target_candidate_mask:
            Non-missing values that remain schema-available and are therefore
            eligible to become reconstruction targets.
        artificial_masked_mask:
            Values actually hidden and used as reconstruction targets.
        encoder_available_mask:
            Tokens visible to the cluster Transformer. This equals
            schema_available_mask & ~artificial_masked_mask.
        actual_schema_dropout_prob:
            Batch-level schema dropout probability sampled from
            [0, schema_dropout_prob_max].
        actual_feature_mask_prob:
            Batch-level feature-mask probability sampled from
            [0, feature_mask_prob_max].
    """

    corrupted_batch: dict[str, torch.Tensor]
    has_value_mask: torch.Tensor
    base_schema_available_mask: torch.Tensor
    schema_available_mask: torch.Tensor
    schema_dropped_mask: torch.Tensor
    target_candidate_mask: torch.Tensor
    artificial_masked_mask: torch.Tensor
    encoder_available_mask: torch.Tensor
    actual_schema_dropout_prob: float
    actual_feature_mask_prob: float


class ClusterMAEReconstructionHead(nn.Module):
    """CLS-conditioned decoder for masked numerical/categorical value prediction.
    The decoder receives: decoder([h_CLS, q_ij, h_CLS * q_ij])
    - where q_ij is produced by FeatureTokenBuilder.metadata_tokens =  context-specific feature semantic token + the feature type token.

    The decoder predicts:
        - numerical values for numerical features 
        - categorical codes for categorical features
    """

    def __init__(
        self,
        *,
        feature_type_ids: torch.Tensor,
        categorical_cardinalities: list[int],
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_type_ids", feature_type_ids.long(), persistent=True)
        self.decoder = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.numeric_head = nn.Linear(d_model, 1) # outputs a single reconstructed value for numerical features
        self.categorical_heads = nn.ModuleList( # outputs logits for each categorical feature, with output dim equal to the feature's cardinality
            [nn.Linear(d_model, max(int(card), 1)) for card in categorical_cardinalities]
        )

    def decode(
        self,
        h_cls: torch.Tensor,
        local_feature_idx: torch.Tensor,
        token_builder: nn.Module,
        query_batch: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Decode masked features from one or multiple expert summaries.

        With multiple summarie tokens, each masked feature query performs parameter-free dot-product attention over the M summaries before the existing decoder MLP. 
        This keeps all summary tokens involved in pretraining without immediately averaging them into a fixed single vector.
        """
        row_idx = local_feature_idx[:, 0]
        col_idx = local_feature_idx[:, 1]

        feature_context_codes = None
        if query_batch is not None and "feature_context_codes" in query_batch:
            feature_context_codes = query_batch["feature_context_codes"]

        batch_size = int(h_cls.shape[0])
        q_all = token_builder.metadata_tokens(
            batch_size=batch_size,
            device=h_cls.device,
            feature_context_codes=feature_context_codes,
        )
        q = q_all[row_idx, col_idx]

        if h_cls.ndim == 2:
            # Original single-CLS path: [B, d].
            h = h_cls[row_idx]
        elif h_cls.ndim == 3:
            # Multi-summary path: [B, M, d]. Each reconstruction query selects its own weighted combination of the expert summaries.
            h_candidates = h_cls[row_idx]  # [n_masked, M, d]
            attention_logits = (h_candidates * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(h_cls.shape[-1]))
            attention_weights = torch.softmax(attention_logits, dim=1)
            h = torch.sum(
                attention_weights.unsqueeze(-1) * h_candidates,
                dim=1,
            )
        else:
            raise ValueError(f"h_cls must have shape [B, d] or [B, M, d], got {tuple(h_cls.shape)}.")

        return self.decoder(torch.cat([h, q, h * q], dim=-1))


def _validate_probability(name: str, value: float) -> None:
    """Validate a probability-like CLI value."""
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _sample_probability_upper_bound(
    max_prob: float,
    *,
    device: torch.device,
    sample: bool = True,
) -> float:
    """Return either `max_prob` or a random probability sampled from [0, max_prob]."""
    max_prob = float(max_prob)
    _validate_probability("max_prob", max_prob)
    if max_prob <= 0.0:
        return 0.0
    if not sample:
        return max_prob
    return float((torch.rand((), device=device) * max_prob).detach().cpu())


def prepare_masked_reconstruction_batch(
    local_batch: Mapping[str, torch.Tensor],
    *,
    feature_mask_prob_max: float,
    schema_dropout_prob_max: float,
    mask_reason_code: int,
    sample_probabilities: bool = True,
    force_target_per_row: bool = True,
    keep_one_observed_value_visible: bool = True,
) -> MaskingResult:
    """Apply schema dropout and artificial value masking to one local batch.

    Args:
        local_batch:
            Local cluster batch with shape [B, A_c] tensors.
        feature_mask_prob_max:
            Upper bound for the per-batch artificial feature-mask probability.
            The actual probability is sampled uniformly from
            [0, feature_mask_prob_max] when sample_probabilities=True.
        schema_dropout_prob_max:
            Upper bound for the per-batch schema-dropout probability. Schema
            dropout removes whole feature columns from encoder evidence and
            from reconstruction-target eligibility.
        mask_reason_code:
            Missing-reason code assigned to artificial reconstruction masks.
        sample_probabilities:
            If True, sample actual probabilities from [0, max]. If False, use
            the provided max values directly. Tests may set this to False.
        force_target_per_row:
            If True, rows with enough eligible target candidates are forced to
            contribute at least one reconstruction target.
        keep_one_observed_value_visible:
            If True, the function avoids masking the last true observed value
            in a row whenever possible. This keeps at least one real-valued
            context feature visible after artificial masking. If schema dropout
            already makes this impossible, the row is accepted and logged.

    Returns:
        MaskingResult with corrupted input channels and all masks used for
        logging, loss target selection, and encoder attention.
    """
    corrupted = {k: v.clone() for k, v in local_batch.items() if k in TOKEN_BATCH_KEYS}
    if "feature_context_codes" in local_batch:
        # Metadata, not a patient value. Keep it unchanged.
        corrupted["feature_context_codes"] = local_batch["feature_context_codes"].clone()

    device = corrupted["numeric_values"].device
    B, A = corrupted["numeric_values"].shape

    # Patient-level missingness. A true value exists only where missing_mask=False.
    has_value_mask = ~corrupted["missing_mask"].to(device=device, dtype=torch.bool)

    # Base schema availability mask. This is usually all True in the current NHANES tokenized data.
    if "observed_mask" in local_batch:
        base_schema_available_mask = local_batch["observed_mask"].to(device=device, dtype=torch.bool).clone()
    else:
        base_schema_available_mask = torch.ones(B, A, dtype=torch.bool, device=device)

    # Schema dropout removes whole feature columns from encoder evidence.
    actual_schema_dropout_prob = _sample_probability_upper_bound(schema_dropout_prob_max, device=device, sample=sample_probabilities)
    if actual_schema_dropout_prob > 0.0 and A > 0:
        dropped_columns = torch.rand(A, device=device) < actual_schema_dropout_prob
    else:
        dropped_columns = torch.zeros(A, dtype=torch.bool, device=device)

    schema_available_mask = base_schema_available_mask & ~dropped_columns.unsqueeze(0)
    schema_dropped_mask = base_schema_available_mask & ~schema_available_mask

    # Only schema-available, originally non-missing values can become reconstruction targets. 
    # Patient-missing values remain visible as missingness-information tokens, but are not targets.
    target_candidate_mask = schema_available_mask & has_value_mask
    actual_feature_mask_prob = _sample_probability_upper_bound(feature_mask_prob_max, device=device, sample=sample_probabilities)
    artificial_masked_mask = target_candidate_mask & (torch.rand(B, A, device=device) < actual_feature_mask_prob)

    # Encourage each row to have at least one target when doing so does not destroy the last real-valued context feature. 
    # If only one true observed value remains after schema dropout, it is left visible by default.
    if force_target_per_row:
        for i in range(B):
            cand = torch.where(target_candidate_mask[i])[0]
            if cand.numel() == 0:
                continue

            if keep_one_observed_value_visible and cand.numel() < 2:
                # There is no way to both mask a value and keep a true observed value as context. 
                # Prefer keeping context and let this row contribute no reconstruction target.
                artificial_masked_mask[i, cand] = False
                continue

            if not artificial_masked_mask[i].any():
                j = cand[torch.randint(cand.numel(), (1,), device=device)]
                artificial_masked_mask[i, j] = True

            if keep_one_observed_value_visible and artificial_masked_mask[i, cand].all():
                # Unmask one candidate so at least one true observed value remains visible to the encoder.
                j = cand[torch.randint(cand.numel(), (1,), device=device)]
                artificial_masked_mask[i, j] = False

    # Corrupt only artificial reconstruction targets. Schema-dropped features are not corrupted because they are not attended to and not targets.
    corrupted["numeric_values"][artificial_masked_mask] = 0.0
    corrupted["continuous_bin_codes"][artificial_masked_mask] = 0
    corrupted["categorical_codes"][artificial_masked_mask] = 0
    corrupted["missing_mask"][artificial_masked_mask] = True
    corrupted["missing_reason_codes"][artificial_masked_mask] = int(mask_reason_code)

    # Encoder sees schema-available tokens except hidden reconstruction targets.
    # This includes true patient-missing tokens if their feature is schema-available.
    encoder_available_mask = schema_available_mask & ~artificial_masked_mask

    return MaskingResult(
        corrupted_batch=corrupted,
        has_value_mask=has_value_mask,
        base_schema_available_mask=base_schema_available_mask,
        schema_available_mask=schema_available_mask,
        schema_dropped_mask=schema_dropped_mask,
        target_candidate_mask=target_candidate_mask,
        artificial_masked_mask=artificial_masked_mask,
        encoder_available_mask=encoder_available_mask,
        actual_schema_dropout_prob=actual_schema_dropout_prob,
        actual_feature_mask_prob=actual_feature_mask_prob,
    )


def categorical_loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_type: str,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """ 
    Categorical reconstruction loss for one categorical feature head.
    Currently, four possible loss options: CAT_LOSS_CHOICES
    """
    if loss_type not in CAT_LOSS_CHOICES:
        raise ValueError(f"Unknown cat loss: {loss_type}. Expected one of {CAT_LOSS_CHOICES}.")

    weights = class_weights.to(logits.device) if class_weights is not None else None

    if loss_type == "ce":
        return F.cross_entropy(logits, targets)

    if loss_type == "weighted_ce":
        return F.cross_entropy(logits, targets, weight=weights)

    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp().clamp(min=1e-8, max=1.0)
    ce = -log_pt

    if loss_type == "weighted_focal" and weights is not None:
        ce = ce * weights[targets]

    focal = (1.0 - pt).pow(float(focal_gamma)) * ce
    return focal.mean()


def reconstruction_loss(
    *,
    decoder: ClusterMAEReconstructionHead,
    h_cls: torch.Tensor,
    token_builder: nn.Module,
    original_local_batch: Mapping[str, torch.Tensor],
    mask_positions: torch.Tensor,
    categorical_weight: float = 1.0,
    cat_loss: str = "ce",
    cat_class_weights: list[torch.Tensor | None] | None = None,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked numerical/categorical value reconstruction loss.

    Numerical and categorical reconstruction losses are computed separately.
    If both are present, the optimized loss is the mean of:
        numeric_mse_raw
        categorical_weight * categorical_loss_raw
    """
    masked_idx = mask_positions.nonzero(as_tuple=False)
    logs = {
        "loss_num": 0.0,
        "loss_cat": 0.0,
        "num_rmse": 0.0,
        "cat_acc_macro_feature": 0.0,
        "n_masked": float(masked_idx.shape[0]),
        "n_masked_num": 0.0,
        "n_masked_cat": 0.0,
    }
    if masked_idx.numel() == 0:
        return h_cls.sum() * 0.0, logs

    d = decoder.decode(
        h_cls=h_cls,
        local_feature_idx=masked_idx,
        token_builder=token_builder,
        query_batch=original_local_batch,
    )
    row_idx, col_idx = masked_idx[:, 0], masked_idx[:, 1]
    type_ids = decoder.feature_type_ids.to(h_cls.device)[col_idx]
    losses: list[torch.Tensor] = []

    num_mask = type_ids == int(FeatureType.NUMERICAL)
    if num_mask.any():
        pred_num = decoder.numeric_head(d[num_mask]).squeeze(-1)
        target_num = original_local_batch["numeric_values"][row_idx[num_mask], col_idx[num_mask]].float()
        loss_num = F.mse_loss(pred_num, target_num)

        losses.append(loss_num)
        logs["loss_num"] = float(loss_num.detach().cpu())
        logs["num_rmse"] = float(torch.sqrt(loss_num.detach()).cpu())
        logs["n_masked_num"] = float(num_mask.sum().detach().cpu())

    cat_mask = type_ids == int(FeatureType.CATEGORICAL)
    if cat_mask.any():
        cat_losses = []
        correct = 0.0
        total = 0.0
        per_feature_acc = []

        for local_j in torch.unique(col_idx[cat_mask]).tolist():
            local_j = int(local_j)
            pos = cat_mask & (col_idx == local_j)
            logits = decoder.categorical_heads[local_j](d[pos])
            targets = original_local_batch["categorical_codes"][row_idx[pos], col_idx[pos]].long()
            class_weights = None if cat_class_weights is None else cat_class_weights[local_j]
            cat_losses.append(
                categorical_loss_fn(
                    logits,
                    targets,
                    loss_type=cat_loss,
                    class_weights=class_weights,
                    focal_gamma=focal_gamma,
                )
            )

            pred = logits.argmax(dim=-1)
            acc_j = (pred == targets).float().mean()
            per_feature_acc.append(acc_j)
            correct += float((pred == targets).sum().detach().cpu())
            total += float(targets.numel())

        loss_cat = torch.stack(cat_losses).mean()
        losses.append(float(categorical_weight) * loss_cat)
        logs["loss_cat"] = float(loss_cat.detach().cpu())
        logs["cat_acc_macro_feature"] = float(torch.stack(per_feature_acc).mean().detach().cpu())
        logs["n_masked_cat"] = float(cat_mask.sum().detach().cpu())

    loss = h_cls.sum() * 0.0 if not losses else torch.stack(losses).mean()
    return loss, logs


def compute_local_categorical_class_weights(
    *,
    dataset: TokenizedTabularDataset,
    feature_indices: torch.Tensor,
    feature_type_ids: torch.Tensor,
    categorical_cardinalities: list[int],
    batch_size: int,
    smoothing: float,
    max_weight: float,
) -> list[torch.Tensor | None]:
    """
    Only used for weighted_ce and weighted_focal losses. 
    For each feature k, compute per-feature weighting weight_k with smoothing, clipping, and mean-normalization. 
    """

    local_weights: list[torch.Tensor | None] = [None for _ in range(int(feature_indices.numel()))]
    counts_by_local: dict[int, torch.Tensor] = {}

    for local_j, global_j in enumerate(feature_indices.tolist()):
        if int(feature_type_ids[int(global_j)]) == int(FeatureType.CATEGORICAL):
            card = max(int(categorical_cardinalities[int(global_j)]), 1)
            counts_by_local[local_j] = torch.zeros(card, dtype=torch.float64)

    if not counts_by_local:
        return local_weights

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    idx = feature_indices.cpu()

    for batch in loader:
        categorical_codes = batch["categorical_codes"].index_select(1, idx).long()
        observed = ~batch["missing_mask"].index_select(1, idx).bool()
        if "observed_mask" in batch:
            observed = observed & batch["observed_mask"].index_select(1, idx).bool()

        for local_j, counts in counts_by_local.items():
            vals = categorical_codes[:, local_j][observed[:, local_j]]
            if vals.numel() == 0:
                continue
            valid = (vals >= 0) & (vals < counts.numel())
            vals = vals[valid]
            if vals.numel() == 0:
                continue
            counts += torch.bincount(vals.cpu(), minlength=counts.numel()).double()

    for local_j, counts in counts_by_local.items():
        smoothed = counts + float(smoothing) # additive smoothing for inverse-frequency safety and to avoid zero weights
        weights = smoothed.sum() / (float(smoothed.numel()) * smoothed) # inverse frequency weighting
        weights = torch.clamp(weights, max=float(max_weight)) # clip max weight to avoid extreme outliers dominating the loss
        weights = weights / weights.mean().clamp_min(1e-12) # mean-normalize weights so that the average weight is 1.0, preventing overall loss scale changes that would affect optimization dynamics
        local_weights[local_j] = weights.float()

    return local_weights


def summarize_masking_result(masking: MaskingResult) -> dict[str, float]:
    """Return scalar logging metrics for schema/value masking."""
    schema_available_counts = masking.schema_available_mask.sum(dim=1).float()
    #schema_dropped_counts = masking.schema_dropped_mask.sum(dim=1).float()
    target_candidate_counts = masking.target_candidate_mask.sum(dim=1).float()
    artificial_masked_counts = masking.artificial_masked_mask.sum(dim=1).float()
    encoder_available_counts = masking.encoder_available_mask.sum(dim=1).float()
    visible_observed_value_counts = (masking.encoder_available_mask & masking.has_value_mask).sum(dim=1).float()
    #total_candidates = float(target_candidate_counts.sum().detach().cpu())
    #total_masked = float(artificial_masked_counts.sum().detach().cpu())

    return {
        "schema_available_mean": float(schema_available_counts.mean().detach().cpu()),
        "artificial_masked_mean": float(artificial_masked_counts.mean().detach().cpu()),
        "encoder_available_mean": float(encoder_available_counts.mean().detach().cpu()),
        "visible_observed_value_mean": float(visible_observed_value_counts.mean().detach().cpu()),
        "actual_schema_dropout_prob": float(masking.actual_schema_dropout_prob),
        "actual_feature_mask_prob": float(masking.actual_feature_mask_prob),
        "fraction_no_encoder_context": float((encoder_available_counts <= 0).float().mean().detach().cpu()),
        "fraction_no_visible_observed_value": float((visible_observed_value_counts <= 0).float().mean().detach().cpu()),
    }



def build_adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Build Transformer-style AdamW parameter groups.
    Weight decay is applied only to regular weight matrices. 
    The whole token builder, biases, normalization parameters, embedding-like parameters, CLS tokens, and frozen parameters are excluded from weight decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        no_decay = (
            "token_builder" in name_lower
            or "embedding" in name_lower
            or "cls_token" in name_lower
            or name_lower.endswith(".bias")
            or "norm" in name_lower
            or "layernorm" in name_lower
            or param.ndim < 2
        )
        if no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def run_epoch(
    branch: ClusterBranch,
    decoder: ClusterMAEReconstructionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    feature_mask_prob_max: float,
    schema_dropout_prob_max: float,
    mask_reason_code: int,
    train: bool,
    max_batches: int = 0,
    categorical_weight: float = 1.0,
    cat_loss: str = "ce",
    cat_class_weights: list[torch.Tensor | None] | None = None,
    focal_gamma: float = 2.0,
    force_target_per_row: bool = True,
    keep_one_observed_value_visible: bool = True,
    # optional multi-summary auxiliary loss.
    summary_aux_loss: str = "none",
    summary_aux_weight: float = 0.0,
    summary_attention_margin: float = 0.9,
    summary_output_margin: float = 0.9,
    summary_mi_beta: float = 1.0,
    summary_mi_temperature: float = 1.0,
) -> dict[str, float | torch.Tensor]:
    """Run one training or validation epoch for one cluster branch."""
    branch.train(train)
    decoder.train(train)

    scalar_keys = [
        "loss", "loss_num", "loss_cat",
        "num_rmse", "cat_acc_macro_feature",
        "n_masked", "n_masked_num", "n_masked_cat",
        "actual_schema_dropout_prob", "schema_available_mean",
        "actual_feature_mask_prob", "artificial_masked_mean",
        "encoder_available_mean", "visible_observed_value_mean",
        "fraction_no_encoder_context", "fraction_no_visible_observed_value",
        # track the optional auxiliary objective.
        "loss_optimization", "summary_aux_raw", "summary_aux_weighted", "summary_aux_ratio",
        "summary_pairwise_similarity", "summary_conditional_entropy", "summary_marginal_entropy",
        "grad_norm",
    ]
    totals = {k: 0.0 for k in scalar_keys}
    totals["n_batches"] = 0.0

    encoder_available_chunks = []
    visible_value_chunks = []

    idx = branch.feature_indices.to(device)
    total_batches = min(len(loader), max_batches) if max_batches else len(loader)
    iterable = islice(loader, max_batches) if max_batches else loader
    iterator = enumerate(iterable)
    iterator = tqdm(iterator, total=total_batches, desc="train" if train else "val", leave=False)

    for _, batch in iterator:
        batch = move_batch_to_device(batch, device)

        local_batch = {k: batch[k].index_select(1, idx) for k in TOKEN_BATCH_KEYS}
        if "observed_mask" in batch:
            local_batch["observed_mask"] = batch["observed_mask"].index_select(1, idx)
        if "feature_context_codes" in batch:
            local_batch["feature_context_codes"] = batch["feature_context_codes"].index_select(1, idx)

        masking = prepare_masked_reconstruction_batch(
            local_batch,
            feature_mask_prob_max=feature_mask_prob_max,
            schema_dropout_prob_max=schema_dropout_prob_max,
            mask_reason_code=mask_reason_code,
            sample_probabilities=True,
            force_target_per_row=force_target_per_row,
            keep_one_observed_value_visible=keep_one_observed_value_visible,
        )
        mask_positions = masking.artificial_masked_mask
        if not mask_positions.any():
            # This can happen when schema dropout removes all true observed
            # target candidates, or when each row has only one true observed
            # value and we keep the last value visible as context.
            continue

        mask_logs = summarize_masking_result(masking)
        encoder_available_chunks.append(masking.encoder_available_mask.sum(dim=1).detach().cpu().float())
        visible_value_chunks.append(
            (masking.encoder_available_mask & masking.has_value_mask).sum(dim=1).detach().cpu().float()
        )

        with torch.set_grad_enabled(train):
            assert branch.token_builder is not None
            z = branch.token_builder(masking.corrupted_batch)

            # attention-based losses need differentiable CLS->feature attention. output_margin and none keep the original cheaper expert forward path.
            need_summary_attention = summary_aux_loss in {"attention_margin", "attention_mi"}
            if need_summary_attention:
                h_cls, _, feature_attention = branch.expert(z, masking.encoder_available_mask, return_attention_overlap=True)
            else:
                h_cls = branch.expert(z, masking.encoder_available_mask)
                feature_attention = None

            loss, logs = reconstruction_loss(
                decoder=decoder,
                h_cls=h_cls,
                token_builder=branch.token_builder,
                original_local_batch=local_batch,
                mask_positions=mask_positions,
                categorical_weight=categorical_weight,
                cat_loss=cat_loss,
                cat_class_weights=cat_class_weights,
                focal_gamma=focal_gamma,
            )

            # reuse exactly the same auxiliary-loss dispatcher as supervised training. 
            # One pretrained expert is represented as K=1 for the shared [B,K,M,D] interface.
            if h_cls.ndim == 2:
                group_tokens = h_cls.unsqueeze(1).unsqueeze(2)
            else:
                group_tokens = h_cls.unsqueeze(1)
            cluster_available_mask = masking.encoder_available_mask.any(dim=1, keepdim=True)
            attention_by_cluster = None if feature_attention is None else {0: feature_attention}
            feature_mask_by_cluster = None if feature_attention is None else {0: masking.encoder_available_mask}
            aux = compute_summary_auxiliary_loss(
                summary_aux_loss,
                group_tokens=group_tokens,
                cluster_available_mask=cluster_available_mask,
                attention_by_cluster=attention_by_cluster,
                feature_mask_by_cluster=feature_mask_by_cluster,
                attention_margin=summary_attention_margin,
                output_margin=summary_output_margin,
                mi_beta=summary_mi_beta,
                mi_temperature=summary_mi_temperature,
            )
            weighted_aux = float(summary_aux_weight) * aux["loss"]
            optimization_loss = loss + weighted_aux

            grad_norm = 0.0
            if train:
                optimizer.zero_grad(set_to_none=True)
                optimization_loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        list(branch.parameters()) + list(decoder.parameters()),
                        1.0,
                    ).detach().cpu()
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        loss_value = float(loss.detach().cpu())
        aux_raw_value = float(aux["loss"].detach().cpu())
        aux_weighted_value = float(weighted_aux.detach().cpu())
        totals["loss"] += loss_value
        totals["loss_optimization"] += float(optimization_loss.detach().cpu())
        totals["summary_aux_raw"] += aux_raw_value
        totals["summary_aux_weighted"] += aux_weighted_value
        totals["summary_aux_ratio"] += abs(aux_weighted_value) / max(abs(loss_value), 1e-12)
        totals["summary_pairwise_similarity"] += float(aux["pairwise_similarity"].detach().cpu())
        totals["summary_conditional_entropy"] += float(aux["conditional_entropy"].detach().cpu())
        totals["summary_marginal_entropy"] += float(aux["marginal_entropy"].detach().cpu())
        for k in [
            "loss_num", "loss_cat",
            "num_rmse", "cat_acc_macro_feature",
            "n_masked", "n_masked_num", "n_masked_cat",
        ]:
            totals[k] += float(logs[k])

        for k, v in mask_logs.items():
            totals[k] += float(v)
        totals["grad_norm"] += grad_norm
        totals["n_batches"] += 1.0

        iterator.set_postfix(
            loss=f"{loss_value:.4f}",
            aux=f"{aux_weighted_value:.4f}",
            masked=int(logs["n_masked"]),
            enc_ctx=f"{mask_logs['encoder_available_mean']:.1f}",
            obs_ctx=f"{mask_logs['visible_observed_value_mean']:.1f}",
        )

    denom = max(totals["n_batches"], 1.0)
    out = {
        k: (v / denom if k not in {"n_masked", "n_masked_num", "n_masked_cat"} else v)
        for k, v in totals.items()
        if k != "n_batches"
    }

    if encoder_available_chunks:
        encoder_counts = torch.cat(encoder_available_chunks).float()
        visible_counts = torch.cat(visible_value_chunks).float()
        for prefix, counts in [
            ("encoder_available", encoder_counts),
            ("visible_observed_value", visible_counts),
        ]:
            out[f"{prefix}_min"] = float(counts.min())
            out[f"{prefix}_p10"] = float(torch.quantile(counts, 0.10))
            out[f"{prefix}_p50"] = float(torch.quantile(counts, 0.50))
            #out[f"{prefix}_p90"] = float(torch.quantile(counts, 0.90))
        out["hist_encoder_available_count_per_patient"] = encoder_counts
        out["hist_visible_observed_value_count_per_patient"] = visible_counts
    else:
        for prefix in ["encoder_available", "visible_observed_value"]:
            out[f"{prefix}_min"] = 0.0
            out[f"{prefix}_p10"] = 0.0
            out[f"{prefix}_p50"] = 0.0
            #out[f"{prefix}_p90"] = 0.0
        out["hist_encoder_available_count_per_patient"] = torch.empty(0)
        out["hist_visible_observed_value_count_per_patient"] = torch.empty(0)

    return out



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Masked feature-value reconstruction pretraining for one clinical cluster branch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token-dir", type=str, default="data/processed/tokenized_nhanes")
    parser.add_argument("--cluster-csv", type=str, required=True)
    parser.add_argument("--num-clusters", type=int, required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-summary-tokens", type=int, default=1,help=("Number of independently initialized CLS summary tokens in this expert."))
    parser.add_argument("--summary-aux-loss", type=str, default="none", choices=SUMMARY_AUX_LOSS_CHOICES, help="Enable exactly one summary-token auxiliary loss, or none.")
    parser.add_argument("--summary-aux-weight", type=float, default=0.0, help="Lambda multiplying the selected summary auxiliary loss.")
    parser.add_argument("--summary-aux-warmup-epochs", type=int, default=5, help="Linearly ramp lambda to its full value over this many epochs; 0 disables the ramp.")
    parser.add_argument("--summary-attention-margin", type=float, default=0.9, help="Maximum unpenalized cosine overlap for attention_margin.")
    parser.add_argument("--summary-output-margin", type=float, default=0.9, help="Maximum unpenalized output cosine similarity for output_margin.")
    parser.add_argument("--summary-mi-beta", type=float, default=1.0, help="Weight of marginal slot-usage entropy in attention_mi.")
    parser.add_argument("--summary-mi-temperature", type=float, default=1.0, help="Temperature for competitive p(slot|feature) in attention_mi.")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"], help="Learning-rate schedule. 'cosine' uses linear warmup followed by cosine decay.")
    parser.add_argument("--warmup-frac", type=float, default=0.05, help="Fraction of training update steps used for linear LR warmup when --lr-schedule cosine.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.05, help="Final LR as a fraction of --lr after cosine decay.")
    parser.add_argument("--feature-mask-prob-max", type=float, default=0.25,
        help=("Upper bound for the batch-level artificial feature-value mask probability. Each batch samples p_feature ~ Uniform(0, feature_mask_prob_max). "
            "This masks patient-feature cells with real values, not whole schema columns."))
    parser.add_argument("--schema-dropout-prob-max", type=float, default=0.0,
        help=("Upper bound for batch-level schema dropout. Each batch samples p_schema ~ Uniform(0, schema_dropout_prob_max), then removes whole "
            "local feature columns from encoder evidence and target eligibility."))
    parser.add_argument("--mask-reason-code", type=int, default=-1)
    parser.add_argument("--no-force-target-per-row", action="store_true")
    parser.add_argument("--allow-mask-last-observed-value", action="store_true")
    parser.add_argument("--categorical-weight", type=float, default=1.0) # Chosen based on observed loss_cat and loss_num 
    parser.add_argument("--cat-loss", type=str, default="focal", choices=CAT_LOSS_CHOICES)
    parser.add_argument("--cat-weight-smoothing", type=float, default=1.0)
    parser.add_argument("--cat-weight-max", type=float, default=5.0)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--val-mask-passes", type=int, default=3, help="Average validation loss over this many deterministic masking passes.")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=str, default="outputs/pretrained_expert_encoders")
    parser.add_argument("--log-dir", type=str, default="runs/pretrain_expert_encoders")
    parser.add_argument("--no-tensorboard", action="store_true")

    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO over lr and weight_decay instead of one fixed run.")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--optuna-study-name", type=str, default=None)
    parser.add_argument("--optuna-storage", type=str, default=None, help="Optuna storage URL. Defaults to sqlite:///<output-dir>/optuna_cluster_<id>.db")
    parser.add_argument("--optuna-lr-low", type=float, default=1e-4)
    parser.add_argument("--optuna-lr-high", type=float, default=1e-2)
    parser.add_argument("--optuna-weight-decay-low", type=float, default=1e-7)
    parser.add_argument("--optuna-weight-decay-high", type=float, default=1e-3)
    parser.add_argument("--optuna-startup-trials", type=int, default=8)
    parser.add_argument("--optuna-pruner-startup-trials", type=int, default=5)
    parser.add_argument("--optuna-pruner-warmup-steps", type=int, default=5)
    args = parser.parse_args()

    _validate_probability("feature_mask_prob_max", args.feature_mask_prob_max)
    _validate_probability("schema_dropout_prob_max", args.schema_dropout_prob_max)
    _validate_probability("warmup_frac", args.warmup_frac)
    _validate_probability("min_lr_ratio", args.min_lr_ratio)
    if args.val_mask_passes < 1:
        raise ValueError("--val-mask-passes must be >= 1.")
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
        if args.summary_aux_weight <= 0:
            raise ValueError("Set --summary-aux-weight > 0 when a summary auxiliary loss is enabled.")
    if args.d_model % args.n_heads != 0:
        raise ValueError("--d-model must be divisible by --n-heads.")
    return args

def set_seed(seed: int) -> None:
    """Set PyTorch RNG seeds used by masking, initialization, and DataLoader shuffling."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def average_logs(logs_list: list[dict[str, float | torch.Tensor]]) -> dict[str, float | torch.Tensor]:
    """Average scalar logs and concatenate histogram tensors across validation passes."""
    out: dict[str, float | torch.Tensor] = {}
    keys = logs_list[0].keys()
    for key in keys:
        values = [logs[key] for logs in logs_list]
        if isinstance(values[0], torch.Tensor):
            tensors = [v for v in values if isinstance(v, torch.Tensor) and v.numel() > 0]
            out[key] = torch.cat(tensors) if tensors else torch.empty(0)
        else:
            out[key] = sum(float(v) for v in values) / len(values)
    return out


def run_training(args: argparse.Namespace, device: torch.device, trial=None) -> float:
    """ 
    Run one fixed hyperparameter training job; 
    return best averaged validation loss.
    """
    
    trial_number = None if trial is None else int(trial.number)
    set_seed(args.seed)

    # ----- Load data/metadata and create fresh branch, decoder, optimizer, loaders, and class weights -----
    token_dir = Path(args.token_dir)
    train_ds = TokenizedTabularDataset.from_dir(token_dir, split="train")
    val_ds = TokenizedTabularDataset.from_dir(token_dir, split="val")
    metadata = train_ds.metadata
    feature_names = list(metadata["feature_names"])
    feature_type_ids = torch.as_tensor(metadata["feature_type_ids"]).long()
    categorical_cardinalities = [int(x) for x in metadata["categorical_cardinalities"]]
    real_missing_cardinality = int(metadata["missing_reason_cardinality"])
    mask_reason_code = real_missing_cardinality if args.mask_reason_code < 0 else int(args.mask_reason_code)
    model_missing_cardinality = max(real_missing_cardinality, mask_reason_code + 1)

    name_embeddings, feature_context_offsets, category_weights = load_embeddings(token_dir, metadata)
    if feature_context_offsets is None and "feature_context_offsets" in metadata:
        feature_context_offsets = torch.as_tensor(metadata["feature_context_offsets"]).long()

    cluster_assignments = load_cluster_assignments(Path(args.cluster_csv), feature_names)
    if cluster_assignments.max().item() >= args.num_clusters:
        raise ValueError("cluster CSV contains cluster_id >= --num-clusters")
    feature_indices = torch.where(cluster_assignments == int(args.cluster_id))[0]
    if feature_indices.numel() == 0:
        raise ValueError(f"cluster {args.cluster_id} has no features; nothing to pretrain")

    branch = ClusterBranch(
        feature_indices=feature_indices,
        name_embeddings=name_embeddings,
        feature_context_offsets=feature_context_offsets,
        feature_type_ids=feature_type_ids,
        categorical_cardinalities=categorical_cardinalities,
        continuous_bin_cardinality=int(metadata["continuous_bin_cardinality"]),
        missing_reason_cardinality=model_missing_cardinality,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        categorical_embedding_weights=category_weights,
        num_summary_tokens=args.num_summary_tokens,
    ).to(device)
    decoder = ClusterMAEReconstructionHead(
        feature_type_ids=feature_type_ids[feature_indices],
        categorical_cardinalities=[categorical_cardinalities[int(i)] for i in feature_indices.tolist()],
        d_model=args.d_model,
        dropout=args.dropout,
    ).to(device)

    # Computes class weights if needed
    cat_class_weights = None
    if args.cat_loss in {"weighted_ce", "weighted_focal"}:
        print("Computing per-feature categorical class weights from the training split...")
        cat_class_weights = compute_local_categorical_class_weights(
            dataset=train_ds,
            feature_indices=feature_indices,
            feature_type_ids=feature_type_ids,
            categorical_cardinalities=categorical_cardinalities,
            batch_size=args.batch_size,
            smoothing=args.cat_weight_smoothing,
            max_weight=args.cat_weight_max,
        )
        cat_class_weights = [w.to(device) if w is not None else None for w in cat_class_weights]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    pretrain_model = nn.ModuleDict({"branch": branch, "decoder": decoder})
    param_groups = build_adamw_param_groups(pretrain_model, weight_decay=float(args.weight_decay))
    optimizer = torch.optim.AdamW(param_groups, lr=float(args.lr))

    steps_per_epoch = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
    total_train_steps = max(1, int(args.epochs) * int(steps_per_epoch))
    scheduler = None
    warmup_steps = 0
    if args.lr_schedule == "cosine":
        # Create linear warmup followed by cosine decay
        total_steps = max(int(total_train_steps), 1)
        warmup_steps = int(float(args.warmup_frac) * total_steps)
        min_lr_ratio = float(args.min_lr_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    out_dir = Path(args.output_dir) if trial is None else Path(args.output_dir) / "optuna_trials" / f"trial_{trial_number:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"cluster{args.cluster_id}_d{args.d_model}_summaryTokens{args.num_summary_tokens}_"
        f"lr{args.lr:.2e}_wd{args.weight_decay:.2e}_"
        f"{args.lr_schedule}_warmup{args.warmup_frac}_minLR{args.min_lr_ratio}_"
        f"featureMaskMax{args.feature_mask_prob_max}_schemaDropMax{args.schema_dropout_prob_max}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if trial is not None:
        run_name = f"optuna_trial{trial_number:04d}_" + run_name
    writer = None if args.no_tensorboard else SummaryWriter(Path(args.log_dir) / run_name)

    metric_keys = [
        "loss", "loss_num", "loss_cat", 
        "num_rmse", "cat_acc_macro_feature",
        "n_masked", "n_masked_num", "n_masked_cat", 
        "actual_schema_dropout_prob", "schema_available_mean", 
        "actual_feature_mask_prob", "artificial_masked_mean", 
        "encoder_available_mean", "encoder_available_min", "encoder_available_p10", "encoder_available_p50", # visible to encoder including patient-missing values
        "visible_observed_value_mean", "visible_observed_value_min", "visible_observed_value_p10", "visible_observed_value_p50", # visible to encoder excluding patient-missing values
        "fraction_no_encoder_context", "fraction_no_visible_observed_value", #% rows where encoder sees no schema-available features, % rows where encoder sees no schema-available features with true observed values
        # auxiliary-loss metrics written to the existing CSV. 
        "loss_optimization", "summary_aux_raw", "summary_aux_weighted", "summary_aux_ratio",
        "summary_pairwise_similarity", "summary_conditional_entropy", "summary_marginal_entropy",
        "grad_norm",
    ]
    csv_path = out_dir / f"cluster_{args.cluster_id}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=["epoch", "lr", "weight_decay"] + [f"train_{k}" for k in metric_keys] + [f"val_{k}" for k in metric_keys]).writeheader()

    best_val = float("inf")
    common = dict(
        feature_mask_prob_max=float(args.feature_mask_prob_max),
        schema_dropout_prob_max=float(args.schema_dropout_prob_max),
        mask_reason_code=mask_reason_code,
        categorical_weight=args.categorical_weight,
        cat_loss=args.cat_loss,
        cat_class_weights=cat_class_weights,
        focal_gamma=args.focal_gamma,
        force_target_per_row=not args.no_force_target_per_row,
        keep_one_observed_value_visible=not args.allow_mask_last_observed_value,
        # fixed auxiliary-loss settings; effective lambda is added per epoch below. 
        summary_aux_loss=args.summary_aux_loss,
        summary_attention_margin=args.summary_attention_margin,
        summary_output_margin=args.summary_output_margin,
        summary_mi_beta=args.summary_mi_beta,
        summary_mi_temperature=args.summary_mi_temperature,
    )
    val_common = dict(common)
    val_common["schema_dropout_prob_max"] = 0.0

    print(
        f"Pretraining cluster {args.cluster_id} with {feature_indices.numel()} "
        f"features and {args.num_summary_tokens} summary token(s) on {device} | "
        f"peak_lr={args.lr:.3g}, weight_decay={args.weight_decay:.3g}, "
        f"schedule={args.lr_schedule}, warmup_steps={warmup_steps}/{total_train_steps}"
    )
    print(
        f"train_feature_mask_prob_max={args.feature_mask_prob_max}, "
        f"train_schema_dropout_prob_max={args.schema_dropout_prob_max}, "
        f"val_feature_mask_prob_max={args.feature_mask_prob_max}, "
        f"val_schema_dropout_prob_max=0.0, "
        f"val_mask_passes={args.val_mask_passes}, "
        f"min_lr_ratio={args.min_lr_ratio}"
    )
    # report the optional summary-token regularizer. 
    print(
        f"summary_aux_loss={args.summary_aux_loss}, "
        f"lambda={args.summary_aux_weight:g}, warmup_epochs={args.summary_aux_warmup_epochs}, "
        f"attention_margin={args.summary_attention_margin:g}, "
        f"output_margin={args.summary_output_margin:g}, "
        f"mi_beta={args.summary_mi_beta:g}, mi_temperature={args.summary_mi_temperature:g}"
    )

    for epoch in range(1, args.epochs + 1):
        # Keep training stochastic but reproducible, and prevent validation seeds
        # from controlling the next training epoch.
        set_seed(args.seed + 10_000 + epoch)

        # same linear auxiliary-loss warmup used by supervised training. 
        aux_scale = (
            1.0 if args.summary_aux_warmup_epochs == 0
            else min(1.0, epoch / float(args.summary_aux_warmup_epochs))
        )
        effective_aux_weight = float(args.summary_aux_weight) * aux_scale

        train_logs = run_epoch(
            branch, decoder, train_loader, optimizer, scheduler, device,
            train=True, max_batches=args.max_train_batches,
            summary_aux_weight=effective_aux_weight, **common,
        )

        val_pass_logs = []
        for pass_idx in range(int(args.val_mask_passes)):
            # Fixed validation corruption:
            # - no schema dropout
            # - same validation mask for this pass across epochs, trials, and reruns
            set_seed(args.seed + 100_000 + pass_idx)
            val_pass_logs.append(
                run_epoch(
                    branch, decoder, val_loader, optimizer, None, device,
                    train=False, max_batches=args.max_val_batches,
                    summary_aux_weight=effective_aux_weight, **val_common,
                )
            )
            
        val_logs = average_logs(val_pass_logs)
        val_loss = float(val_logs["loss"])
        lr = float(optimizer.param_groups[0]["lr"])
        wd = float(optimizer.param_groups[0]["weight_decay"])

        # show auxiliary contribution only when enabled.
        aux_text = (
            "" if args.summary_aux_loss == "none"
            else f" | train_aux {train_logs['summary_aux_weighted']:.4f} | aux_ratio {train_logs['summary_aux_ratio']:.3f}"
        )
        print(
            f"epoch {epoch:03d} | lr {lr:.2e} | train {train_logs['loss']:.4f}{aux_text} | val_avg {val_loss:.4f} | "
            f"val_num {val_logs['loss_num']:.4f} | val_cat {val_logs['loss_cat']:.4f} | "
            f"enc_ctx {val_logs['encoder_available_mean']:.1f} | obs_ctx {val_logs['visible_observed_value_mean']:.1f} | "
            f"masked {int(float(val_logs['n_masked']))}"
        )

        if writer is not None:
            for key in ["loss", "loss_num", "loss_cat"]:
                writer.add_scalars(f"loss/{key}", {"train": train_logs[key], "val": val_logs[key]}, epoch)
            # auxiliary-loss TensorBoard metrics mirror supervised training. 
            writer.add_scalars("loss/optimization", {"train": train_logs["loss_optimization"], "val": val_logs["loss_optimization"]}, epoch)
            writer.add_scalars("summary_aux/raw", {"train": train_logs["summary_aux_raw"], "val": val_logs["summary_aux_raw"]}, epoch)
            writer.add_scalars("summary_aux/weighted", {"train": train_logs["summary_aux_weighted"], "val": val_logs["summary_aux_weighted"]}, epoch)
            writer.add_scalar("summary_aux/train_ratio_to_reconstruction", train_logs["summary_aux_ratio"], epoch)
            writer.add_scalar("summary_aux/effective_weight", effective_aux_weight, epoch)
            writer.add_scalars("summary_aux/pairwise_similarity", {"train": train_logs["summary_pairwise_similarity"], "val": val_logs["summary_pairwise_similarity"]}, epoch)
            writer.add_scalars("summary_aux/conditional_entropy", {"train": train_logs["summary_conditional_entropy"], "val": val_logs["summary_conditional_entropy"]}, epoch)
            writer.add_scalars("summary_aux/marginal_entropy", {"train": train_logs["summary_marginal_entropy"], "val": val_logs["summary_marginal_entropy"]}, epoch)
            for key in ["num_rmse", "cat_acc_macro_feature"]:
                writer.add_scalars(f"metric/{key}", {"train": train_logs[key], "val": val_logs[key]}, epoch)

            for key in [
                "actual_schema_dropout_prob", "schema_available_mean", 
                "actual_feature_mask_prob", "artificial_masked_mean",
                "encoder_available_mean", "visible_observed_value_mean", 
                "fraction_no_encoder_context", "fraction_no_visible_observed_value",
            ]:
                writer.add_scalars(f"masking/{key}", {"train": train_logs[key], "val": val_logs[key]}, epoch)
            writer.add_scalar("optim/lr", lr, epoch)
            writer.add_scalar("optim/weight_decay", wd, epoch)
            writer.add_scalar("optim/grad_norm", train_logs["grad_norm"], epoch)
            writer.flush()

        row = {"epoch": epoch, "lr": lr, "weight_decay": wd}
        for split, logs in [("train", train_logs), ("val", val_logs)]:
            for key in metric_keys:
                row[f"{split}_{key}"] = logs[key]
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=row.keys()).writerow(row)

        ckpt = {
            "epoch": epoch,
            "cluster_id": int(args.cluster_id),
            "feature_indices": feature_indices.cpu(),
            "branch_state_dict": branch.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "cluster_assignments": cluster_assignments.cpu(),
            "mask_reason_code": int(mask_reason_code),
            "model_missing_reason_cardinality": int(model_missing_cardinality),
            "num_summary_tokens": int(args.num_summary_tokens),
            "feature_context_offsets": feature_context_offsets.cpu() if feature_context_offsets is not None else None,
            "feature_mask_prob_max": float(args.feature_mask_prob_max),
            "schema_dropout_prob_max": float(args.schema_dropout_prob_max),
            "lr_schedule": args.lr_schedule,
            "warmup_frac": float(args.warmup_frac),
            "min_lr_ratio": float(args.min_lr_ratio),
            "warmup_steps": int(warmup_steps),
            "total_train_steps": int(total_train_steps),
            "cat_loss": args.cat_loss,
            "categorical_weight": float(args.categorical_weight),
            "cat_class_weights": [w.detach().cpu() if w is not None else None for w in cat_class_weights] if cat_class_weights is not None else None,
            "args": vars(args),
            "val_loss": val_loss,
        }
        torch.save(ckpt, out_dir / f"cluster_{args.cluster_id}_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / f"cluster_{args.cluster_id}_best.pt")

        if trial is not None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                if writer is not None:
                    writer.close()
                raise RuntimeError("OPTUNA_PRUNED")

    if writer is not None:
        writer.close()
    return best_val


def main() -> None:
    """Run either one fixed pretraining job or Optuna HPO over lr/weight_decay."""
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not args.hpo:
        run_training(args, device)
        return

    def objective(trial):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.lr = trial.suggest_float("lr", args.optuna_lr_low, args.optuna_lr_high, log=True)
        trial_args.weight_decay = trial.suggest_float("weight_decay", args.optuna_weight_decay_low, args.optuna_weight_decay_high, log=True)
        try:
            return run_training(trial_args, device, trial=trial)
        except RuntimeError as exc:
            if str(exc) == "OPTUNA_PRUNED":
                raise optuna.TrialPruned()
            raise

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    storage = args.optuna_storage or f"sqlite:///{Path(args.output_dir) / f'optuna_cluster_{args.cluster_id}.db'}"
    study_name = args.optuna_study_name or f"cluster{args.cluster_id}_pretrain_lr_wd"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            multivariate=True,
            n_startup_trials=args.optuna_startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.optuna_pruner_startup_trials,
            n_warmup_steps=args.optuna_pruner_warmup_steps,
            interval_steps=1,
        ),
    )

    study.optimize(objective, n_trials=args.optuna_trials)
    print("Best trial:")
    print(f"  value: {study.best_trial.value}")
    print(f"  params: {study.best_trial.params}")
    print(f"  storage: {storage}")


if __name__ == "__main__":
    main()
