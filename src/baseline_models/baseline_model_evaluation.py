"""Enhanced classification / regression evaluation for the team pipeline.

Drop-in replacement: same `evaluate_classification` / `evaluate_regression`
signatures as before, so `model_runner.py` continues to work unchanged.

Adds (over the previous accuracy / F1 / log-loss / confusion baseline):
  - per-class AUROC + AUPRC + AP-lift (one-vs-rest), printed and returned
  - macro AUROC / AUPRC with bootstrap 95% CIs (1000 resamples by default)
  - multiclass Brier score (sum over the probability simplex)
  - Expected Calibration Error (top-label, 10 bins)
  - quadratic-weighted Cohen's kappa  — correct metric for ORDINAL targets
    such as CKM_Stage (0 < 1 < 2); enabled via `ordinal=True`
  - row-normalized confusion matrix (where does each true class go?)
  - reusable post-hoc calibration helpers
    (`fit_apply_isotonic`, `fit_apply_sigmoid`, `expected_calibration_error`,
    `reliability_points`) — lift these into your own scripts when a model
    needs calibration on top of training.

These functions also RETURN a metrics dict (the old version returned None)
so callers that want to persist results have something to save. Callers
that ignore the return value (the existing `model_runner.py`) work unchanged.
"""
from __future__ import annotations

import warnings
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

SEED = 42
N_BOOT = 1000
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.clinical_cluster_experts.utils import resolve_device
from src.name_value_transformer.checkpoint import load_flat_encoder_and_head
from src.name_value_transformer.data import select_batch_features
from src.name_value_transformer.model import TransformerOutput
from src.tokenizer.dataset import TokenizedTabularDataset, batch_from_dataset_tensors, load_tokenized_batch
from src.tokenizer.preprocessing import slice_transformed_batch


@dataclass(frozen=True)
class TaskConfig:
    """Binary target descriptor exposed to evaluation/visualization code."""

    name: str
    target_column: str
    task_type: str = "binary"
    num_outputs: int = 1
    cols_to_drop: tuple[str, ...] = ()


def load_token_targets(token_dir: Path, split: str) -> dict[str, list[object]]:
    payload = torch.load(token_dir / f"{split}_targets.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {split}_targets.pt to contain a dictionary.")
    return payload


def valid_target_mask(targets: torch.Tensor, task_type: str = "binary") -> torch.Tensor:
    del task_type
    return ~torch.isnan(targets.squeeze(-1))


def apply_task_feature_mask(batch, allowed_features):
    del allowed_features
    return batch


class _FlatEvaluationModel:
    """Adapter exposing the flat encoder/head to existing analysis utilities."""

    def __init__(self, encoder, head, inputs, target_columns: list[str]):
        self.encoder = encoder
        self.head = head
        self.inputs = inputs
        self.target_columns = list(target_columns)
        self.task_indices = {
            target.removeprefix("label_"): idx for idx, target in enumerate(self.target_columns)
        }
        self.n_features = inputs.feature_view.n_global_features

    def eval(self):
        self.encoder.eval()
        self.head.eval()
        return self

    def encode(
        self,
        numeric_values,
        continuous_bin_codes,
        categorical_codes,
        missing_reason_codes,
        missing_mask,
        observed_mask=None,
        feature_context_codes=None,
    ) -> TransformerOutput:
        global_batch = {
            "numeric_values": numeric_values,
            "continuous_bin_codes": continuous_bin_codes,
            "categorical_codes": categorical_codes,
            "missing_reason_codes": missing_reason_codes,
            "missing_mask": missing_mask,
        }
        if observed_mask is not None:
            global_batch["observed_mask"] = observed_mask
        if feature_context_codes is not None:
            global_batch["feature_context_codes"] = feature_context_codes
        batch = select_batch_features(global_batch, self.inputs.feature_view)
        return self.encoder(batch, batch.get("observed_mask"))

    def predict(
        self,
        token_embeddings,
        task_name=None,
        observed_mask=None,
        patient_embedding=None,
    ):
        del token_embeddings, observed_mask
        if patient_embedding is None:
            raise ValueError("patient_embedding is required.")
        if task_name not in self.task_indices:
            raise ValueError(f"Unknown binary task: {task_name}")
        logits = self.head(patient_embedding)["binary_logits"]
        return logits[:, self.task_indices[task_name]:self.task_indices[task_name] + 1]


# ── Reusable calibration helpers (binary; multiclass via OvR + renorm) ─────

def fit_apply_isotonic(val_p, val_y, test_p):
    """Fit isotonic on (val_p, val_y) and apply to test_p (1D arrays, binary)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_p, val_y)
    return iso.predict(test_p)


def fit_apply_sigmoid(val_p, val_y, test_p):
    """Platt-style sigmoid recalibration. Stable for rare classes when isotonic
    has too few positives. Binary 1D arrays."""
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(np.asarray(val_p).reshape(-1, 1), val_y)
    return lr.predict_proba(np.asarray(test_p).reshape(-1, 1))[:, 1]


def expected_calibration_error(correct, conf, n_bins=10):
    """Top-label ECE: binned mean |accuracy - confidence|."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins[1:-1])
    n = len(conf); ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_points(y, p, n_bins=10):
    """Return (mean_pred, frac_pos) per occupied bin for plotting reliability."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def multiclass_brier(y_true, y_proba, classes):
    Y = label_binarize(y_true, classes=classes)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    return float(np.mean(np.sum((y_proba - Y) ** 2, axis=1)))


# ── evaluate_classification (drop-in signature, enhanced output) ───────────

def evaluate_classification(
    y_true,
    y_pred,
    y_proba=None,
    class_labels=None,
    feature_importances=None,
    feature_names=None,
    ordinal: bool = False,
) -> dict:
    """Print a rich classification report and return a metrics dict.

    Same signature as the previous version; adds optional `ordinal=True` for
    targets like CKM_Stage where the classes are ordered.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = list(class_labels) if class_labels is not None else sorted(np.unique(y_true).tolist())
    n_classes = len(classes)
    out: dict = {}

    print("\n--- Model Evaluation ---")
    out["accuracy"]          = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"]          = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["weighted_f1"]       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    print(f"Accuracy:           {out['accuracy']:.4f}")
    print(f"Balanced accuracy:  {out['balanced_accuracy']:.4f}")
    print(f"Macro F1:           {out['macro_f1']:.4f}")
    print(f"Weighted F1:        {out['weighted_f1']:.4f}")

    if ordinal and n_classes > 2:
        out["quadratic_weighted_kappa"] = float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        )
        print(f"Quadratic-weighted kappa (ordinal): {out['quadratic_weighted_kappa']:.4f}")

    # Probabilistic metrics
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        if y_proba.shape[1] != n_classes:
            warnings.warn(
                f"y_proba has {y_proba.shape[1]} columns but there are {n_classes} classes "
                f"({classes}); skipping probabilistic metrics. Pass class_labels in the model's "
                f"classes_ order so the probability columns line up.",
                stacklevel=2,
            )
            y_proba = None

    if y_proba is not None:
        Ybin = label_binarize(y_true, classes=classes)
        if Ybin.shape[1] == 1:
            Ybin = np.hstack([1 - Ybin, Ybin])

        try:
            out["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
            print(f"Log loss:           {out['log_loss']:.4f}")
        except ValueError as e:
            out["log_loss"] = float("nan")
            warnings.warn(f"log_loss failed ({e}); reported as NaN.", stacklevel=2)

        out["multiclass_brier"] = multiclass_brier(y_true, y_proba, classes)
        conf = y_proba.max(axis=1)
        correct = (np.array([classes[i] for i in y_proba.argmax(axis=1)]) == y_true).astype(float)
        out["ece_top_label"] = expected_calibration_error(correct, conf)
        print(f"Brier (multiclass): {out['multiclass_brier']:.4f}")
        print(f"ECE (top-label):    {out['ece_top_label']:.4f}")

        # Per-class discrimination + precision lift
        print(f"\n  {'class':>8} {'prev':>7} {'AUROC':>7} {'AUPRC':>7} {'AP-lift':>8}")
        per_class = {}
        auroc_list, auprc_list = [], []
        for k, c in enumerate(classes):
            yk = Ybin[:, k]
            prev = float(yk.mean())
            if 0 < int(yk.sum()) < len(yk):
                au = float(roc_auc_score(yk, y_proba[:, k]))
                ap = float(average_precision_score(yk, y_proba[:, k]))
            else:
                au = ap = float("nan")
            lift = ap / prev if prev > 0 else float("nan")
            per_class[str(c)] = {"prevalence": prev, "auroc": au, "auprc": ap, "ap_lift": lift}
            auroc_list.append(au); auprc_list.append(ap)
            print(f"  {str(c):>8} {prev:>7.3f} {au:>7.3f} {ap:>7.3f} {lift:>7.2f}x")
        out["per_class"]   = per_class
        out["macro_auroc"] = float(np.nanmean(auroc_list))
        out["macro_auprc"] = float(np.nanmean(auprc_list))

        # Bootstrap CIs on macro AUROC / macro AUPRC. Within each resample, average the
        # per-class one-vs-rest scores with nanmean (a class absent from the resample
        # contributes NaN), so the CI is built the same way as the point estimates above.
        # Resamples in which no class is scorable are dropped and counted.
        rng = np.random.default_rng(SEED)
        n = len(y_true)
        boot_auroc, boot_auprc, n_skipped = [], [], 0
        for _ in range(N_BOOT):
            ix = rng.integers(0, n, size=n)
            au_k = np.full(n_classes, np.nan)
            ap_k = np.full(n_classes, np.nan)
            for k in range(n_classes):
                yk = Ybin[ix, k]
                if 0 < int(yk.sum()) < len(yk):
                    au_k[k] = roc_auc_score(yk, y_proba[ix, k])
                    ap_k[k] = average_precision_score(yk, y_proba[ix, k])
            if np.all(np.isnan(au_k)):
                n_skipped += 1
                continue
            boot_auroc.append(float(np.nanmean(au_k)))
            boot_auprc.append(float(np.nanmean(ap_k)))
        if boot_auroc:
            out["macro_auroc_ci95"] = [float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))]
            out["macro_auprc_ci95"] = [float(np.percentile(boot_auprc, 2.5)), float(np.percentile(boot_auprc, 97.5))]
            out["bootstrap_resamples_skipped"] = n_skipped
            print(f"\n  macro AUROC {out['macro_auroc']:.4f}  "
                  f"CI95 [{out['macro_auroc_ci95'][0]:.4f}, {out['macro_auroc_ci95'][1]:.4f}]")
            print(f"  macro AUPRC {out['macro_auprc']:.4f}  "
                  f"CI95 [{out['macro_auprc_ci95'][0]:.4f}, {out['macro_auprc_ci95'][1]:.4f}]")
            if n_skipped:
                print(f"  ({n_skipped}/{N_BOOT} resamples skipped: no scorable class)")

    # Confusion matrix (raw + normalized)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                         columns=[f"pred_{c}" for c in classes])
    cm_norm_df = pd.DataFrame(cm_norm, index=[f"true_{c}" for c in classes],
                              columns=[f"pred_{c}" for c in classes])
    out["confusion_matrix"]            = cm_df.to_dict()
    out["confusion_matrix_normalized"] = cm_norm_df.to_dict()
    print("\n--- Confusion Matrix ---")
    print(cm_df.to_string())
    print("\n--- Row-normalized confusion matrix (recall per class) ---")
    print(cm_norm_df.round(3).to_string())

    if feature_importances is not None and feature_names is not None:
        imp = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
        print("\n--- Top 10 Most Important Features ---")
        print(imp.head(10).round(4).to_string())
        out["top_features"] = imp.head(10).to_dict()

    return out


def evaluate_regression(
    y_true,
    y_pred,
    feature_importances=None,
    feature_names=None,
) -> dict:
    """Regression eval with bootstrap CI on R^2 (added)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }
    rng = np.random.default_rng(SEED)
    n = len(y_true); vals = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, n, size=n)
        try:
            vals.append(r2_score(y_true[ix], y_pred[ix]))
        except ValueError:
            continue
    out["r2_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2

    print("\n--- Model Evaluation ---")
    print(f"Test RMSE: {out['rmse']:.4f}")
    print(f"Test MAE:  {out['mae']:.4f}")
    print(f"Test R^2:  {out['r2']:.4f}  CI95 [{out['r2_ci95'][0]:.4f}, {out['r2_ci95'][1]:.4f}]")

    if feature_importances is not None and feature_names is not None:
        imp = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
        print("\n--- Top 10 Most Important Features ---")
        print(imp.head(10).round(4).to_string())
        out["top_features"] = imp.head(10).to_dict()

    return out
    print("\n--- Top 10 Most Important Features ---")
    importance = pd.Series(feature_importances, index=feature_names)
    importance = importance.sort_values(ascending=False)
    print(importance.head(10))


def load_name_value_transformer_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[_FlatEvaluationModel, list[TaskConfig], dict[str, dict[str, int] | None], dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = dict(checkpoint.get("args", {}))
    token_dir_value = args.get("token_dir")
    cluster_csv_value = args.get("cluster_csv")
    if not token_dir_value or not cluster_csv_value:
        raise ValueError("Flat checkpoint args must contain token_dir and cluster_csv paths.")
    encoder, head, checkpoint, inputs = load_flat_encoder_and_head(
        checkpoint_path,
        token_dir=Path(token_dir_value),
        cluster_csv=Path(cluster_csv_value),
        device=device,
    )
    target_columns = [str(value) for value in checkpoint["target_columns"]]
    task_configs = [TaskConfig(name=value.removeprefix("label_"), target_column=value) for value in target_columns]
    model = _FlatEvaluationModel(encoder, head, inputs, target_columns).eval()
    label_mappings = {task.name: None for task in task_configs}
    metadata = {
        "global_feature_mask": None,
        "feature_indices": inputs.feature_view.global_indices,
        "args": args,
    }
    return model, task_configs, label_mappings, metadata


@torch.no_grad()
def infer_name_value_transformer(
    model: _FlatEvaluationModel,
    loader: DataLoader,
    task_configs: list[TaskConfig],
    targets_by_task: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    rows_by_task: dict[str, list[np.ndarray]] = {task.name: [] for task in task_configs}
    true_by_task: dict[str, list[np.ndarray]] = {task.name: [] for task in task_configs}
    pred_by_task: dict[str, list[np.ndarray]] = {task.name: [] for task in task_configs}
    proba_by_task: dict[str, list[np.ndarray]] = {task.name: [] for task in task_configs}

    for payload in tqdm(loader, desc="inference", leave=False):
        batch = batch_from_dataset_tensors(payload, device)
        row_idx = payload["row_idx"].to(device)
        output = model.encode(
            batch.numeric_values,
            batch.continuous_bin_codes,
            batch.categorical_codes,
            batch.missing_reason_codes,
            batch.missing_mask,
            batch.observed_mask,
            batch.feature_context_codes,
        )
        for task in task_configs:
            targets = targets_by_task[task.name].to(device)[row_idx]
            valid = valid_target_mask(targets, task.task_type)
            if not bool(valid.any()):
                continue

            logits = model.predict(
                output.token_embeddings[valid],
                task.name,
                batch.observed_mask[valid],
                output.patient_embedding[valid] if output.patient_embedding is not None else None,
            )
            valid_rows = row_idx[valid].detach().cpu().numpy()
            valid_targets = targets[valid].detach().cpu().numpy()
            if task.task_type == "multiclass":
                probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                predictions = probabilities.argmax(axis=1)
                proba_by_task[task.name].append(probabilities)
            elif task.task_type == "binary":
                probabilities = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                predictions = (probabilities >= 0.5).astype(int)
                proba_by_task[task.name].append(np.column_stack([1.0 - probabilities, probabilities]))
            else:
                predictions = logits.detach().cpu().numpy().reshape(-1)

            rows_by_task[task.name].append(valid_rows)
            true_by_task[task.name].append(valid_targets.reshape(-1))
            pred_by_task[task.name].append(predictions.reshape(-1))

    outputs = {}
    for task in task_configs:
        if not rows_by_task[task.name]:
            outputs[task.name] = {
                "row_idx": np.array([], dtype=int),
                "y_true": np.array([]),
                "y_pred": np.array([]),
            }
            continue
        task_output = {
            "row_idx": np.concatenate(rows_by_task[task.name]),
            "y_true": np.concatenate(true_by_task[task.name]),
            "y_pred": np.concatenate(pred_by_task[task.name]),
        }
        if proba_by_task[task.name]:
            task_output["y_proba"] = np.concatenate(proba_by_task[task.name])
        outputs[task.name] = task_output
    return outputs


def encode_targets_for_checkpoint(
    token_dir: Path,
    split: str,
    task_configs: list[TaskConfig],
    label_mappings: Mapping[str, dict[str, int] | None],
    max_rows: int | None,
) -> dict[str, torch.Tensor]:
    raw_targets = load_token_targets(token_dir, split)
    encoded = {}
    for task in task_configs:
        if task.target_column not in raw_targets:
            available = sorted(raw_targets)
            raise ValueError(
                f"Target column {task.target_column!r} for task {task.name!r} not found in {split}_targets.pt. "
                f"Available columns: {available}"
            )
        series = pd.Series(raw_targets[task.target_column], name=task.target_column)
        if max_rows is not None:
            series = series.head(max_rows)

        if task.task_type == "multiclass":
            mapping = label_mappings.get(task.name)
            if mapping is None:
                classes = sorted(series.dropna().unique().tolist())
                mapping = {str(label): idx for idx, label in enumerate(classes)}
            values = [mapping[str(value)] if not pd.isna(value) else -1 for value in series]
            encoded[task.name] = torch.tensor(values, dtype=torch.long)
        else:
            encoded[task.name] = torch.tensor(series.astype(float).to_numpy(), dtype=torch.float32).unsqueeze(-1)
    return encoded


def save_name_value_predictions(
    outputs: Mapping[str, dict[str, np.ndarray]],
    task_configs: list[TaskConfig],
    label_mappings: Mapping[str, dict[str, int] | None],
    model_name: str,
    results_dir: Path,
    split: str,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for task in task_configs:
        output = outputs[task.name]
        pred_df = pd.DataFrame(
            {
                "row_idx": output["row_idx"],
                "y_true": output["y_true"],
                f"{model_name}_pred": output["y_pred"],
            }
        ).set_index("row_idx")

        y_proba = output.get("y_proba")
        if y_proba is not None:
            mapping = label_mappings.get(task.name)
            if mapping:
                labels = [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
            else:
                labels = list(range(y_proba.shape[1]))
            proba_df = pd.DataFrame(
                y_proba,
                columns=[f"{model_name}_proba_{label}" for label in labels],
                index=pred_df.index,
            )
            pred_df = pd.concat([pred_df, proba_df], axis=1)

        output_path = results_dir / f"{model_name}_{task.name}_{split}_predictions.parquet"
        try:
            pred_df.to_parquet(output_path)
            print(f"[OK] Predictions saved to {output_path}")
        except ImportError as exc:
            csv_path = output_path.with_suffix(".csv")
            pred_df.to_csv(csv_path)
            print(f"[WARN] Parquet support unavailable ({exc}); predictions saved to {csv_path}")


def run_name_value_transformer_inference(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, task_configs, label_mappings, metadata = load_name_value_transformer_checkpoint(args.checkpoint, device)
    if args.tasks:
        selected = set(args.tasks)
        unknown = sorted(selected - {task.name for task in task_configs})
        if unknown:
            raise ValueError(f"Task(s) not found in checkpoint: {unknown}")
        task_configs = [task for task in task_configs if task.name in selected]

    batch = load_tokenized_batch(args.token_dir, args.split)
    batch = slice_transformed_batch(batch, args.max_rows)
    global_feature_mask = metadata.get("global_feature_mask")
    if global_feature_mask is not None:
        batch = apply_task_feature_mask(batch, global_feature_mask)

    targets_by_task = encode_targets_for_checkpoint(
        args.token_dir,
        args.split,
        task_configs,
        label_mappings,
        args.max_rows,
    )
    loader = DataLoader(TokenizedTabularDataset(batch=batch, metadata={}, split=args.split), batch_size=args.batch_size)
    outputs = infer_name_value_transformer(model, loader, task_configs, targets_by_task, device)

    for task in task_configs:
        output = outputs[task.name]
        print(f"\n=== {task.name} ({task.task_type}) ===")
        if task.task_type == "regression":
            evaluate_regression(pd.Series(output["y_true"]), output["y_pred"], None, None)
        else:
            y_proba = output.get("y_proba")
            class_labels = None
            mapping = label_mappings.get(task.name)
            if mapping:
                class_labels = [idx for _, idx in sorted(mapping.items(), key=lambda item: item[1])]
            evaluate_classification(
                pd.Series(output["y_true"]).astype(int),
                output["y_pred"].astype(int),
                y_proba=y_proba,
                class_labels=class_labels,
            )

    save_name_value_predictions(outputs, task_configs, label_mappings, args.model_name, args.results_dir, args.split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained name-value transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt.")
    parser.add_argument(
        "--token-dir",
        type=Path,
        required=True,
        help="Directory containing *_tokens.pt and *_targets.pt files.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset of checkpoint task heads.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-name", default="name_value_transformer")
    parser.add_argument("--results-dir", type=Path, default=Path("data/results/name_value_transformer/inference"))
    return parser.parse_args()


if __name__ == "__main__":
    run_name_value_transformer_inference(parse_args())
