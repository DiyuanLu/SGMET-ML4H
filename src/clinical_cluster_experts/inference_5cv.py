"""
Inference and evaluation for v4 supervised SGMET binary downstream tasks.

Five-fold example:
python -m src.clinical_cluster_experts.inference_5cv \
  --run-cv \
  --cv-root data/processed/tokenized_nhanes_v5/cv_splits \
  --cluster-csv-name feature_clusters_biolord_v5_k7_leiden.csv \
  --checkpoint-root outputs/20260816_sgmet_e2e_scratch_semantic_3cls_attnmi_b1_t1_lam0p005_fd10_cd90_5cv \
  --split test \
  --num-clusters 7 \
  --active-clusters 0,1,2,3,4,5,6 \
  --ignore-clusters "" \
  --batch-size 128 \
  --device mps \
  --no-cluster-embedding \
  --save-cv-results \
  --save-fold-metrics-json \
  --save-metric-summary-csv \
  --save-predictions \
  --save-summary-diagnostics 


To print model architect only: 
python -m src.clinical_cluster_experts.inference_5cv \
  --token-dir data/processed/tokenized_nhanes_v5/cv_splits/fold0 \
  --cluster-csv data/processed/tokenized_nhanes_v5/cluster_maps/feature_clusters_biolord_v5_k7_leiden.csv \
  --num-summary-tokens 1 \
  --split test \
  --num-clusters 7 \
  --batch-size 128 \
  --device mps \
  --no-cluster-embedding \
  --print-model-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.tokenizer.dataset import TokenizedTabularDataset
from src.tokenizer.schema import FeatureType
from src.clinical_cluster_experts.model import ClinicalClusterEncoder, DownstreamPredictionHead
from src.clinical_cluster_experts.utils import (
    resolve_device,
    move_batch_to_device,
    load_cluster_assignments,
    load_embeddings,
)
from src.clinical_cluster_experts.summary_diagnostics import (
    SummaryTokenDiagnosticsAccumulator,
    append_summary_diagnostics_csv,
)
from src.baseline_models.baseline_model_evaluation import evaluate_classification


MACRO_METRIC_KEYS = [
    "macro_auroc",
    "macro_auprc",
]

PER_TASK_METRIC_KEYS = [
    "positive_class_auroc",
    "positive_class_auprc",
]



def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def run_cross_validation(args: argparse.Namespace) -> None:
    """Evaluate each test fold and aggregate per-target AUROC/AUPRC across folds."""
    import tempfile

    if not args.cv_root or not args.checkpoint_root:
        raise ValueError("Both --cv-root and --checkpoint-root are required for --run-cv.")

    folds = _parse_int_list(args.folds)
    if not folds:
        raise ValueError("--folds must contain at least one fold ID.")

    cv_root = Path(args.cv_root)
    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.cv_output_dir or (checkpoint_root / f"inference_{args.split}"))
    repo_root = Path(__file__).resolve().parents[2]

    save_any_cv_output = bool(
        args.save_cv_results
        or args.save_predictions
        or args.save_fold_metrics_json
        or args.save_metric_summary_csv
        or args.save_branch_params_csv
        or args.save_summary_diagnostics
    )
    if save_any_cv_output:
        output_root.mkdir(parents=True, exist_ok=True)

    fold_summary_rows: list[dict] = []
    per_task_fold_rows: list[dict] = []
    fold_metrics_payload: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="sgmet_cv_inference_") as tmp_dir:
        tmp_root = Path(tmp_dir)

        for fold in folds:
            token_dir = cv_root / f"fold{fold}"
            cluster_csv = token_dir / args.cluster_csv_name
            checkpoint = checkpoint_root / f"fold{fold}" / args.checkpoint_name
            fold_out = output_root / f"fold{fold}"
            if save_any_cv_output:
                fold_out.mkdir(parents=True, exist_ok=True)

            # A metrics JSON is needed internally by the CV wrapper. When the user
            # did not request it, write it to a temporary directory and delete it
            # automatically after aggregation.
            metrics_json = (
                fold_out / f"{args.split}_metrics.json"
                if args.save_fold_metrics_json
                else tmp_root / f"fold{fold}_{args.split}_metrics.json"
            )
            predictions_csv = fold_out / f"{args.split}_predictions.csv"

            for path, description in [
                (token_dir, "fold token directory"),
                (cluster_csv, "cluster assignment CSV"),
                (checkpoint, "supervised checkpoint"),
            ]:
                if not path.exists():
                    raise FileNotFoundError(f"Missing {description} for fold {fold}: {path}")

            cmd = [
                sys.executable,
                "-m",
                "src.clinical_cluster_experts.inference_5cv",
                "--token-dir", str(token_dir),
                "--cluster-csv", str(cluster_csv),
                "--split", args.split,
                "--num-clusters", str(args.num_clusters),
                "--checkpoint", str(checkpoint),
                "--active-clusters", args.active_clusters,
                "--ignore-clusters", args.ignore_clusters,
                "--d-model", str(args.d_model),
                "--expert-n-heads", str(args.expert_n_heads),
                "--expert-n-layers", str(args.expert_n_layers),
                "--fusion-n-heads", str(args.fusion_n_heads),
                "--fusion-n-layers", str(args.fusion_n_layers),
                "--dropout", str(args.dropout),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--device", args.device,
                "--threshold", str(args.threshold),
                "--warmup-batches", str(args.warmup_batches),
                "--metrics-json", str(metrics_json),
            ]
            if args.num_summary_tokens is not None:
                cmd.extend(["--num-summary-tokens", str(args.num_summary_tokens)])
            if args.no_cluster_embedding:
                cmd.append("--no-cluster-embedding")
            if args.target_columns:
                cmd.extend(["--target-columns", args.target_columns])
            if args.max_rows is not None:
                cmd.extend(["--max-rows", str(args.max_rows)])
            if args.print_model_summary:
                cmd.append("--print-model-summary")
            if args.save_predictions:
                cmd.extend(["--output-csv", str(predictions_csv)])
            if args.save_metric_summary_csv:
                cmd.extend(["--metric-summary-csv", str(fold_out / f"{args.split}_metric_summary.csv")])
            if args.save_branch_params_csv:
                cmd.extend(["--branch-param-csv", str(fold_out / f"{args.split}_branch_params.csv")])
            if args.save_summary_diagnostics:
                cmd.extend(["--summary-diagnostics-dir", str(fold_out / f"{args.split}_summary_diagnostics")])
            cmd.extend(["--fusion-summary-slots", args.fusion_summary_slots,])

            print("\n" + "#" * 88)
            print(f"[CV] Fold {fold}: evaluating {checkpoint}")
            print("#" * 88)
            subprocess.run(cmd, check=True, cwd=repo_root)

            fold_metrics = json.loads(metrics_json.read_text())
            if args.save_fold_metrics_json or args.save_cv_results:
                fold_metrics_payload[str(fold)] = fold_metrics

            fold_row = {"fold": int(fold), "checkpoint": str(checkpoint)}
            for key in MACRO_METRIC_KEYS:
                fold_row[key] = float(fold_metrics.get(key, np.nan))
            fold_summary_rows.append(fold_row)

            for task_name, task_metrics in fold_metrics.get("per_task", {}).items():
                per_task_fold_rows.append({
                    "fold": int(fold),
                    "task": task_name,
                    "auroc": float(task_metrics.get("positive_class_auroc", np.nan)),
                    "auprc": float(task_metrics.get("positive_class_auprc", np.nan)),
                    "n": int(task_metrics.get("n", 0)),
                    "n_pos": int(task_metrics.get("n_pos", 0)),
                    "n_neg": int(task_metrics.get("n_neg", 0)),
                })

    fold_summary_df = pd.DataFrame(fold_summary_rows).sort_values("fold").reset_index(drop=True)
    per_task_fold_df = pd.DataFrame(per_task_fold_rows).sort_values(["task", "fold"]).reset_index(drop=True)

    macro_aggregate_rows = []
    for key in MACRO_METRIC_KEYS:
        mean, std = _mean_std(fold_summary_df[key].tolist())
        macro_aggregate_rows.append({
            "metric": key,
            "mean": mean,
            "std": std,
            "n_folds": int(fold_summary_df[key].notna().sum()),
        })
    macro_aggregate_df = pd.DataFrame(macro_aggregate_rows)

    per_task_aggregate_rows = []
    for task_name, group in per_task_fold_df.groupby("task", sort=True):
        auroc_mean, auroc_std = _mean_std(group["auroc"].tolist())
        auprc_mean, auprc_std = _mean_std(group["auprc"].tolist())
        per_task_aggregate_rows.append({
            "task": task_name,
            "auroc_mean": auroc_mean,
            "auroc_std": auroc_std,
            "auprc_mean": auprc_mean,
            "auprc_std": auprc_std,
            "n_folds_auroc": int(group["auroc"].notna().sum()),
            "n_folds_auprc": int(group["auprc"].notna().sum()),
            "n_total_across_test_folds": int(group["n"].sum()),
            "n_pos_total_across_test_folds": int(group["n_pos"].sum()),
            "n_neg_total_across_test_folds": int(group["n_neg"].sum()),
        })
    per_task_aggregate_df = pd.DataFrame(per_task_aggregate_rows)

    print("\n" + "=" * 100)
    print("[PER-TARGET, PER-FOLD TEST METRICS]")
    print(per_task_fold_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[PER-TARGET ACROSS-FOLD MEAN ± SAMPLE STD]")
    display_cols = ["task", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std"]
    print(per_task_aggregate_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[MACRO METRICS PER FOLD]")
    print(fold_summary_df[["fold", "macro_auroc", "macro_auprc"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[MACRO METRICS ACROSS FOLDS]")
    for row in macro_aggregate_rows:
        print(f"{row['metric']:20s} {row['mean']:.4f} ± {row['std']:.4f}")
    print("=" * 100)

    if args.save_cv_results:
        fold_csv = output_root / "macro_metrics_per_fold.csv"
        per_task_fold_csv = output_root / "per_target_metrics_per_fold.csv"
        per_task_aggregate_csv = output_root / "per_target_metrics_mean_std.csv"
        aggregate_json = output_root / "cv_metrics.json"

        fold_summary_df.to_csv(fold_csv, index=False)
        per_task_fold_df.to_csv(per_task_fold_csv, index=False)
        per_task_aggregate_df.to_csv(per_task_aggregate_csv, index=False)
        aggregate_json.write_text(json.dumps({
            "split": args.split,
            "folds": folds,
            "checkpoint_root": str(checkpoint_root),
            "cv_root": str(cv_root),
            "std_definition": "sample standard deviation of the fold-level point estimates (ddof=1)",
            "fold_metrics": fold_metrics_payload,
            "macro_aggregate": macro_aggregate_rows,
            "per_task_aggregate": per_task_aggregate_rows,
        }, indent=2))

        print(f"Saved macro metrics per fold:      {fold_csv}")
        print(f"Saved per-target fold metrics:     {per_task_fold_csv}")
        print(f"Saved per-target mean/std metrics: {per_task_aggregate_csv}")
        print(f"Saved complete CV metrics:         {aggregate_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal inference/evaluation for frozen-expert SGMET on v4 binary tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument("--cluster-csv", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-clusters", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--print-model-only", action="store_true")
    parser.add_argument("--print-model-summary", action="store_true", help="Print the active-forward architecture and parameter summary.",)
    parser.add_argument("--active-clusters", type=str, default="checkpoint", help="'checkpoint', 'all', or comma-separated IDs.")
    parser.add_argument("--ignore-clusters", type=str, default="7", help="Comma-separated cluster IDs to ignore, e.g. '7'.")
    parser.add_argument("--target-columns", type=str, default=None)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-summary-tokens", type=int, default=None,
        help=("Summary tokens per expert. By default this is read from the checkpoint, falling back to 1 for legacy checkpoints."))
    parser.add_argument("--fusion-summary-slots", type=str, default="all",
        help=("Summary slots passed to fusion. Use 'all' or one-based slot IDs, for example '1', '2', or '1,2'."))
    
    parser.add_argument("--expert-n-heads", type=int, default=4)
    parser.add_argument("--expert-n-layers", type=int, default=2)
    parser.add_argument("--fusion-n-heads", type=int, default=4)
    parser.add_argument("--fusion-n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-cluster-embedding", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--metrics-json", type=str, default=None, help="Optional path for detailed single-fold metrics JSON.")
    parser.add_argument("--metric-summary-csv", type=str, default=None, help="Optional path for the single-fold per-target AUROC/AUPRC table.")
    parser.add_argument("--branch-param-csv", type=str, default=None, help="Optional path for branch parameter counts.")
    parser.add_argument("--summary-diagnostics-dir", type=str, default=None,help=("Optional directory for dataset-level summary-token diagnostics. "))

    # Multi-fold inference mode. The script reuses the exact single-fold path above
    # for each fold, then reports fold-wise and mean ± sample-std metrics.
    parser.add_argument("--run-cv", action="store_true")
    parser.add_argument("--cv-root", type=str, default=None, help="Directory containing fold0, ..., fold4 token directories.")
    parser.add_argument("--checkpoint-root", type=str, default=None, help="Directory containing fold0/best.pt, ..., fold4/best.pt.")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--cluster-csv-name", type=str, default="feature_clusters_biolord_v4_k8_leiden.csv")
    parser.add_argument("--cv-output-dir", type=str, default=None)
    parser.add_argument("--save-cv-results", action="store_true", help="Save the final CV per-fold and mean/std tables.")
    parser.add_argument("--save-predictions", action="store_true", help="Save per-patient predictions for each fold.")
    parser.add_argument("--save-fold-metrics-json", action="store_true", help="Keep the detailed metrics JSON for each fold.")
    parser.add_argument("--save-metric-summary-csv", action="store_true", help="Save a compact per-target AUROC/AUPRC CSV for each fold.")
    parser.add_argument("--save-branch-params-csv", action="store_true", help="Save branch parameter counts for each fold.")
    parser.add_argument("--save-summary-diagnostics", action="store_true", help="Save summary-token diagnostic CSVs for each CV fold.",)
    args = parser.parse_args()
    if args.num_summary_tokens is not None and args.num_summary_tokens < 1:
        parser.error("--num-summary-tokens must be >= 1.")

    if args.run_cv:
        run_cross_validation(args)
        return

    if args.token_dir is None or args.cluster_csv is None:
        parser.error("Single-fold inference requires --token-dir and --cluster-csv.")
    if not args.print_model_only and args.checkpoint is None:
        parser.error("Inference requires --checkpoint unless --print-model-only is used.")
    if args.print_model_only:
        args.print_model_summary = True

    device = resolve_device(args.device)
    token_dir = Path(args.token_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint is not None else {}
    checkpoint_num_summary_tokens = int(
        ckpt.get(
            "num_summary_tokens",
            ckpt.get("args", {}).get("num_summary_tokens", 1),
        )
    )
    if args.num_summary_tokens is None:
        num_summary_tokens = checkpoint_num_summary_tokens
    else:
        num_summary_tokens = int(args.num_summary_tokens)
        if args.checkpoint is not None and num_summary_tokens != checkpoint_num_summary_tokens:
            raise ValueError(
                "Summary-token mismatch: checkpoint uses "
                f"{checkpoint_num_summary_tokens}, but --num-summary-tokens="
                f"{num_summary_tokens}."
            )
        
    slot_text = str(args.fusion_summary_slots).strip().lower()
    if slot_text == "all":
        fusion_summary_slot_indices = None
        selected_summary_slots = list(range(1, num_summary_tokens + 1))
    else:
        selected_summary_slots = [
            int(x.strip()) for x in slot_text.split(",") if x.strip()
        ]

        if not selected_summary_slots:
            raise ValueError("--fusion-summary-slots must not be empty.")
        if len(set(selected_summary_slots)) != len(selected_summary_slots):
            raise ValueError(
                f"Duplicate summary slots: {selected_summary_slots}"
            )
        if min(selected_summary_slots) < 1 or max(selected_summary_slots) > num_summary_tokens:
            raise ValueError(
                f"Checkpoint has {num_summary_tokens} summary tokens, but received "
                f"--fusion-summary-slots={args.fusion_summary_slots}."
            )

        # Convert human-facing 1-based IDs to Python's zero-based indices.
        fusion_summary_slot_indices = [
            slot - 1 for slot in selected_summary_slots
        ]

    dataset = TokenizedTabularDataset.from_dir(token_dir, split=args.split)
    dataset_for_loader = Subset(dataset, list(range(int(args.max_rows)))) if args.max_rows is not None else dataset

    metadata = dataset.metadata
    feature_names = list(metadata["feature_names"])
    target_columns = (
        [x.strip() for x in args.target_columns.split(",") if x.strip()]
        if args.target_columns
        else list(ckpt.get("target_columns", metadata["target_columns"]))
    )

    cluster_assignments = load_cluster_assignments(Path(args.cluster_csv), feature_names)
    ignored_clusters = [int(x.strip()) for x in str(args.ignore_clusters).split(",") if x.strip()]
    ignored_feature_mask = torch.zeros(len(feature_names), dtype=torch.bool)
    for c in ignored_clusters:
        ignored_feature_mask |= cluster_assignments.cpu().eq(int(c))

    active_text = str(args.active_clusters).strip().lower()
    if active_text == "checkpoint":
        active_clusters = [int(x) for x in ckpt.get("active_clusters", list(range(args.num_clusters)))]
    elif active_text == "all":
        active_clusters = list(range(args.num_clusters))
    elif active_text in {"", "none"}:
        active_clusters = []
    else:
        active_clusters = [int(x.strip()) for x in active_text.split(",") if x.strip()]
    active_clusters = [c for c in active_clusters if c not in set(ignored_clusters)]

    name_embeddings, feature_context_offsets, category_embedding_weights = load_embeddings(token_dir, metadata)
    missing_reason_cardinality = max(
        int(metadata["missing_reason_cardinality"]),
        int(ckpt.get("model_missing_reason_cardinality", metadata["missing_reason_cardinality"])),
    )

    encoder = ClinicalClusterEncoder(
        cluster_assignments=cluster_assignments,
        name_embeddings=name_embeddings,
        feature_context_offsets=feature_context_offsets,
        feature_type_ids=torch.as_tensor(metadata["feature_type_ids"]).long(),
        categorical_cardinalities=[int(x) for x in metadata["categorical_cardinalities"]],
        continuous_bin_cardinality=int(metadata["continuous_bin_cardinality"]),
        missing_reason_cardinality=missing_reason_cardinality,
        num_clusters=int(args.num_clusters),
        d_model=args.d_model,
        expert_n_heads=args.expert_n_heads,
        expert_n_layers=args.expert_n_layers,
        fusion_n_heads=args.fusion_n_heads,
        fusion_n_layers=args.fusion_n_layers,
        dropout=args.dropout,
        categorical_embedding_weights=category_embedding_weights,
        use_cluster_embedding=not args.no_cluster_embedding,
        num_summary_tokens=num_summary_tokens,
    ).to(device)

    head = DownstreamPredictionHead(
        d_model=args.d_model,
        n_binary_targets=len(target_columns),
        dropout=args.dropout,
    ).to(device)

    if args.checkpoint is not None:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        head.load_state_dict(ckpt["head_state_dict"])
    else:
        print("[INFO] No checkpoint provided. Randomly initialized model is used only for parameter counting.")
    
    encoder.eval()
    head.eval()

    def n_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    def n_trainable_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def fmt_params(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.3f}M ({n:,})"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K ({n:,})"
        return str(n)

    def sync_if_needed() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def n_nontrainable_params(module: torch.nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if not p.requires_grad)

    def module_param_info(module: torch.nn.Module | None) -> dict[str, int]:
        if module is None:
            return {"total": 0, "trainable": 0, "frozen": 0}
        total = n_params(module)
        trainable = n_trainable_params(module)
        return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}

    def add_counts(dst: dict[str, int], prefix: str, info: dict[str, int]) -> None:
        dst[f"{prefix}_total"] = dst.get(f"{prefix}_total", 0) + int(info["total"])
        dst[f"{prefix}_trainable"] = dst.get(f"{prefix}_trainable", 0) + int(info["trainable"])
        dst[f"{prefix}_frozen"] = dst.get(f"{prefix}_frozen", 0) + int(info["frozen"])

    def category_lookup_info(token_builder: torch.nn.Module | None) -> dict[str, int]:
        info = {
            "total": 0,
            "trainable": 0,
            "frozen": 0,
            "used_total": 0,
            "used_trainable": 0,
            "used_frozen": 0,
            "unused_total": 0,
            "unused_trainable": 0,
            "unused_frozen": 0,
        }
        if token_builder is None or not hasattr(token_builder, "category_embeddings"):
            return info

        type_ids = getattr(token_builder, "feature_type_ids", None)
        if type_ids is None or int(type_ids.numel()) == 0:
            categorical_mask = None
        else:
            categorical_mask = type_ids.detach().cpu().long().eq(int(FeatureType.CATEGORICAL))

        for feature_idx, emb in enumerate(token_builder.category_embeddings):
            if not hasattr(emb, "weight"):
                continue
            total = int(emb.weight.numel())
            trainable = int(emb.weight.numel()) if emb.weight.requires_grad else 0
            frozen = int(total - trainable)
            info["total"] += total
            info["trainable"] += trainable
            info["frozen"] += frozen

            used_in_forward = True if categorical_mask is None else bool(categorical_mask[feature_idx].item())
            if used_in_forward:
                info["used_total"] += total
                info["used_trainable"] += trainable
                info["used_frozen"] += frozen
            else:
                info["unused_total"] += total
                info["unused_trainable"] += trainable
                info["unused_frozen"] += frozen
        return info

    def token_builder_forward_info(token_builder: torch.nn.Module | None) -> dict[str, int]:
        info = {"total": 0, "trainable": 0, "frozen": 0}
        if token_builder is None:
            return info

        type_ids = getattr(token_builder, "feature_type_ids", None)
        if type_ids is None or int(type_ids.numel()) == 0:
            has_numerical = True
            has_categorical = True
        else:
            type_ids = type_ids.detach().cpu().long()
            has_numerical = bool(type_ids.eq(int(FeatureType.NUMERICAL)).any().item())
            has_categorical = bool(type_ids.eq(int(FeatureType.CATEGORICAL)).any().item())

        for module_name in ["name_projection", "type_embedding", "missing_embedding", "token_norm"]:
            add_counts(info, "tmp", module_param_info(getattr(token_builder, module_name, None)))

        if has_numerical:
            for module_name in ["numeric_encoder", "numeric_film", "continuous_bin_embedding"]:
                add_counts(info, "tmp", module_param_info(getattr(token_builder, module_name, None)))

        if has_categorical:
            cat_lookup = category_lookup_info(token_builder)
            info["tmp_total"] = info.get("tmp_total", 0) + int(cat_lookup["used_total"])
            info["tmp_trainable"] = info.get("tmp_trainable", 0) + int(cat_lookup["used_trainable"])
            info["tmp_frozen"] = info.get("tmp_frozen", 0) + int(cat_lookup["used_frozen"])
            add_counts(info, "tmp", module_param_info(getattr(token_builder, "category_projection", None)))

        return {
            "total": int(info.get("tmp_total", 0)),
            "trainable": int(info.get("tmp_trainable", 0)),
            "frozen": int(info.get("tmp_frozen", 0)),
        }

    def component_counts_for_branch(branch: torch.nn.Module) -> dict[str, int]:
        token_builder = getattr(branch, "token_builder", None)
        expert = getattr(branch, "expert", None)
        token_total = module_param_info(token_builder)
        token_forward = token_builder_forward_info(token_builder)
        expert_total = module_param_info(expert)
        cat_lookup = category_lookup_info(token_builder)
        category_projection = module_param_info(getattr(token_builder, "category_projection", None) if token_builder is not None else None)
        name_projection = module_param_info(getattr(token_builder, "name_projection", None) if token_builder is not None else None)
        numeric_encoder = module_param_info(getattr(token_builder, "numeric_encoder", None) if token_builder is not None else None)
        numeric_film = module_param_info(getattr(token_builder, "numeric_film", None) if token_builder is not None else None)
        continuous_bin_embedding = module_param_info(getattr(token_builder, "continuous_bin_embedding", None) if token_builder is not None else None)
        type_embedding = module_param_info(getattr(token_builder, "type_embedding", None) if token_builder is not None else None)
        missing_embedding = module_param_info(getattr(token_builder, "missing_embedding", None) if token_builder is not None else None)
        token_norm = module_param_info(getattr(token_builder, "token_norm", None) if token_builder is not None else None)
        expert_cls_params = int(expert.cls_token.numel()) if expert is not None and hasattr(expert, "cls_token") else 0
        expert_transformer = module_param_info(expert.encoder if expert is not None and hasattr(expert, "encoder") else None)

        return {
            "token_builder_total": int(token_total["total"]),
            "token_builder_trainable": int(token_total["trainable"]),
            "token_builder_frozen": int(token_total["frozen"]),
            "token_builder_forward_total": int(token_forward["total"]),
            "token_builder_forward_trainable": int(token_forward["trainable"]),
            "token_builder_forward_frozen": int(token_forward["frozen"]),
            "category_lookup_total": int(cat_lookup["total"]),
            "category_lookup_trainable": int(cat_lookup["trainable"]),
            "category_lookup_frozen": int(cat_lookup["frozen"]),
            "category_lookup_used_total": int(cat_lookup["used_total"]),
            "category_lookup_used_trainable": int(cat_lookup["used_trainable"]),
            "category_lookup_used_frozen": int(cat_lookup["used_frozen"]),
            "category_lookup_unused_total": int(cat_lookup["unused_total"]),
            "category_projection_total": int(category_projection["total"]),
            "category_projection_trainable": int(category_projection["trainable"]),
            "name_projection_total": int(name_projection["total"]),
            "numeric_encoder_total": int(numeric_encoder["total"]),
            "numeric_film_total": int(numeric_film["total"]),
            "continuous_bin_embedding_total": int(continuous_bin_embedding["total"]),
            "type_embedding_total": int(type_embedding["total"]),
            "missing_embedding_total": int(missing_embedding["total"]),
            "token_norm_total": int(token_norm["total"]),
            "expert_total": int(expert_total["total"]),
            "expert_trainable": int(expert_total["trainable"]),
            "expert_frozen": int(expert_total["frozen"]),
            "expert_cls_total": int(expert_cls_params),
            "expert_transformer_total": int(expert_transformer["total"]),
            "expert_transformer_trainable": int(expert_transformer["trainable"]),
        }

    fusion_info = module_param_info(encoder.fusion)
    prediction_head_info = module_param_info(head)

    active_feature_mask = torch.zeros(len(feature_names), dtype=torch.bool)
    for c in active_clusters:
        active_feature_mask |= cluster_assignments.cpu().eq(int(c))

    branch_param_rows = []
    active_summary = {
        "branch_forward_total": 0,
        "branch_forward_trainable": 0,
        "branch_forward_frozen": 0,
        "branch_stored_total": 0,
        "branch_stored_trainable": 0,
        "branch_stored_frozen": 0,
        "token_builder_forward_total": 0,
        "token_builder_forward_trainable": 0,
        "token_builder_forward_frozen": 0,
        "token_builder_stored_total": 0,
        "token_builder_stored_trainable": 0,
        "token_builder_stored_frozen": 0,
        "category_lookup_used_total": 0,
        "category_lookup_used_trainable": 0,
        "category_lookup_used_frozen": 0,
        "category_lookup_total": 0,
        "category_lookup_trainable": 0,
        "category_lookup_frozen": 0,
        "category_lookup_unused_total": 0,
        "category_projection_total": 0,
        "category_projection_trainable": 0,
        "expert_total": 0,
        "expert_trainable": 0,
        "expert_frozen": 0,
        "expert_cls_total": 0,
        "expert_transformer_total": 0,
        "expert_transformer_trainable": 0,
    }

    for c in active_clusters:
        branch = encoder.expert_bank.branches[int(c)]
        counts = component_counts_for_branch(branch)
        n_branch_features = int((cluster_assignments.cpu() == int(c)).sum().item())
        branch_forward_total = counts["token_builder_forward_total"] + counts["expert_total"]
        branch_forward_trainable = counts["token_builder_forward_trainable"] + counts["expert_trainable"]
        branch_stored_total = counts["token_builder_total"] + counts["expert_total"]
        branch_stored_trainable = counts["token_builder_trainable"] + counts["expert_trainable"]
        branch_row = {
            "cluster": int(c),
            "n_features": n_branch_features,
            "forward_total_params": int(branch_forward_total),
            "forward_trainable_params": int(branch_forward_trainable),
            "forward_frozen_params": int(branch_forward_total - branch_forward_trainable),
            "stored_total_params": int(branch_stored_total),
            "stored_trainable_params": int(branch_stored_trainable),
            "stored_frozen_params": int(branch_stored_total - branch_stored_trainable),
            "token_builder_forward_total": counts["token_builder_forward_total"],
            "token_builder_forward_trainable": counts["token_builder_forward_trainable"],
            "token_builder_forward_frozen": counts["token_builder_forward_frozen"],
            "category_lookup_used_total": counts["category_lookup_used_total"],
            "category_lookup_used_trainable": counts["category_lookup_used_trainable"],
            "category_lookup_used_frozen": counts["category_lookup_used_frozen"],
            "category_lookup_unused_total": counts["category_lookup_unused_total"],
            "category_projection_total": counts["category_projection_total"],
            "category_projection_trainable": counts["category_projection_trainable"],
            "expert_total": counts["expert_total"],
            "expert_trainable": counts["expert_trainable"],
        }
        branch_param_rows.append(branch_row)
        active_summary["branch_forward_total"] += branch_row["forward_total_params"]
        active_summary["branch_forward_trainable"] += branch_row["forward_trainable_params"]
        active_summary["branch_forward_frozen"] += branch_row["forward_frozen_params"]
        active_summary["branch_stored_total"] += branch_row["stored_total_params"]
        active_summary["branch_stored_trainable"] += branch_row["stored_trainable_params"]
        active_summary["branch_stored_frozen"] += branch_row["stored_frozen_params"]
        active_summary["token_builder_forward_total"] += counts["token_builder_forward_total"]
        active_summary["token_builder_forward_trainable"] += counts["token_builder_forward_trainable"]
        active_summary["token_builder_forward_frozen"] += counts["token_builder_forward_frozen"]
        active_summary["token_builder_stored_total"] += counts["token_builder_total"]
        active_summary["token_builder_stored_trainable"] += counts["token_builder_trainable"]
        active_summary["token_builder_stored_frozen"] += counts["token_builder_frozen"]
        active_summary["category_lookup_used_total"] += counts["category_lookup_used_total"]
        active_summary["category_lookup_used_trainable"] += counts["category_lookup_used_trainable"]
        active_summary["category_lookup_used_frozen"] += counts["category_lookup_used_frozen"]
        active_summary["category_lookup_total"] += counts["category_lookup_total"]
        active_summary["category_lookup_trainable"] += counts["category_lookup_trainable"]
        active_summary["category_lookup_frozen"] += counts["category_lookup_frozen"]
        active_summary["category_lookup_unused_total"] += counts["category_lookup_unused_total"]
        active_summary["category_projection_total"] += counts["category_projection_total"]
        active_summary["category_projection_trainable"] += counts["category_projection_trainable"]
        active_summary["expert_total"] += counts["expert_total"]
        active_summary["expert_trainable"] += counts["expert_trainable"]
        active_summary["expert_frozen"] += counts["expert_frozen"]
        active_summary["expert_cls_total"] += counts["expert_cls_total"]
        active_summary["expert_transformer_total"] += counts["expert_transformer_total"]
        active_summary["expert_transformer_trainable"] += counts["expert_transformer_trainable"]

    total_forward_params = active_summary["branch_forward_total"] + fusion_info["total"] + prediction_head_info["total"]
    total_forward_trainable = active_summary["branch_forward_trainable"] + fusion_info["trainable"] + prediction_head_info["trainable"]
    total_forward_frozen = total_forward_params - total_forward_trainable
    total_stored_active_params = active_summary["branch_stored_total"] + fusion_info["total"] + prediction_head_info["total"]
    total_stored_active_trainable = active_summary["branch_stored_trainable"] + fusion_info["trainable"] + prediction_head_info["trainable"]
    downstream_trainable_with_frozen_experts = fusion_info["total"] + prediction_head_info["total"]

    model_info = {
        "n_features_total": int(len(feature_names)),
        "n_features_used": int(active_feature_mask.sum()),
        "num_clusters_total": int(args.num_clusters),
        "active_clusters": active_clusters,
        "ignored_clusters": ignored_clusters,
        "d_model": int(args.d_model),
        "num_summary_tokens": int(num_summary_tokens),
        "num_fusion_group_tokens_total": int(args.num_clusters * num_summary_tokens),
        "num_fusion_group_tokens_active": int(len(active_clusters) * num_summary_tokens),
        "expert_n_layers": int(args.expert_n_layers),
        "expert_n_heads": int(args.expert_n_heads),
        "expert_head_dim": int(args.d_model // args.expert_n_heads),
        "fusion_n_layers": int(args.fusion_n_layers),
        "fusion_n_heads": int(args.fusion_n_heads),
        "fusion_head_dim": int(args.d_model // args.fusion_n_heads),
        "dropout": float(args.dropout),
        "use_cluster_embedding": bool(not args.no_cluster_embedding),
        "n_binary_targets": int(len(target_columns)),
        "params_forward_total": int(total_forward_params),
        "params_forward_trainable": int(total_forward_trainable),
        "params_forward_frozen": int(total_forward_frozen),
        "params_stored_active_total": int(total_stored_active_params),
        "params_stored_active_trainable": int(total_stored_active_trainable),
        "params_active_branches_forward": int(active_summary["branch_forward_total"]),
        "params_active_token_builders_forward": int(active_summary["token_builder_forward_total"]),
        "params_active_token_builders_forward_trainable": int(active_summary["token_builder_forward_trainable"]),
        "params_active_category_lookup_used": int(active_summary["category_lookup_used_total"]),
        "params_active_category_lookup_used_trainable": int(active_summary["category_lookup_used_trainable"]),
        "params_active_category_lookup_used_frozen": int(active_summary["category_lookup_used_frozen"]),
        "params_active_category_lookup_stored": int(active_summary["category_lookup_total"]),
        "params_active_category_lookup_stored_trainable": int(active_summary["category_lookup_trainable"]),
        "params_active_category_lookup_stored_frozen": int(active_summary["category_lookup_frozen"]),
        "params_active_category_projection": int(active_summary["category_projection_total"]),
        "params_active_category_projection_trainable": int(active_summary["category_projection_trainable"]),
        "params_active_expert_encoders": int(active_summary["expert_total"]),
        "params_active_expert_transformers_without_cls": int(active_summary["expert_transformer_total"]),
        "params_active_expert_cls_tokens": int(active_summary["expert_cls_total"]),
        "params_fusion": int(fusion_info["total"]),
        "params_prediction_head": int(prediction_head_info["total"]),
        "params_downstream_trainable_with_frozen_experts": int(downstream_trainable_with_frozen_experts),
        "branch_param_rows": branch_param_rows,
    }

    if args.print_model_summary:
        print(f"[INFO] device={device}")
        print(f"[INFO] split={args.split}, rows={len(dataset_for_loader)}, features={dataset.n_features}")
        print(f"[INFO] target_columns={target_columns}")
        print(f"[INFO] active_clusters={active_clusters}")
        print(f"[INFO] ignored_clusters={ignored_clusters}")

        print("\n" + "=" * 80)
        print("[ACTIVE FORWARD ARCHITECTURE / PARAMETER SUMMARY]")
        print(f"{'Total tokenizer features':42s} {model_info['n_features_total']}")
        print(f"{'Features used by active forward pass':42s} {model_info['n_features_used']}")
        print(f"{'Active clusters used in forward pass':42s} {active_clusters}")
        print(f"{'Ignored clusters not counted':42s} {ignored_clusters}")
        print(f"{'Token / embedding dimension':42s} {args.d_model}")
        print(f"{'Summary tokens per expert':42s} {num_summary_tokens}")
        print(
            f"{'Active group-summary tokens at fusion':42s} "
            f"{len(active_clusters) * num_summary_tokens}"
        )
        print(f"{'Expert Transformer layers':42s} {args.expert_n_layers}")
        print(f"{'Expert attention heads':42s} {args.expert_n_heads}")
        print(f"{'Expert attention head dim':42s} {model_info['expert_head_dim']}")
        print(f"{'Fusion Transformer layers':42s} {args.fusion_n_layers}")
        print(f"{'Fusion attention heads':42s} {args.fusion_n_heads}")
        print(f"{'Fusion attention head dim':42s} {model_info['fusion_head_dim']}")
        print(f"{'Dropout':42s} {args.dropout}")
        print(f"{'Cluster embedding':42s} {not args.no_cluster_embedding}")
        print(f"{'Binary output tasks':42s} {len(target_columns)}")
        print("-" * 80)
        print(f"{'Total params used by active forward pass':42s} {fmt_params(total_forward_params)}")
        print(f"{'  trainable in current model object':42s} {fmt_params(total_forward_trainable)}")
        print(f"{'  frozen in current model object':42s} {fmt_params(total_forward_frozen)}")
        print(f"{'Stored active params':42s} {fmt_params(total_stored_active_params)}")
        print(f"{'  stored but not used in active forward':42s} {fmt_params(total_stored_active_params - total_forward_params)}")
        print("-" * 80)
        print(f"{'Active cluster branches, forward-used':42s} {fmt_params(active_summary['branch_forward_total'])}")
        print(f"{'  token builders, forward-used':42s} {fmt_params(active_summary['token_builder_forward_total'])}")
        print(f"{'    category lookup tables used (blue)':42s} {fmt_params(active_summary['category_lookup_used_total'])}")
        print(f"{'      trainable':42s} {fmt_params(active_summary['category_lookup_used_trainable'])}")
        print(f"{'      frozen':42s} {fmt_params(active_summary['category_lookup_used_frozen'])}")
        print(f"{'    category projection adapters (orange)':42s} {fmt_params(active_summary['category_projection_total'])}")
        print(f"{'      trainable':42s} {fmt_params(active_summary['category_projection_trainable'])}")
        print(f"{'    unused category lookup params':42s} {fmt_params(active_summary['category_lookup_unused_total'])}")
        print(f"{'  expert encoders':42s} {fmt_params(active_summary['expert_total'])}")
        print(f"{'    expert CLS tokens':42s} {fmt_params(active_summary['expert_cls_total'])}")
        print(f"{'    expert Transformers':42s} {fmt_params(active_summary['expert_transformer_total'])}")
        print("-" * 80)
        print(f"{'Fusion params':42s} {fmt_params(fusion_info['total'])}")
        print(f"{'Prediction head params':42s} {fmt_params(prediction_head_info['total'])}")
        print(f"{'Downstream trainable if experts frozen':42s} {fmt_params(downstream_trainable_with_frozen_experts)}")
    
        #print("-" * 80)
        #print("[ACTIVE BRANCH PARAMETER SUMMARY]")
        # branch_table = pd.DataFrame(branch_param_rows)
        # branch_table_print = branch_table.copy()
        # param_cols = [
        #     "forward_total_params",
        #     "forward_trainable_params",
        #     "forward_frozen_params",
        #     "stored_total_params",
        #     "token_builder_forward_total",
        #     "token_builder_forward_trainable",
        #     "token_builder_forward_frozen",
        #     "category_lookup_used_total",
        #     "category_lookup_used_trainable",
        #     "category_lookup_used_frozen",
        #     "category_lookup_unused_total",
        #     "category_projection_total",
        #     "category_projection_trainable",
        #     "expert_total",
        #     "expert_trainable",
        # ]
        # for col in param_cols:
        #     branch_table_print[col] = branch_table_print[col].map(lambda x: fmt_params(int(x)))
        #print(branch_table_print.to_string(index=False))
        print("=" * 80)
    if args.print_model_only:
        return

    loader = DataLoader(dataset_for_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows, logits_all, probs_all = [], [], []
    ignored_feature_mask_device = ignored_feature_mask.to(device)
    summary_accumulator = (
        SummaryTokenDiagnosticsAccumulator(
            num_clusters=encoder.expert_bank.n_clusters,
            num_summary_tokens=num_summary_tokens,
            d_model=args.d_model,
        )
        if args.summary_diagnostics_dir is not None
        else None
    )

    forward_time_sec, forward_batches, forward_samples = 0.0, 0, 0
    sync_if_needed()
    e2e_start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="inference", leave=False):
            batch = move_batch_to_device(batch, device)
            B, N = batch["numeric_values"].shape

            if "observed_mask" in batch:
                feature_available_mask = batch["observed_mask"].bool()
            else:
                feature_available_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            feature_available_mask = feature_available_mask & (~ignored_feature_mask_device).unsqueeze(0)

            sync_if_needed()
            t0 = time.perf_counter()
            enc_out = encoder.forward_active_clusters(
                batch=batch,
                active_clusters=active_clusters,
                feature_available_mask=feature_available_mask,
                return_summary_attention=summary_accumulator is not None,
                fusion_summary_slot_indices=fusion_summary_slot_indices,
            )
            if summary_accumulator is not None:
                summary_accumulator.update(
                    group_tokens=enc_out["group_tokens"],
                    cluster_available_mask=enc_out["cluster_available_mask"],
                    attention_overlap=enc_out.get("summary_attention_overlap"),
                )
            logits = head(enc_out["patient_embedding"])["binary_logits"]
            probs = torch.sigmoid(logits)
            sync_if_needed()
            dt = time.perf_counter() - t0

            if len(rows) >= int(args.warmup_batches):
                forward_time_sec += dt
                forward_batches += 1
                forward_samples += int(B)

            rows.append(batch["row_idx"].detach().cpu())
            logits_all.append(logits.detach().cpu())
            probs_all.append(probs.detach().cpu())

    sync_if_needed()
    e2e_time_sec = time.perf_counter() - e2e_start

    runtime_info = {
        "batch_size": int(args.batch_size),
        "warmup_batches": int(args.warmup_batches),
        "timed_forward_batches": int(forward_batches),
        "timed_forward_samples": int(forward_samples),
        "forward_time_sec": float(forward_time_sec),
        "forward_ms_per_batch": float(1000.0 * forward_time_sec / max(forward_batches, 1)),
        "forward_ms_per_sample": float(1000.0 * forward_time_sec / max(forward_samples, 1)),
        "forward_samples_per_sec": float(forward_samples / max(forward_time_sec, 1e-12)),
        "end_to_end_loop_sec": float(e2e_time_sec),
    }

    print("\n" + "=" * 80)
    print("[INFERENCE RUNTIME]")
    print(f"{'Batch size':35s} {runtime_info['batch_size']}")
    print(f"{'Warmup batches skipped':35s} {runtime_info['warmup_batches']}")
    print(f"{'Timed forward batches':35s} {runtime_info['timed_forward_batches']}")
    print(f"{'Timed forward samples':35s} {runtime_info['timed_forward_samples']}")
    print(f"{'Forward time':35s} {runtime_info['forward_time_sec']:.4f} s")
    print(f"{'Forward ms / batch':35s} {runtime_info['forward_ms_per_batch']:.3f}")
    print(f"{'Forward ms / sample':35s} {runtime_info['forward_ms_per_sample']:.5f}")
    print(f"{'Forward samples / sec':35s} {runtime_info['forward_samples_per_sec']:.1f}")
    print(f"{'End-to-end loop time':35s} {runtime_info['end_to_end_loop_sec']:.4f} s")
    print("=" * 80)

    row_idx = torch.cat(rows).numpy().astype(int)
    logits = torch.cat(logits_all).numpy()
    probs = torch.cat(probs_all).numpy()
    targets = torch.load(token_dir / f"{args.split}_targets.pt", map_location="cpu", weights_only=True)

    output = {"row_idx": row_idx}
    metrics = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "active_clusters": active_clusters,
        "ignored_clusters": ignored_clusters,
        "target_columns": target_columns,
        "threshold": float(args.threshold),
        "num_summary_tokens": int(num_summary_tokens),
        "model_info": model_info,
        "runtime_info": runtime_info,
        "per_task": {},
    }

    macro_aurocs, macro_auprcs, macro_baccs, macro_f1s = [], [], [], []
    metric_summary_rows = []

    for j, name in enumerate(target_columns):
        output[f"logit_{name}"] = logits[:, j]
        output[f"prob_{name}"] = probs[:, j]
        output[f"pred_{name}"] = (probs[:, j] >= float(args.threshold)).astype(int)

        y_values, y_mask = [], []
        for i in row_idx:
            x = targets[name][int(i)]
            if torch.is_tensor(x):
                x = x.item()
            if x is None or pd.isna(x):
                y_values.append(np.nan)
                y_mask.append(False)
            else:
                y_values.append(float(x))
                y_mask.append(True)

        y_values = np.asarray(y_values, dtype=float)
        y_mask = np.asarray(y_mask, dtype=bool)
        output[f"true_{name}"] = y_values
        output[f"mask_{name}"] = y_mask.astype(int)

        y_true = y_values[y_mask].astype(int)
        y_prob = probs[:, j][y_mask]
        y_pred = (y_prob >= float(args.threshold)).astype(int)

        # print("\n" + "=" * 80)
        print(f"Evaluation: {name}")
        # print("=" * 80)
        task_metrics = evaluate_classification(
            y_true,
            y_pred,
            y_proba=y_prob,
            class_labels=[0, 1],
            ordinal=False,
        )
        # Fold-level confidence intervals are not used in the final CV report.
        # Uncertainty is reported as variation of the five fold-level point estimates.
        for key in list(task_metrics):
            if "ci" in key.lower():
                task_metrics.pop(key, None)
        for class_metrics in task_metrics.get("per_class", {}).values():
            if isinstance(class_metrics, dict):
                for key in list(class_metrics):
                    if "ci" in key.lower():
                        class_metrics.pop(key, None)

        pos_metrics = task_metrics.get("per_class", {}).get("1", {})
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        positive_f1 = float(2 * tp / max(2 * tp + fp + fn, 1))

        task_metrics["n"] = int(y_mask.sum())
        task_metrics["n_pos"] = int((y_true == 1).sum())
        task_metrics["n_neg"] = int((y_true == 0).sum())
        task_metrics["positive_rate"] = float((y_true == 1).mean())
        task_metrics["positive_class_auroc"] = pos_metrics.get("auroc")
        task_metrics["positive_class_auprc"] = pos_metrics.get("auprc")
        task_metrics["positive_class_ap_lift"] = pos_metrics.get("ap_lift")
        task_metrics["positive_class_f1"] = positive_f1
        metrics["per_task"][name] = task_metrics
        metric_summary_rows.append({
            "task": name,
            "auroc": task_metrics["positive_class_auroc"],
            "auprc": task_metrics["positive_class_auprc"],
            "n": int(y_mask.sum()),
            "n_pos": int((y_true == 1).sum()),
            "n_neg": int((y_true == 0).sum()),
        })

        macro_aurocs.append(task_metrics["positive_class_auroc"])
        macro_auprcs.append(task_metrics["positive_class_auprc"])
        macro_baccs.append(task_metrics["balanced_accuracy"])
        macro_f1s.append(positive_f1)

    metrics["macro_auroc"] = float(np.nanmean(np.asarray(macro_aurocs, dtype=float)))
    metrics["macro_auprc"] = float(np.nanmean(np.asarray(macro_auprcs, dtype=float)))
    metrics["macro_balanced_accuracy"] = float(np.nanmean(np.asarray(macro_baccs, dtype=float)))
    metrics["macro_positive_f1"] = float(np.nanmean(np.asarray(macro_f1s, dtype=float)))

    metric_summary_rows.append({
        "task": "MACRO_AVG",
        "auroc": metrics["macro_auroc"],
        "auprc": metrics["macro_auprc"],
        "n": "",
        "n_pos": "",
        "n_neg": "",
    })
    metric_summary_df = pd.DataFrame(metric_summary_rows)
    metrics["metric_summary"] = metric_summary_df.to_dict(orient="records")

    summary_diagnostic_paths: tuple[Path, Path] | None = None
    if summary_accumulator is not None:
        summary_diagnostics = summary_accumulator.compute()
        summary_diagnostic_paths = append_summary_diagnostics_csv(
            summary_diagnostics,
            output_dir=args.summary_diagnostics_dir,
            split=args.split,
        )
        metrics["summary_diagnostics"] = summary_diagnostics

    saved_paths: list[tuple[str, Path]] = []

    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(output).to_csv(output_csv, index=False)
        saved_paths.append(("predictions", output_csv))

    if args.metrics_json is not None:
        metrics_json = Path(args.metrics_json)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(
            json.dumps(
                metrics,
                indent=2,
                default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o),
            )
        )
        saved_paths.append(("metrics", metrics_json))



    if args.metric_summary_csv is not None:
        metric_summary_csv = Path(args.metric_summary_csv)
        metric_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        metric_summary_df.to_csv(metric_summary_csv, index=False)
        saved_paths.append(("metric summary", metric_summary_csv))

    if args.branch_param_csv is not None:
        branch_param_csv = Path(args.branch_param_csv)
        branch_param_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(branch_param_rows).to_csv(branch_param_csv, index=False)
        saved_paths.append(("branch params", branch_param_csv))

    if summary_diagnostic_paths is not None:
        token_stats_path, pair_stats_path = summary_diagnostic_paths
        saved_paths.append(("summary token stats", token_stats_path))
        saved_paths.append(("summary pair stats", pair_stats_path))

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print(f"macro_auroc: {metrics['macro_auroc']:.4f}")
    print(f"macro_auprc: {metrics['macro_auprc']:.4f}")
    if saved_paths:
        for label, path in saved_paths:
            print(f"Saved {label}: {path}")

    print("\n" + "=" * 80)
    print("[PER-TARGET AUROC/AUPRC]")
    table = metric_summary_df.copy()
    for col in ["auroc", "auprc"]:
        table[col] = table[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
