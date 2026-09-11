#!/usr/bin/env python3
"""Paired validation-only robustness gate for missingness-augmented FT.

Both the clean and augmented checkpoints receive byte-identical artificial
missingness masks. This script never loads held-out test data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import ft_transformer_v5 as clean_ft
import ft_transformer_missingness_augmented_v5 as augmented_ft


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/ft_transformer_v5_matched_missingness_uniform_groups_seed42"
CLEAN_OUTPUT = ROOT / "outputs/ft_transformer_v5_fivefold_final_seed42/matched"
FEATURE_DROP_LEVELS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)
GROUP_DROP_COUNTS = tuple(range(augmented_ft.N_GROUPS))
DRAWS = 5


def validation_view(
    tensors: tuple[torch.Tensor, ...],
    groups: torch.Tensor,
    family: str,
    severity: float,
    seed: int,
) -> tuple[TensorDataset, str]:
    numeric, categorical, missing, y, y_mask = tensors
    numeric = numeric.clone()
    categorical = categorical.clone()
    missing = missing.clone()
    observed = ~missing
    generator = torch.Generator().manual_seed(seed)
    artificial = torch.zeros(missing.shape, dtype=torch.bool)
    if family == "feature_fraction":
        artificial = (
            torch.rand(missing.shape, generator=generator) < severity
        ) & observed
    elif family == "group_count":
        count = int(severity)
        removed_columns = torch.zeros(groups.shape, dtype=torch.bool)
        for group_id in torch.randperm(augmented_ft.N_GROUPS, generator=generator)[
            :count
        ].tolist():
            removed_columns |= groups == group_id
        artificial = observed & removed_columns.unsqueeze(0)
    elif family == "leave_one_group_out":
        artificial = observed & (groups == int(severity)).unsqueeze(0)
    else:
        raise ValueError(f"unknown mask family {family}")

    numeric[artificial] = 0.0
    categorical[artificial] = 0
    missing[artificial] = True
    digest = hashlib.sha256(
        np.packbits(artificial.numpy(), axis=None).tobytes()
    ).hexdigest()
    return TensorDataset(numeric, categorical, missing, y, y_mask), digest


def load_model(checkpoint_path: Path, fold: int, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != augmented_ft.ARCHITECTURE:
        raise ValueError(f"unexpected architecture in {checkpoint_path}")
    metadata = clean_ft.metadata_for_fold(fold)
    model = clean_ft.model_for(augmented_ft.ARCHITECTURE, metadata)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def scenarios() -> list[tuple[str, float, int]]:
    result = []
    for level in FEATURE_DROP_LEVELS:
        result.extend(
            ("feature_fraction", level, draw)
            for draw in range(1 if level == 0 else DRAWS)
        )
    for count in GROUP_DROP_COUNTS:
        result.extend(
            ("group_count", float(count), draw)
            for draw in range(1 if count == 0 else DRAWS)
        )
    result.extend(
        ("leave_one_group_out", float(group), 0)
        for group in range(augmented_ft.N_GROUPS)
    )
    return result


def evaluate_fold(fold: int) -> None:
    fold_output = OUTPUT / f"fold{fold}"
    if not (fold_output / "DONE").exists():
        raise RuntimeError(f"fold {fold}: augmented training is not complete")
    output = fold_output / "validation_robustness"
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite {output}")

    metadata = clean_ft.metadata_for_fold(fold)
    target_names = list(metadata["target_columns"])
    groups = augmented_ft.cluster_ids(list(metadata["feature_names"]))
    val = clean_ft.load_split(fold, "val", target_names)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_sets = {
        "clean_ft_matched": load_model(
            CLEAN_OUTPUT / f"fold{fold}/best.pt", fold, device
        )[0],
        "augmented_ft_matched": load_model(
            fold_output / "best.pt", fold, device
        )[0],
    }

    task_rows, macro_rows = [], []
    for scenario_index, (family, severity, draw) in enumerate(scenarios()):
        seed = 900_001 + fold * 10_000 + scenario_index
        view, mask_hash = validation_view(val.tensors, groups, family, severity, seed)
        loader = DataLoader(
            view, batch_size=clean_ft.PROTOCOL["batch_size"], shuffle=False
        )
        for model_name, model in model_sets.items():
            with torch.no_grad():
                metrics = clean_ft.run_epoch(model, loader, device, target_names, None)
            if not np.isfinite(metrics["macro_auroc"]) or not np.isfinite(
                metrics["macro_auprc"]
            ):
                raise RuntimeError("non-finite robustness metric")
            macro_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "mask_family": family,
                    "severity": severity,
                    "draw": draw,
                    "mask_seed": seed,
                    "mask_sha256": mask_hash,
                    "validation_macro_auroc": metrics["macro_auroc"],
                    "validation_macro_auprc": metrics["macro_auprc"],
                    "validation_loss": metrics["loss"],
                }
            )
            for target in target_names:
                short = target.removeprefix("label_")
                task_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "mask_family": family,
                        "severity": severity,
                        "draw": draw,
                        "mask_seed": seed,
                        "mask_sha256": mask_hash,
                        "target": short,
                        "validation_auroc": metrics[f"{short}_auroc"],
                        "validation_auprc": metrics[f"{short}_auprc"],
                    }
                )
        print(
            f"fold={fold} {family} severity={severity:g} draw={draw} complete",
            flush=True,
        )

    macro = pd.DataFrame(macro_rows)
    tasks = pd.DataFrame(task_rows)
    summary = (
        macro.groupby(["fold", "model", "mask_family", "severity"], as_index=False)
        .agg(
            draws=("draw", "count"),
            validation_macro_auroc_mean=("validation_macro_auroc", "mean"),
            validation_macro_auroc_sd=("validation_macro_auroc", "std"),
            validation_macro_auprc_mean=("validation_macro_auprc", "mean"),
            validation_macro_auprc_sd=("validation_macro_auprc", "std"),
        )
        .fillna(0.0)
    )
    macro.to_csv(output / "validation_robustness_macro_by_draw.csv", index=False)
    tasks.to_csv(output / "validation_robustness_task_scores.csv", index=False)
    summary.to_csv(output / "validation_robustness_summary.csv", index=False)

    paired = macro.pivot_table(
        index=["fold", "mask_family", "severity", "draw", "mask_seed", "mask_sha256"],
        columns="model",
        values=["validation_macro_auroc", "validation_macro_auprc"],
    ).reset_index()
    paired.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in paired.columns
    ]
    for metric in ("auroc", "auprc"):
        stem = f"validation_macro_{metric}"
        paired[f"delta_{metric}"] = (
            paired[f"{stem}_augmented_ft_matched"]
            - paired[f"{stem}_clean_ft_matched"]
        )
    paired.to_csv(output / "paired_deltas_by_draw.csv", index=False)

    clean_condition = paired[
        (paired.mask_family == "feature_fraction") & (paired.severity == 0)
    ].iloc[0]
    nonclean = paired[
        ~(
            ((paired.mask_family == "feature_fraction") & (paired.severity == 0))
            | ((paired.mask_family == "group_count") & (paired.severity == 0))
        )
    ]
    gate = {
        "fold": fold,
        "clean_validation_auroc_delta": float(clean_condition.delta_auroc),
        "clean_validation_auprc_delta": float(clean_condition.delta_auprc),
        "nonclean_mean_paired_auroc_delta": float(nonclean.delta_auroc.mean()),
        "nonclean_mean_paired_auprc_delta": float(nonclean.delta_auprc.mean()),
        "clean_auroc_retention_threshold": -0.01,
    }
    gate["clean_performance_gate_pass"] = (
        gate["clean_validation_auroc_delta"] >= gate["clean_auroc_retention_threshold"]
    )
    gate["robustness_auroc_gate_pass"] = (
        gate["nonclean_mean_paired_auroc_delta"] > 0
    )
    gate["robustness_auprc_gate_pass"] = (
        gate["nonclean_mean_paired_auprc_delta"] > 0
    )
    gate["fivefold_expansion_recommended"] = bool(
        gate["clean_performance_gate_pass"]
        and gate["robustness_auroc_gate_pass"]
        and gate["robustness_auprc_gate_pass"]
    )
    (output / "FOLD_GATE.json").write_text(json.dumps(gate, indent=2))
    config = {
        "protocol": "paired fixed validation-only missingness gate",
        "feature_drop_levels": list(FEATURE_DROP_LEVELS),
        "group_drop_counts": list(GROUP_DROP_COUNTS),
        "leave_one_group_out": list(range(augmented_ft.N_GROUPS)),
        "draws_per_nonclean_random_scenario": DRAWS,
        "same_masks_for_both_models": True,
        "selection_or_tuning_from_robustness_results": False,
        "test_loaded_or_scored": False,
        "source_sha256": clean_ft.sha256(Path(__file__).resolve()),
    }
    (output / "CONFIG.json").write_text(json.dumps(config, indent=2))
    (output / "DONE").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))


def status(fold: int) -> None:
    output = OUTPUT / f"fold{fold}/validation_robustness"
    done = output / "DONE"
    if done.exists():
        print(done.read_text())
    elif output.exists():
        print(f"fold={fold}: incomplete")
    else:
        print(f"fold={fold}: pending")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--status", type=int, choices=range(5))
    args = parser.parse_args()
    if args.fold is not None:
        evaluate_fold(args.fold)
    elif args.status is not None:
        status(args.status)
    else:
        parser.error("use --fold N or --status N")


if __name__ == "__main__":
    main()
