#!/usr/bin/env python3
"""Post-hoc test LODO audit of five locked, unaugmented reference FT models.

No training, parameter selection or external-cohort evaluation. Historical
intact-input metrics must reproduce before any domain-removal result is accepted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import ft_transformer_v5 as ft
import ft_transformer_missingness_augmented_v5 as aug
from ft_transformer_missingness_validation_v5 import validation_view


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.project.resolve(), args.output.resolve()
    require(not output.exists(), f"Refusing to overwrite {output}")
    ft.DATA = root / "data/tokenized_nhanes_v5"
    aug.CLUSTER_MAP = ft.DATA / "cluster_maps/feature_clusters_biolord_v5_k7_leiden.csv"
    trained = root / "outputs/ft_transformer_v5_fivefold_final_seed42/reference"
    original = root / "outputs/ft_transformer_v5_fivefold_test_seed42/reference"
    previous_masks = root / "outputs/ft_transformer_v5_matched_missingness_uniform_groups_seed42/test_report"
    groups_source = root / "outputs/019faa3c-8a06-7e02-a965-6b8a857006c5/v5_group_removal_replot/feature_groups_k7_summary.csv"
    group_table = pd.read_csv(groups_source).set_index("group_id")
    config = json.loads((trained / "CONFIG.json").read_text())
    require(ft.sha256(Path(ft.__file__)) == config["source_sha256"], "Training source hash changed")
    require(ft.sha256(ft.DATA / "MANIFEST.json") == config["dataset_manifest_sha256"], "Dataset manifest changed")
    require(config["architecture"]["name"] == "reference", "Wrong architecture")
    output.mkdir(parents=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    ft.seed_all(42)
    save_json(output / "CONFIG.json", {
        "purpose": "post-hoc NHANES held-out test LODO, no model selection",
        "architecture": config["architecture"], "training_seed": 42,
        "folds": list(range(5)), "physical_features": 149, "targets": 11,
        "device": str(device), "torch_version": torch.__version__,
        "script_sha256": ft.sha256(Path(__file__)),
        "training_config": config, "cluster_map_sha256": ft.sha256(aug.CLUSTER_MAP),
        "mask_code_sha256": ft.sha256(Path(validation_view.__code__.co_filename)),
        "intact_reproduction_tolerance": {"per_task": 1e-4, "macro": 1e-5},
        "reproduction_note": "The unchanged original test evaluator also differed from its saved fold-1 task AUPRC by 7.72359533175726e-6 on this runtime. Checkpoints, test tensors and training code have identical hashes. Report actual numerical differences rather than claim bitwise reproduction.",
        "masking": "Same deterministic masks as matched FT. Observed cells in removed group: numeric=0, categorical=0, missing=True; natural missing cells and labels unchanged.",
        "uncertainty": "sample SD across five source folds at one training seed, not 25 independent seeds or a confidence interval",
    })
    started = time.monotonic()
    macro_rows, task_rows, checks = [], [], []
    for fold in range(5):
        fold_started = time.monotonic()
        checkpoint_path = trained / f"fold{fold}/best.pt"
        historical = json.loads((original / f"fold{fold}/TEST_DONE.json").read_text())
        done = json.loads((trained / f"fold{fold}/DONE").read_text())
        hashes = {
            "checkpoint_sha256": ft.sha256(checkpoint_path),
            "test_tokens_sha256": ft.sha256(ft.DATA / f"cv_splits/fold{fold}/test_tokens.pt"),
            "test_targets_sha256": ft.sha256(ft.DATA / f"cv_splits/fold{fold}/test_targets.pt"),
            "training_source_sha256": ft.sha256(Path(ft.__file__)),
            "dataset_manifest_sha256": ft.sha256(ft.DATA / "MANIFEST.json"),
        }
        for key, value in hashes.items():
            require(historical[key] == value, f"fold {fold}: {key} mismatch")
        require(done["stopping_reason"] == "early_stopping", "Unexpected training termination")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(checkpoint["architecture"] == "reference", "Wrong checkpoint architecture")
        require(checkpoint["config"] == config, "Checkpoint configuration mismatch")
        require(checkpoint["epoch"] == done["selected_epoch"] == historical["selected_epoch"], "Selected epoch mismatch")
        metadata = ft.metadata_for_fold(fold)
        targets = list(metadata["target_columns"])
        require(len(metadata["feature_names"]) == 149 and len(targets) == 11, "Feature/target count mismatch")
        groups = aug.cluster_ids(list(metadata["feature_names"]))
        for group in range(7):
            require(int((groups == group).sum()) == int(group_table.loc[group, "feature_count"]), "Group size mismatch")
        model = ft.model_for("reference", metadata)
        require(ft.parameter_count(model) == config["trainable_parameters"], "Parameter count mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        data = ft.load_split(fold, "test", targets)
        require(len(data) == historical["test_patients"], "Patient count mismatch")
        saved_tasks = pd.read_csv(original / f"fold{fold}/test_scores.csv").set_index("target")
        old_masks = pd.read_csv(previous_masks / f"fold{fold}/test_robustness_macro_by_draw.csv")
        max_error = 0.0
        full_metrics = None
        for group in [-1, *range(7)]:
            scenario_started = time.monotonic()
            family, severity = ("feature_fraction", 0.0) if group == -1 else ("leave_one_group_out", float(group))
            view, mask_hash = validation_view(data.tensors, groups, family, severity, 42)
            expected = old_masks[(old_masks.mask_family == family) & (old_masks.severity == severity)]
            require(len(expected) == 2 and set(expected.mask_sha256) == {mask_hash}, "Mask differs from existing clean/augmented matched FT")
            artificial = ~data.tensors[2] & ((groups == group).unsqueeze(0))
            require(torch.equal(view.tensors[2], data.tensors[2] | artificial), "Wrong missingness mask")
            for tensor_index in [0, 1]:
                require(torch.equal(view.tensors[tensor_index][~artificial], data.tensors[tensor_index][~artificial]), "Untargeted values changed")
                require(bool((view.tensors[tensor_index][artificial] == 0).all()), "Removed values not neutralized")
            require(torch.equal(view.tensors[3], data.tensors[3]) and torch.equal(view.tensors[4], data.tensors[4]), "Labels or eligibility changed")
            with torch.inference_mode():
                metrics = ft.run_epoch(model, DataLoader(view, batch_size=128, shuffle=False), device, targets, None)
            require(all(np.isfinite(value) for value in metrics.values()), "Non-finite metric")
            if group == -1:
                full_metrics = metrics
                errors = [abs(metrics[f"macro_{metric}"] - historical[f"test_macro_{metric}"]) for metric in ["auroc", "auprc"]]
                require(max(errors) <= 1e-5, f"Macro reproduction difference {max(errors)}")
                for j, target in enumerate(targets):
                    saved = saved_tasks.loc[target.removeprefix("label_")]
                    require(int(data.tensors[4][:, j].sum()) == int(saved.test_n), "Task eligibility mismatch")
                    for metric in ["auroc", "auprc"]:
                        errors.append(abs(metrics[f"{target.removeprefix('label_')}_{metric}"] - saved[f"test_{metric}"]))
                max_error = max(errors)
                require(max_error <= 1e-4, f"Intact predictions do not reproduce: max metric difference {max_error}")
            common = {"model": "clean_ft_reference", "split": "test", "fold": fold, "seed": 42,
                      "group_id": group, "group_name": "All groups" if group == -1 else group_table.loc[group, "group_name"],
                      "removed_feature_count": 0 if group == -1 else int(group_table.loc[group, "feature_count"]),
                      "mask_sha256": mask_hash}
            macro_rows.append({**common, "auroc": metrics["macro_auroc"], "auprc": metrics["macro_auprc"],
                               "delta_auroc": metrics["macro_auroc"] - full_metrics["macro_auroc"],
                               "delta_auprc": metrics["macro_auprc"] - full_metrics["macro_auprc"],
                               "runtime_seconds": time.monotonic() - scenario_started})
            for j, target in enumerate(targets):
                short = target.removeprefix("label_")
                eligible = data.tensors[4][:, j]
                task_rows.append({**common, "target": target, "n": int(eligible.sum()),
                                  "positive": int(data.tensors[3][eligible, j].sum()),
                                  **{metric: metrics[f"{short}_{metric}"] for metric in ["auroc", "auprc"]},
                                  **{f"delta_{metric}": metrics[f"{short}_{metric}"] - full_metrics[f"{short}_{metric}"] for metric in ["auroc", "auprc"]}})
            pd.DataFrame(macro_rows).to_csv(output / "macro_by_fold.csv", index=False)
            pd.DataFrame(task_rows).to_csv(output / "per_task_by_fold.csv", index=False)
            print(f"fold {fold} {common['group_name']}: AUROC={metrics['macro_auroc']:.6f} AUPRC={metrics['macro_auprc']:.6f}", flush=True)
        checks.append({"fold": fold, **hashes, "selected_epoch": done["selected_epoch"],
                       "intact_max_metric_difference": max_error, "all_eight_masks_match_existing_results": True,
                       "runtime_seconds": time.monotonic() - fold_started})
        save_json(output / "checks.json", checks)
        del model, checkpoint, data
        if device.type == "mps":
            torch.mps.empty_cache()
    macro, tasks = pd.DataFrame(macro_rows), pd.DataFrame(task_rows)
    require(len(macro) == 40 and len(tasks) == 440, "Incomplete results")
    for frame, keys, name in [(macro, ["group_id", "group_name", "removed_feature_count"], "macro_summary"),
                              (tasks, ["group_id", "group_name", "target"], "per_task_summary")]:
        summary = frame.groupby(keys, sort=True).agg(
            folds=("fold", "nunique"), auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
            delta_auroc_mean=("delta_auroc", "mean"), delta_auroc_sd=("delta_auroc", "std"),
            delta_auprc_mean=("delta_auprc", "mean"), delta_auprc_sd=("delta_auprc", "std"))
        require(bool((summary.folds == 5).all()), "Missing fold in summary")
        summary.to_csv(output / f"{name}.csv")
    save_json(output / "DONE.json", {"folds": 5, "inference_passes": 40, "task_scores": 440,
                                      "runtime_seconds": time.monotonic() - started,
                                      "intact_reproduction_max_difference": max(c["intact_max_metric_difference"] for c in checks),
                                      "all_checks_passed": True, "used_for_model_selection": False})


if __name__ == "__main__":
    main()
