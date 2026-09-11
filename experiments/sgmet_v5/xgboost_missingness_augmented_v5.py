#!/usr/bin/env python3
"""Train final-v5 XGBoost controls with SGMET-matched input corruption.

Only training rows are augmented. Validation stays clean for early stopping,
and this script never loads the held-out test tensors.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
DATASET = PROJECT / "data/tokenized_nhanes_v5"
CV = DATASET / "cv_splits"
MANIFEST = DATASET / "MANIFEST.json"
CLUSTER_MAP = DATASET / "cluster_maps/feature_clusters_biolord_v5_k7_leiden.csv"
CLEAN_XGB = PROJECT / "outputs/xgboost_v5_raw149_fivefold_seed42_optuna25"
DEFAULT_OUT = PROJECT / "outputs/xgboost_v5_raw149_missingness_uniform_groups_seed42_r8"
SEED = 42
N_FEATURES = 149
N_GROUPS = 7
N_ESTIMATORS = 2000
EARLY_STOPPING = 50
FEATURE_DROP_LEVELS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)
GROUP_DROP_COUNTS = tuple(range(N_GROUPS))
VALIDATION_DRAWS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_targets(token_dir: Path, split: str, columns: list[str]) -> dict[str, np.ndarray]:
    payload = torch.load(
        token_dir / f"{split}_targets.pt", map_location="cpu", weights_only=False
    )
    return {
        column.removeprefix("label_"): np.asarray(
            [np.nan if value is None else float(value) for value in payload[column]],
            dtype="float64",
        )
        for column in columns
    }


def load_matrix(
    token_dir: Path, split: str, continuous: np.ndarray, feature_count: int
) -> np.ndarray:
    payload = torch.load(
        token_dir / f"{split}_tokens.pt", map_location="cpu", weights_only=False
    )
    numeric = payload["numeric_values"].numpy().astype("float32")
    categorical = payload["categorical_codes"].numpy().astype("float32")
    missing = payload["missing_mask"].numpy().astype(bool)
    values = np.where(continuous[None, :], numeric, categorical)
    values[missing] = np.nan
    if values.shape[1] != feature_count:
        raise ValueError(f"{token_dir} {split}: expected {feature_count} features")
    return values


def cluster_ids(feature_names: list[str]) -> np.ndarray:
    frame = pd.read_csv(CLUSTER_MAP)
    if set(frame.columns) != {"feature_name", "cluster_id"}:
        raise ValueError(f"unexpected cluster-map columns in {CLUSTER_MAP}")
    if frame.feature_name.duplicated().any():
        raise ValueError("cluster map contains duplicate feature names")
    mapping = frame.set_index("feature_name").cluster_id.to_dict()
    missing = [name for name in feature_names if name not in mapping]
    extra = sorted(set(mapping) - set(feature_names))
    if missing or extra:
        raise ValueError(f"cluster-map mismatch: missing={missing}, extra={extra}")
    result = np.asarray([int(mapping[name]) for name in feature_names], dtype="int8")
    if set(result.tolist()) != set(range(N_GROUPS)):
        raise ValueError(f"expected K=7 ids 0..6, found {sorted(set(result.tolist()))}")
    return result


def load_fold(fold: int) -> dict:
    token_dir = CV / f"fold{fold}"
    metadata = torch.load(
        token_dir / "tokenizer_metadata.pt", map_location="cpu", weights_only=False
    )
    manifest = json.loads(MANIFEST.read_text())
    feature_names = list(metadata["feature_names"])
    target_columns = list(metadata["target_columns"])
    if feature_names != manifest["feature_order"] or len(feature_names) != N_FEATURES:
        raise ValueError(f"fold={fold}: clean-v5 feature contract mismatch")
    continuous = metadata["feature_type_ids"].numpy() == 0
    matrices = {
        split: load_matrix(token_dir, split, continuous, len(feature_names))
        for split in ("train", "val")
    }
    outcomes = {
        split: load_targets(token_dir, split, target_columns)
        for split in ("train", "val")
    }
    if np.isnan(matrices["train"]).all(axis=0).any():
        raise ValueError(f"fold={fold}: all-missing training column")
    for split in ("train", "val"):
        for target, values in outcomes[split].items():
            observed = values[np.isfinite(values)]
            if set(np.unique(observed)) != {0.0, 1.0}:
                raise ValueError(f"fold={fold} {split} {target}: lacks both classes")
    return {
        "matrices": matrices,
        "outcomes": outcomes,
        "feature_names": feature_names,
        "target_names": [column.removeprefix("label_") for column in target_columns],
        "cluster_ids": cluster_ids(feature_names),
    }


def load_test_fold(fold: int) -> dict:
    token_dir = CV / f"fold{fold}"
    metadata = torch.load(
        token_dir / "tokenizer_metadata.pt", map_location="cpu", weights_only=False
    )
    manifest = json.loads(MANIFEST.read_text())
    feature_names = list(metadata["feature_names"])
    target_columns = list(metadata["target_columns"])
    if feature_names != manifest["feature_order"] or len(feature_names) != N_FEATURES:
        raise ValueError(f"fold={fold}: clean-v5 test feature contract mismatch")
    continuous = metadata["feature_type_ids"].numpy() == 0
    values = load_matrix(token_dir, "test", continuous, len(feature_names))
    outcomes = load_targets(token_dir, "test", target_columns)
    for target, labels in outcomes.items():
        observed = labels[np.isfinite(labels)]
        if set(np.unique(observed)) != {0.0, 1.0}:
            raise ValueError(f"fold={fold} test {target}: lacks both classes")
    return {
        "values": values,
        "outcomes": outcomes,
        "target_names": [column.removeprefix("label_") for column in target_columns],
        "cluster_ids": cluster_ids(feature_names),
    }


def augment_training(
    values: np.ndarray,
    groups: np.ndarray,
    replicas: int,
    virtual_batch_size: int,
    feature_dropout_max: float,
    group_dropout_max: float,
    group_dropout_mode: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create deterministic Monte Carlo views matching SGMET's training corruption."""
    if replicas < 1 or virtual_batch_size < 1:
        raise ValueError("replicas and virtual_batch_size must be positive")
    if not 0.0 <= feature_dropout_max <= 1.0:
        raise ValueError("feature_dropout_max must be in [0, 1]")
    if not 0.0 <= group_dropout_max <= 1.0:
        raise ValueError("group_dropout_max must be in [0, 1]")
    if values.shape[1] != len(groups):
        raise ValueError("feature and group dimensions differ")

    rng = np.random.default_rng(seed)
    rows, columns = values.shape
    augmented = np.empty((rows * replicas, columns), dtype="float32")
    weights = np.full(rows * replicas, 1.0 / replicas, dtype="float32")
    batch_group_hist = np.zeros(N_GROUPS + 1, dtype="int64")
    group_keep_batches = np.zeros(N_GROUPS, dtype="int64")
    batches = 0
    sampled_feature_p = 0.0
    sampled_group_p = 0.0
    sampled_groups_dropped = 0
    observed_cells = 0
    feature_dropped_cells = 0
    combined_dropped_cells = 0

    for replica in range(replicas):
        permutation = rng.permutation(rows)
        destination = augmented[replica * rows : (replica + 1) * rows]
        for start in range(0, rows, virtual_batch_size):
            indices = permutation[start : start + virtual_batch_size]
            block = values[indices].copy()
            observed = ~np.isnan(block)

            p_feature = float(rng.random() * feature_dropout_max)
            feature_mask = (rng.random(block.shape) < p_feature) & observed

            if group_dropout_mode == "uniform_count":
                groups_dropped = int(rng.integers(0, N_GROUPS))
                keep_groups = np.ones(N_GROUPS, dtype=bool)
                if groups_dropped:
                    keep_groups[
                        rng.choice(N_GROUPS, size=groups_dropped, replace=False)
                    ] = False
                p_group = groups_dropped / N_GROUPS
            elif group_dropout_mode == "independent_probability":
                p_group = float(rng.random() * group_dropout_max)
                keep_groups = rng.random(N_GROUPS) >= p_group
                if not keep_groups.any():
                    keep_groups[rng.integers(N_GROUPS)] = True
                groups_dropped = int((~keep_groups).sum())
            else:
                raise ValueError(f"unknown group dropout mode: {group_dropout_mode}")
            dropped_columns = ~keep_groups[groups]
            combined_mask = feature_mask | (observed & dropped_columns[None, :])
            block[combined_mask] = np.nan
            destination[indices] = block

            batches += 1
            sampled_feature_p += p_feature
            sampled_group_p += p_group
            sampled_groups_dropped += groups_dropped
            groups_used = int(keep_groups.sum())
            batch_group_hist[groups_used] += 1
            group_keep_batches += keep_groups
            observed_cells += int(observed.sum())
            feature_dropped_cells += int(feature_mask.sum())
            combined_dropped_cells += int(combined_mask.sum())

    original_missing = np.isnan(values)
    for replica in range(replicas):
        view = augmented[replica * rows : (replica + 1) * rows]
        if not np.all(np.isnan(view)[original_missing]):
            raise RuntimeError("augmentation converted a naturally missing cell to observed")
    if not np.allclose(weights.reshape(replicas, rows).sum(axis=0), 1.0):
        raise RuntimeError("replica weights do not sum to one per patient")
    if batch_group_hist[0] != 0:
        raise RuntimeError("augmentation produced a virtual batch with zero active groups")

    audit = {
        "seed": seed,
        "replicas_per_patient": replicas,
        "virtual_batch_size": virtual_batch_size,
        "training_rows_original": rows,
        "training_rows_augmented": int(len(augmented)),
        "physical_features": columns,
        "virtual_batches": batches,
        "feature_dropout_probability_mean": sampled_feature_p / batches,
        "group_dropout_probability_mean": sampled_group_p / batches,
        "group_dropout_mode": group_dropout_mode,
        "groups_dropped_mean": sampled_groups_dropped / batches,
        "feature_only_removed_fraction_of_observed_cells": (
            feature_dropped_cells / observed_cells
        ),
        "combined_removed_fraction_of_observed_cells": (
            combined_dropped_cells / observed_cells
        ),
        "active_group_count_histogram_by_virtual_batch": {
            str(index): int(value)
            for index, value in enumerate(batch_group_hist)
            if value
        },
        "group_keep_fraction_by_virtual_batch": {
            str(index): float(value / batches)
            for index, value in enumerate(group_keep_batches)
        },
        "all_nan_augmented_row_fraction": float(np.isnan(augmented).all(axis=1).mean()),
        "patient_total_weight": 1.0,
    }
    return augmented, weights, audit


def fit_model(
    params: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_estimators: int = N_ESTIMATORS,
) -> xgb.XGBClassifier:
    scale_pos_weight = float(
        train_weight[train_y == 0].sum() / max(train_weight[train_y == 1].sum(), 1e-12)
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=n_estimators,
        early_stopping_rounds=min(EARLY_STOPPING, max(5, n_estimators // 4)),
        n_jobs=1,
        random_state=SEED,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=train_weight,
        eval_set=[(val_x, val_y)],
        verbose=False,
    )
    return model


def eligible(
    values: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mask = np.isfinite(labels)
    return values[mask], labels[mask].astype("int8"), None if weights is None else weights[mask]


def clean_params(fold: int) -> dict:
    path = CLEAN_XGB / f"fold{fold}/VALIDATION_DONE"
    if not path.exists():
        raise FileNotFoundError(f"locked clean XGBoost parameters not found: {path}")
    return dict(json.loads(path.read_text())["best_params"])


def validation_view(
    values: np.ndarray,
    groups: np.ndarray,
    family: str,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, str]:
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(values)
    artificial = np.zeros(values.shape, dtype=bool)
    if family == "feature_fraction":
        artificial = (rng.random(values.shape) < severity) & observed
    elif family == "group_count":
        count = int(severity)
        if count:
            dropped = rng.choice(N_GROUPS, size=count, replace=False)
            artificial = observed & np.isin(groups, dropped)[None, :]
    elif family == "leave_one_group_out":
        artificial = observed & (groups == int(severity))[None, :]
    else:
        raise ValueError(f"unknown validation-mask family {family}")
    masked = values.copy()
    masked[artificial] = np.nan
    digest = hashlib.sha256(np.packbits(artificial, axis=None).tobytes()).hexdigest()
    return masked, digest


def load_models(directory: Path, targets: list[str]) -> dict[str, xgb.XGBClassifier]:
    models = {}
    for target in targets:
        path = directory / f"{target}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        model = xgb.XGBClassifier()
        model.load_model(path)
        models[target] = model
    return models


def evaluate_validation(args: argparse.Namespace, fold: int) -> None:
    fold_dir = args.output / f"fold{fold}"
    if not (fold_dir / "VALIDATION_DONE").exists():
        raise RuntimeError(f"fold={fold}: augmented training must finish first")
    data = load_fold(fold)
    values = data["matrices"]["val"]
    outcomes = data["outcomes"]["val"]
    model_sets = {
        "clean_xgboost": load_models(CLEAN_XGB / f"fold{fold}/models", data["target_names"]),
        "augmented_xgboost": load_models(fold_dir / "models", data["target_names"]),
    }
    scenarios: list[tuple[str, float, int]] = []
    for level in FEATURE_DROP_LEVELS:
        draws = 1 if level == 0 else VALIDATION_DRAWS
        scenarios.extend(("feature_fraction", level, draw) for draw in range(draws))
    for count in GROUP_DROP_COUNTS:
        draws = 1 if count == 0 else VALIDATION_DRAWS
        scenarios.extend(("group_count", float(count), draw) for draw in range(draws))
    scenarios.extend(("leave_one_group_out", float(group), 0) for group in range(N_GROUPS))

    rows = []
    for scenario_index, (family, severity, draw) in enumerate(scenarios):
        seed = 900_001 + fold * 10_000 + scenario_index
        masked, mask_hash = validation_view(
            values, data["cluster_ids"], family, severity, seed
        )
        for model_name, models in model_sets.items():
            for target in data["target_names"]:
                target_x, target_y, _ = eligible(masked, outcomes[target])
                probability = models[target].predict_proba(target_x)[:, 1]
                rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "mask_family": family,
                        "severity": severity,
                        "draw": draw,
                        "mask_seed": seed,
                        "mask_sha256": mask_hash,
                        "target": target,
                        "validation_n": len(target_y),
                        "validation_auroc": float(roc_auc_score(target_y, probability)),
                        "validation_auprc": float(
                            average_precision_score(target_y, probability)
                        ),
                    }
                )

    task_scores = pd.DataFrame(rows)
    if not np.isfinite(
        task_scores[["validation_auroc", "validation_auprc"]].to_numpy()
    ).all():
        raise RuntimeError(f"fold={fold}: non-finite robustness-validation metric")
    macro = (
        task_scores.groupby(
            [
                "fold",
                "model",
                "mask_family",
                "severity",
                "draw",
                "mask_seed",
                "mask_sha256",
            ],
            as_index=False,
        )
        .agg(
            validation_macro_auroc=("validation_auroc", "mean"),
            validation_macro_auprc=("validation_auprc", "mean"),
        )
    )
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
    task_scores.to_csv(fold_dir / "validation_robustness_task_scores.csv", index=False)
    macro.to_csv(fold_dir / "validation_robustness_macro_by_draw.csv", index=False)
    summary.to_csv(fold_dir / "validation_robustness_summary.csv", index=False)
    config = {
        "protocol": "fixed_validation_only_missingness_decision_gate",
        "fold": fold,
        "feature_drop_levels": list(FEATURE_DROP_LEVELS),
        "group_drop_counts": list(GROUP_DROP_COUNTS),
        "leave_one_group_out": list(range(N_GROUPS)),
        "draws_per_nonclean_random_scenario": VALIDATION_DRAWS,
        "same_masks_for_both_models": True,
        "test_loaded_or_scored": False,
    }
    (fold_dir / "VALIDATION_ROBUSTNESS_CONFIG.json").write_text(
        json.dumps(config, indent=2)
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def evaluate_test(args: argparse.Namespace, fold: int) -> None:
    if not (args.output / "validation_report/REPORT.json").exists():
        raise RuntimeError("finish and summarize all validation folds before test scoring")
    test_dir = args.output / "test_report" / f"fold{fold}"
    test_dir.mkdir(parents=True, exist_ok=True)
    done = test_dir / "TEST_DONE"
    if done.exists():
        print(f"fold={fold}: held-out test already scored")
        return

    data = load_test_fold(fold)
    model_sets = {
        "clean_xgboost": load_models(CLEAN_XGB / f"fold{fold}/models", data["target_names"]),
        "augmented_xgboost": load_models(
            args.output / f"fold{fold}/models", data["target_names"]
        ),
    }
    scenarios: list[tuple[str, float, int]] = []
    for level in FEATURE_DROP_LEVELS:
        draws = 1 if level == 0 else VALIDATION_DRAWS
        scenarios.extend(("feature_fraction", level, draw) for draw in range(draws))
    for count in GROUP_DROP_COUNTS:
        draws = 1 if count == 0 else VALIDATION_DRAWS
        scenarios.extend(("group_count", float(count), draw) for draw in range(draws))
    scenarios.extend(
        ("leave_one_group_out", float(group), 0) for group in range(N_GROUPS)
    )

    rows = []
    for scenario_index, (family, severity, draw) in enumerate(scenarios):
        seed = 1_900_001 + fold * 10_000 + scenario_index
        masked, mask_hash = validation_view(
            data["values"], data["cluster_ids"], family, severity, seed
        )
        for model_name, models in model_sets.items():
            for target in data["target_names"]:
                target_x, target_y, _ = eligible(masked, data["outcomes"][target])
                probability = models[target].predict_proba(target_x)[:, 1]
                rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "mask_family": family,
                        "severity": severity,
                        "draw": draw,
                        "mask_seed": seed,
                        "mask_sha256": mask_hash,
                        "target": target,
                        "test_n": len(target_y),
                        "test_auroc": float(roc_auc_score(target_y, probability)),
                        "test_auprc": float(
                            average_precision_score(target_y, probability)
                        ),
                    }
                )

    task_scores = pd.DataFrame(rows)
    if not np.isfinite(task_scores[["test_auroc", "test_auprc"]].to_numpy()).all():
        raise RuntimeError(f"fold={fold}: non-finite held-out test metric")
    macro = (
        task_scores.groupby(
            [
                "fold",
                "model",
                "mask_family",
                "severity",
                "draw",
                "mask_seed",
                "mask_sha256",
            ],
            as_index=False,
        )
        .agg(
            test_macro_auroc=("test_auroc", "mean"),
            test_macro_auprc=("test_auprc", "mean"),
        )
    )
    summary = (
        macro.groupby(["fold", "model", "mask_family", "severity"], as_index=False)
        .agg(
            draws=("draw", "count"),
            test_macro_auroc_mean=("test_macro_auroc", "mean"),
            test_macro_auroc_sd=("test_macro_auroc", "std"),
            test_macro_auprc_mean=("test_macro_auprc", "mean"),
            test_macro_auprc_sd=("test_macro_auprc", "std"),
        )
        .fillna(0.0)
    )
    task_scores.to_csv(test_dir / "test_robustness_task_scores.csv", index=False)
    macro.to_csv(test_dir / "test_robustness_macro_by_draw.csv", index=False)
    summary.to_csv(test_dir / "test_robustness_summary.csv", index=False)
    config = {
        "protocol": "one_time_held_out_test_missingness_evaluation",
        "fold": fold,
        "feature_drop_levels": list(FEATURE_DROP_LEVELS),
        "group_drop_counts": list(GROUP_DROP_COUNTS),
        "leave_one_group_out": list(range(N_GROUPS)),
        "draws_per_nonclean_random_scenario": VALIDATION_DRAWS,
        "same_masks_for_both_models": True,
        "model_or_protocol_changes_after_test": False,
        "test_loaded_and_scored": True,
    }
    (test_dir / "TEST_CONFIG.json").write_text(json.dumps(config, indent=2))
    done.write_text(json.dumps(config, indent=2))
    print(f"fold={fold}: held-out test scoring complete")


def train_fold(args: argparse.Namespace, fold: int) -> dict:
    fold_dir = args.output / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    done = fold_dir / "VALIDATION_DONE"
    if done.exists():
        return json.loads(done.read_text())

    data = load_fold(fold)
    params = clean_params(fold)
    augmented, weights, audit = augment_training(
        data["matrices"]["train"],
        data["cluster_ids"],
        args.replicas,
        args.virtual_batch_size,
        args.feature_dropout_max,
        args.group_dropout_max,
        args.group_dropout_mode,
        SEED + 100_003 * fold,
    )
    audit_path = fold_dir / "augmentation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    config = {
        "protocol": "clean_v5_raw149_xgboost_sgmet_matched_training_corruption",
        "fold": fold,
        "seed": SEED,
        "augmentation_seed": SEED + 100_003 * fold,
        "replicas_per_patient": args.replicas,
        "replica_weight": 1.0 / args.replicas,
        "virtual_batch_size": args.virtual_batch_size,
        "feature_dropout": "p per virtual batch ~ Uniform(0, max), independent observed patient-feature cells",
        "feature_dropout_max": args.feature_dropout_max,
        "group_dropout": (
            "n per virtual batch ~ Uniform{0,...,6}, remove exactly n groups"
            if args.group_dropout_mode == "uniform_count"
            else "p per virtual batch ~ Uniform(0, max), independent groups, force at least one kept"
        ),
        "group_dropout_mode": args.group_dropout_mode,
        "group_dropout_max": args.group_dropout_max,
        "groups": N_GROUPS,
        "cluster_map": str(CLUSTER_MAP),
        "cluster_map_sha256": sha256(CLUSTER_MAP),
        "dataset_manifest": str(MANIFEST),
        "dataset_manifest_sha256": sha256(MANIFEST),
        "hyperparameters": "locked clean Optuna winner for this fold",
        "clean_validation_done": str(CLEAN_XGB / f"fold{fold}/VALIDATION_DONE"),
        "best_params": params,
        "selection_metric": "clean validation macro AUROC",
        "test_loaded_or_scored": False,
        "xgboost_version": xgb.__version__,
    }
    config_path = fold_dir / "CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"fold={fold}: configuration mismatch")
    config_path.write_text(json.dumps(config, indent=2))

    train_outcomes = data["outcomes"]["train"]
    val_outcomes = data["outcomes"]["val"]
    rows = []
    models_dir = fold_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for target in data["target_names"]:
        tiled_labels = np.tile(train_outcomes[target], args.replicas)
        train_x, train_y, train_weight = eligible(augmented, tiled_labels, weights)
        val_x, val_y, _ = eligible(
            data["matrices"]["val"], val_outcomes[target]
        )
        assert train_weight is not None
        model = fit_model(params, train_x, train_y, train_weight, val_x, val_y)
        probability = model.predict_proba(val_x)[:, 1]
        rows.append(
            {
                "fold": fold,
                "target": target,
                "training_rows_augmented_eligible": len(train_y),
                "training_effective_weight": float(train_weight.sum()),
                "validation_n": len(val_y),
                "validation_prevalence": float(val_y.mean()),
                "validation_auroc": float(roc_auc_score(val_y, probability)),
                "validation_auprc": float(average_precision_score(val_y, probability)),
                "best_iteration": int(model.best_iteration),
            }
        )
        model.save_model(models_dir / f"{target}.json")
        print(
            f"fold={fold} target={target}: validation AUROC "
            f"{rows[-1]['validation_auroc']:.6f}, AUPRC "
            f"{rows[-1]['validation_auprc']:.6f}",
            flush=True,
        )

    scores = pd.DataFrame(rows)
    if not np.isfinite(scores[["validation_auroc", "validation_auprc"]]).all().all():
        raise RuntimeError(f"fold={fold}: non-finite validation metric")
    scores.to_csv(fold_dir / "validation_scores.csv", index=False)
    result = {
        "fold": fold,
        "validation_macro_auroc": float(scores.validation_auroc.mean()),
        "validation_macro_auprc": float(scores.validation_auprc.mean()),
        "replicas_per_patient": args.replicas,
        "physical_features": N_FEATURES,
        "test_loaded_or_scored": False,
    }
    done.write_text(json.dumps(result, indent=2))
    return result


def preflight(folds: list[int]) -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["physical_feature_count"] != N_FEATURES:
        raise ValueError("clean-v5 manifest mismatch")
    for fold in folds:
        data = load_fold(fold)
        clean_params(fold)
        print(
            f"fold={fold}: train={data['matrices']['train'].shape}, "
            f"val={data['matrices']['val'].shape}, K={len(set(data['cluster_ids']))}, "
            "test not loaded",
            flush=True,
        )
    print("preflight passed")


def smoke_test(args: argparse.Namespace) -> None:
    data = load_fold(0)
    sample = data["matrices"]["train"][:1024]
    augmented, weights, audit = augment_training(
        sample,
        data["cluster_ids"],
        replicas=2,
        virtual_batch_size=128,
        feature_dropout_max=args.feature_dropout_max,
        group_dropout_max=args.group_dropout_max,
        group_dropout_mode=args.group_dropout_mode,
        seed=SEED,
    )
    repeated, _, _ = augment_training(
        sample,
        data["cluster_ids"],
        replicas=2,
        virtual_batch_size=128,
        feature_dropout_max=args.feature_dropout_max,
        group_dropout_max=args.group_dropout_max,
        group_dropout_mode=args.group_dropout_mode,
        seed=SEED,
    )
    changed, _, _ = augment_training(
        sample,
        data["cluster_ids"],
        replicas=2,
        virtual_batch_size=128,
        feature_dropout_max=args.feature_dropout_max,
        group_dropout_max=args.group_dropout_max,
        group_dropout_mode=args.group_dropout_mode,
        seed=SEED + 1,
    )
    if not np.array_equal(augmented, repeated, equal_nan=True):
        raise RuntimeError("same seed did not reproduce augmentation")
    if np.array_equal(augmented, changed, equal_nan=True):
        raise RuntimeError("different seeds produced identical augmentation")
    if augmented.shape != (2048, N_FEATURES) or len(weights) != 2048:
        raise RuntimeError("augmentation shape check failed")

    target = "arthritis"
    labels = np.tile(data["outcomes"]["train"][target][:1024], 2)
    train_x, train_y, train_weight = eligible(augmented, labels, weights)
    val_x, val_y, _ = eligible(
        data["matrices"]["val"], data["outcomes"]["val"][target]
    )
    assert train_weight is not None
    model = fit_model(
        clean_params(0),
        train_x,
        train_y,
        train_weight,
        val_x,
        val_y,
        n_estimators=20,
    )
    score = float(roc_auc_score(val_y, model.predict_proba(val_x)[:, 1]))
    if not np.isfinite(score) or not 0.5 < score < 1.0:
        raise RuntimeError("smoke-test metric failed")
    print(json.dumps(audit, indent=2))
    print(f"smoke test passed: fold=0 arthritis validation AUROC={score:.4f}")


def parse_folds(value: str) -> list[int]:
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds or any(fold not in range(5) for fold in folds) or len(set(folds)) != len(folds):
        raise argparse.ArgumentTypeError("folds must be unique values from 0,1,2,3,4")
    return folds


def status(output: Path) -> None:
    for fold in range(5):
        path = output / f"fold{fold}/VALIDATION_DONE"
        if path.exists():
            result = json.loads(path.read_text())
            print(
                f"fold={fold}: validation done, AUROC "
                f"{result['validation_macro_auroc']:.4f}, AUPRC "
                f"{result['validation_macro_auprc']:.4f}"
            )
        else:
            print(f"fold={fold}: pending")


def summarize_validation(output: Path, folds: list[int]) -> None:
    """Create the shareable five-fold validation tables and robustness figures."""
    summaries = []
    task_scores = []
    for fold in folds:
        fold_dir = output / f"fold{fold}"
        summary_path = fold_dir / "validation_robustness_summary.csv"
        task_path = fold_dir / "validation_robustness_task_scores.csv"
        if not summary_path.exists() or not task_path.exists():
            raise RuntimeError(f"fold={fold}: run --evaluate-validation first")
        summaries.append(pd.read_csv(summary_path))
        task_scores.append(pd.read_csv(task_path))

    all_summary = pd.concat(summaries, ignore_index=True)
    all_tasks = pd.concat(task_scores, ignore_index=True)
    report_dir = output / "validation_report"
    report_dir.mkdir(exist_ok=True)

    clean_foldwise = all_summary[
        (all_summary.mask_family == "feature_fraction")
        & (all_summary.severity == 0)
    ][
        [
            "fold",
            "model",
            "validation_macro_auroc_mean",
            "validation_macro_auprc_mean",
        ]
    ].rename(
        columns={
            "validation_macro_auroc_mean": "validation_macro_auroc",
            "validation_macro_auprc_mean": "validation_macro_auprc",
        }
    )
    clean_foldwise.to_csv(report_dir / "clean_validation_foldwise.csv", index=False)
    clean_summary = (
        clean_foldwise.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            validation_macro_auroc_mean=("validation_macro_auroc", "mean"),
            validation_macro_auroc_sd=("validation_macro_auroc", "std"),
            validation_macro_auprc_mean=("validation_macro_auprc", "mean"),
            validation_macro_auprc_sd=("validation_macro_auprc", "std"),
        )
    )
    clean_summary.to_csv(report_dir / "clean_validation_summary.csv", index=False)

    clean_tasks = all_tasks[
        (all_tasks.mask_family == "feature_fraction")
        & (all_tasks.severity == 0)
        & (all_tasks.draw == 0)
    ]
    task_summary = (
        clean_tasks.groupby(["target", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            validation_auroc_mean=("validation_auroc", "mean"),
            validation_auroc_sd=("validation_auroc", "std"),
            validation_auprc_mean=("validation_auprc", "mean"),
            validation_auprc_sd=("validation_auprc", "std"),
        )
    )
    task_summary.to_csv(report_dir / "clean_validation_per_task.csv", index=False)

    robustness = (
        all_summary.groupby(["model", "mask_family", "severity"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            validation_macro_auroc_mean=("validation_macro_auroc_mean", "mean"),
            validation_macro_auroc_sd=("validation_macro_auroc_mean", "std"),
            validation_macro_auprc_mean=("validation_macro_auprc_mean", "mean"),
            validation_macro_auprc_sd=("validation_macro_auprc_mean", "std"),
        )
    )
    robustness.to_csv(report_dir / "robustness_summary.csv", index=False)

    wide = all_summary.pivot_table(
        index=["fold", "mask_family", "severity"],
        columns="model",
        values=["validation_macro_auroc_mean", "validation_macro_auprc_mean"],
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in wide.columns
    ]
    for metric in ("auroc", "auprc"):
        stem = f"validation_macro_{metric}_mean"
        wide[f"delta_{metric}"] = (
            wide[f"{stem}_augmented_xgboost"] - wide[f"{stem}_clean_xgboost"]
        )
    deltas = (
        wide.groupby(["mask_family", "severity"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            delta_auroc_mean=("delta_auroc", "mean"),
            delta_auroc_sd=("delta_auroc", "std"),
            delta_auprc_mean=("delta_auprc", "mean"),
            delta_auprc_sd=("delta_auprc", "std"),
        )
    )
    deltas.to_csv(report_dir / "paired_robustness_deltas.csv", index=False)

    colors = {
        "clean_xgboost": "#6b7280",
        "augmented_xgboost": "#087e8b",
    }
    labels = {
        "clean_xgboost": "Clean-trained XGBoost",
        "augmented_xgboost": "Missingness-augmented XGBoost",
    }
    for family, title, xlabel, filename in (
        (
            "feature_fraction",
            "Robustness to unavailable individual features",
            "Artificially removed observed feature cells",
            "feature_missingness_robustness.png",
        ),
        (
            "group_count",
            "Robustness to unavailable semantic feature groups",
            "Number of semantic groups removed",
            "group_missingness_robustness.png",
        ),
    ):
        subset = robustness[robustness.mask_family == family]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for axis, metric, ylabel in (
            (axes[0], "auroc", "Validation macro AUROC"),
            (axes[1], "auprc", "Validation macro AUPRC"),
        ):
            for model in ("clean_xgboost", "augmented_xgboost"):
                model_rows = subset[subset.model == model].sort_values("severity")
                x = model_rows.severity.to_numpy()
                mean = model_rows[f"validation_macro_{metric}_mean"].to_numpy()
                sd = model_rows[f"validation_macro_{metric}_sd"].to_numpy()
                axis.plot(
                    x,
                    mean,
                    marker="o",
                    linewidth=2.2,
                    color=colors[model],
                    label=labels[model],
                )
                axis.fill_between(x, mean - sd, mean + sd, color=colors[model], alpha=0.15)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            if family == "feature_fraction":
                axis.set_xticks(FEATURE_DROP_LEVELS, [f"{int(x * 100)}%" for x in FEATURE_DROP_LEVELS])
            else:
                axis.set_xticks(GROUP_DROP_COUNTS)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{title}\nMean ± sample SD across five folds", fontsize=14)
        fig.savefig(report_dir / filename, dpi=200)
        plt.close(fig)

    report = {
        "protocol": "five_fold_validation_only_missingness_comparison",
        "folds": folds,
        "selection_or_tuning_from_robustness_results": False,
        "same_validation_masks_for_both_models": True,
        "test_loaded_or_scored": False,
        "artifacts": sorted(path.name for path in report_dir.iterdir()),
    }
    (report_dir / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(clean_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"validation report written to {report_dir}")


def summarize_test(output: Path, folds: list[int]) -> None:
    report_dir = output / "test_report"
    summaries = []
    task_scores = []
    for fold in folds:
        fold_dir = report_dir / f"fold{fold}"
        if not (fold_dir / "TEST_DONE").exists():
            raise RuntimeError(f"fold={fold}: held-out test scoring is incomplete")
        summaries.append(pd.read_csv(fold_dir / "test_robustness_summary.csv"))
        task_scores.append(pd.read_csv(fold_dir / "test_robustness_task_scores.csv"))
    all_summary = pd.concat(summaries, ignore_index=True)
    all_tasks = pd.concat(task_scores, ignore_index=True)

    clean_foldwise = all_summary[
        (all_summary.mask_family == "feature_fraction")
        & (all_summary.severity == 0)
    ][["fold", "model", "test_macro_auroc_mean", "test_macro_auprc_mean"]].rename(
        columns={
            "test_macro_auroc_mean": "test_macro_auroc",
            "test_macro_auprc_mean": "test_macro_auprc",
        }
    )
    clean_foldwise.to_csv(report_dir / "clean_test_foldwise.csv", index=False)
    clean_summary = (
        clean_foldwise.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            test_macro_auroc_mean=("test_macro_auroc", "mean"),
            test_macro_auroc_sd=("test_macro_auroc", "std"),
            test_macro_auprc_mean=("test_macro_auprc", "mean"),
            test_macro_auprc_sd=("test_macro_auprc", "std"),
        )
    )
    clean_summary.to_csv(report_dir / "clean_test_summary.csv", index=False)

    clean_tasks = all_tasks[
        (all_tasks.mask_family == "feature_fraction")
        & (all_tasks.severity == 0)
        & (all_tasks.draw == 0)
    ]
    task_summary = (
        clean_tasks.groupby(["target", "model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            test_auroc_mean=("test_auroc", "mean"),
            test_auroc_sd=("test_auroc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_sd=("test_auprc", "std"),
        )
    )
    task_summary.to_csv(report_dir / "clean_test_per_task.csv", index=False)

    robustness = (
        all_summary.groupby(["model", "mask_family", "severity"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            test_macro_auroc_mean=("test_macro_auroc_mean", "mean"),
            test_macro_auroc_sd=("test_macro_auroc_mean", "std"),
            test_macro_auprc_mean=("test_macro_auprc_mean", "mean"),
            test_macro_auprc_sd=("test_macro_auprc_mean", "std"),
        )
    )
    robustness.to_csv(report_dir / "robustness_summary.csv", index=False)
    wide = all_summary.pivot_table(
        index=["fold", "mask_family", "severity"],
        columns="model",
        values=["test_macro_auroc_mean", "test_macro_auprc_mean"],
    ).reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in wide.columns
    ]
    for metric in ("auroc", "auprc"):
        stem = f"test_macro_{metric}_mean"
        wide[f"delta_{metric}"] = (
            wide[f"{stem}_augmented_xgboost"] - wide[f"{stem}_clean_xgboost"]
        )
    deltas = (
        wide.groupby(["mask_family", "severity"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            delta_auroc_mean=("delta_auroc", "mean"),
            delta_auroc_sd=("delta_auroc", "std"),
            delta_auprc_mean=("delta_auprc", "mean"),
            delta_auprc_sd=("delta_auprc", "std"),
        )
    )
    deltas.to_csv(report_dir / "paired_robustness_deltas.csv", index=False)

    colors = {"clean_xgboost": "#6b7280", "augmented_xgboost": "#087e8b"}
    labels = {
        "clean_xgboost": "Clean-trained XGBoost",
        "augmented_xgboost": "Missingness-augmented XGBoost",
    }
    for family, title, xlabel, filename in (
        (
            "feature_fraction",
            "Held-out test robustness to unavailable individual features",
            "Artificially removed observed feature cells",
            "feature_missingness_robustness.png",
        ),
        (
            "group_count",
            "Held-out test robustness to unavailable semantic feature groups",
            "Number of semantic groups removed",
            "group_missingness_robustness.png",
        ),
    ):
        subset = robustness[robustness.mask_family == family]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
        for axis, metric, ylabel in (
            (axes[0], "auroc", "Test macro AUROC"),
            (axes[1], "auprc", "Test macro AUPRC"),
        ):
            for model in ("clean_xgboost", "augmented_xgboost"):
                model_rows = subset[subset.model == model].sort_values("severity")
                x = model_rows.severity.to_numpy()
                mean = model_rows[f"test_macro_{metric}_mean"].to_numpy()
                sd = model_rows[f"test_macro_{metric}_sd"].to_numpy()
                axis.plot(
                    x,
                    mean,
                    marker="o",
                    linewidth=2.2,
                    color=colors[model],
                    label=labels[model],
                )
                axis.fill_between(x, mean - sd, mean + sd, color=colors[model], alpha=0.15)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            if family == "feature_fraction":
                axis.set_xticks(
                    FEATURE_DROP_LEVELS,
                    [f"{int(value * 100)}%" for value in FEATURE_DROP_LEVELS],
                )
            else:
                axis.set_xticks(GROUP_DROP_COUNTS)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{title}\nMean ± sample SD across five folds", fontsize=14)
        fig.savefig(report_dir / filename, dpi=200)
        plt.close(fig)

    report = {
        "protocol": "one_time_five_fold_held_out_test_missingness_comparison",
        "folds": folds,
        "model_or_protocol_changes_after_test": False,
        "same_test_masks_for_both_models": True,
        "test_loaded_and_scored": True,
        "artifacts": sorted(path.name for path in report_dir.iterdir()),
    }
    (report_dir / "REPORT.json").write_text(json.dumps(report, indent=2))
    (output / "TEST_COMPLETE").write_text(json.dumps(report, indent=2))
    print(clean_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"held-out test report written to {report_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0"))
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--virtual-batch-size", type=int, default=128)
    parser.add_argument("--feature-dropout-max", type=float, default=0.10)
    parser.add_argument("--group-dropout-max", type=float, default=0.90)
    parser.add_argument(
        "--group-dropout-mode",
        choices=("uniform_count", "independent_probability"),
        default="uniform_count",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--evaluate-validation", action="store_true")
    parser.add_argument("--summarize-validation", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--summarize-test", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        preflight(args.folds)
        return 0
    if args.smoke_test:
        smoke_test(args)
        return 0
    if args.evaluate_validation:
        for fold in args.folds:
            evaluate_validation(args, fold)
        return 0
    if args.summarize_validation:
        summarize_validation(args.output, args.folds)
        return 0
    if args.evaluate_test:
        for fold in args.folds:
            evaluate_test(args, fold)
        return 0
    if args.summarize_test:
        summarize_test(args.output, args.folds)
        return 0
    if args.status:
        status(args.output)
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another augmented XGBoost runner is active") from error
    lock.write(str(os.getpid()))
    lock.flush()

    preflight(args.folds)
    started = time.time()
    results = [train_fold(args, fold) for fold in args.folds]
    payload = {
        "protocol": "validation_only_xgboost_v5_sgmet_matched_training_corruption",
        "folds_completed": args.folds,
        "results": results,
        "wall_minutes": round((time.time() - started) / 60, 1),
        "test_loaded_or_scored": False,
    }
    (args.output / "VALIDATION_COMPLETE").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
