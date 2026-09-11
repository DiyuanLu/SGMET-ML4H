#!/usr/bin/env python3
"""Five-fold Optuna-tuned XGBoost baseline on the clean v5 raw-value view."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/tokenized_nhanes_v5"
CV = DATASET / "cv_splits"
MANIFEST = DATASET / "MANIFEST.json"
SGMET_TRAINING = (
    ROOT
    / "outputs/v5_literal149_k7_then_clinical_k10_fivefold_seed42_focal_cd90_ep150"
)
SGMET_TEST = ROOT / "outputs/v5_k7_fivefold_test_seed42"
OUT = ROOT / "outputs/xgboost_v5_raw149_fivefold_seed42_optuna25"
SHARE = ROOT / "share/sgmet_k7_vs_xgboost_v5_fivefold"
LEGACY_SCRIPT = Path(__file__).with_name("xgboost_fit_common.py")
FOLDS = range(5)
SEED = 42
N_TRIALS = 25
WAIT_SECONDS = 60


def load_legacy():
    spec = importlib.util.spec_from_file_location("xgb_stage1_legacy", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy()
TARGET_LABELS = LEGACY.TARGET_LABELS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def targets(token_dir: Path, split: str, columns: list[str]) -> dict[str, np.ndarray]:
    payload = torch.load(
        token_dir / f"{split}_targets.pt", map_location="cpu", weights_only=False
    )
    result = {}
    for column in columns:
        result[column.removeprefix("label_")] = np.asarray(
            [np.nan if value is None else float(value) for value in payload[column]],
            dtype="float64",
        )
    return result


def matrix(
    token_dir: Path,
    split: str,
    continuous: np.ndarray,
    feature_count: int,
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


def load_fold(
    fold: int, include_test: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict]:
    token_dir = CV / f"fold{fold}"
    metadata = torch.load(
        token_dir / "tokenizer_metadata.pt", map_location="cpu", weights_only=False
    )
    manifest = json.loads(MANIFEST.read_text())
    feature_names = list(metadata["feature_names"])
    target_columns = list(metadata["target_columns"])
    if feature_names != manifest["feature_order"] or len(feature_names) != 149:
        raise ValueError(f"fold={fold}: clean-v5 feature contract mismatch")
    if [column.removeprefix("label_") for column in target_columns] != list(
        TARGET_LABELS
    ):
        raise ValueError(f"fold={fold}: target order mismatch")
    continuous = metadata["feature_type_ids"].numpy() == 0

    splits = ["train", "val"] + (["test"] if include_test else [])
    matrices = {
        split: matrix(token_dir, split, continuous, len(feature_names))
        for split in splits
    }
    outcomes = {
        split: targets(token_dir, split, target_columns) for split in splits
    }
    if np.isnan(matrices["train"]).all(axis=0).any():
        bad = np.flatnonzero(np.isnan(matrices["train"]).all(axis=0))
        raise ValueError(
            f"fold={fold}: all-missing clean-v5 columns "
            f"{[feature_names[index] for index in bad]}"
        )
    for split in splits:
        if len(matrices[split]) != len(next(iter(outcomes[split].values()))):
            raise ValueError(f"fold={fold} {split}: row mismatch")
        for target, values in outcomes[split].items():
            observed = values[np.isfinite(values)]
            if set(np.unique(observed)) != {0.0, 1.0}:
                raise ValueError(f"fold={fold} {split} {target}: lacks both classes")

    info = {
        "fold": fold,
        "feature_view": "raw_value_per_clean_v5_feature_with_native_nan",
        "physical_features": len(feature_names),
        "feature_names": feature_names,
        "train_rows": len(matrices["train"]),
        "validation_rows": len(matrices["val"]),
        "test_loaded_or_scored": include_test,
    }
    if include_test:
        info["test_rows"] = len(matrices["test"])
    return matrices, outcomes, info


def subset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return LEGACY.subset(X, y)


def fit_model(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = LEGACY.N_ESTIMATORS,
) -> xgb.XGBClassifier:
    return LEGACY.fit_model(
        params, X_train, y_train, X_val, y_val, n_estimators=n_estimators
    )


def complete_trials(study: optuna.Study) -> list[optuna.Trial]:
    return [trial for trial in study.trials if trial.state.name == "COMPLETE"]


def write_fold_status(study: optuna.Study, fold: int) -> None:
    complete = complete_trials(study)
    states = pd.Series([trial.state.name for trial in study.trials]).value_counts()
    payload = {
        "fold": fold,
        "requested_trials": N_TRIALS,
        "recorded_trials": len(study.trials),
        "complete_trials": len(complete),
        "states": {key: int(value) for key, value in states.items()},
        "best_validation_macro_auroc": (
            float(study.best_value) if complete else None
        ),
        "best_trial": int(study.best_trial.number) if complete else None,
        "test_loaded_or_scored": False,
        "updated_at": time.strftime("%F %T"),
    }
    fold_dir = OUT / f"fold{fold}"
    (fold_dir / "STATUS.json").write_text(json.dumps(payload, indent=2))
    study.trials_dataframe().to_csv(fold_dir / "trials.csv", index=False)


def tune_fold(fold: int) -> dict:
    fold_dir = OUT / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir / "VALIDATION_DONE").exists():
        return json.loads((fold_dir / "VALIDATION_DONE").read_text())

    matrices, outcomes, info = load_fold(fold, include_test=False)
    config = {
        "protocol": "clean_v5_raw149_xgboost_optuna",
        "fold": fold,
        "seed": SEED,
        "n_trials": N_TRIALS,
        "selection_metric": "validation_macro_auroc",
        "shared_hyperparameters_across_11_target_models": True,
        "n_estimators_cap": LEGACY.N_ESTIMATORS,
        "early_stopping_rounds": LEGACY.EARLY_STOPPING,
        "sampler": "Optuna TPESampler",
        "sampler_startup_trials": 10,
        "class_imbalance": "per-target scale_pos_weight from train",
        "test_loaded_or_scored": False,
        "dataset_manifest": str(MANIFEST),
        "dataset_manifest_sha256": sha256(MANIFEST),
        "data": info,
        "xgboost_version": xgb.__version__,
        "optuna_version": optuna.__version__,
    }
    config_path = fold_dir / "CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"fold={fold}: configuration mismatch")
    config_path.write_text(json.dumps(config, indent=2))

    storage = f"sqlite:///{fold_dir / 'optuna.db'}"
    study = optuna.create_study(
        study_name=f"xgboost_v5_raw149_fold{fold}_seed42_optuna25",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=10),
    )

    def objective(trial: optuna.Trial) -> float:
        params = LEGACY.params_from_trial(trial)
        scores = []
        for target in TARGET_LABELS:
            xtr, ytr = subset(matrices["train"], outcomes["train"][target])
            xva, yva = subset(matrices["val"], outcomes["val"][target])
            model = fit_model(params, xtr, ytr, xva, yva)
            score = float(roc_auc_score(yva, model.predict_proba(xva)[:, 1]))
            if not np.isfinite(score):
                raise RuntimeError(f"fold={fold} {target}: non-finite validation AUROC")
            trial.set_user_attr(f"{target}_val_auroc", score)
            trial.set_user_attr(f"{target}_best_iteration", int(model.best_iteration))
            scores.append(score)
        macro = float(np.mean(scores))
        print(
            f"fold={fold} trial={trial.number:02d}: "
            f"validation macro AUROC={macro:.6f}",
            flush=True,
        )
        return macro

    remaining = N_TRIALS - len(complete_trials(study))
    if remaining < 0:
        raise RuntimeError(f"fold={fold}: study has more than {N_TRIALS} trials")
    print(
        f"fold={fold}: train={matrices['train'].shape}, "
        f"val={matrices['val'].shape}, remaining_trials={remaining}, test untouched",
        flush=True,
    )
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            callbacks=[lambda current, _: write_fold_status(current, fold)],
        )
    write_fold_status(study, fold)
    if len(study.trials) != N_TRIALS or len(complete_trials(study)) != N_TRIALS:
        raise RuntimeError(f"fold={fold}: not all {N_TRIALS} trials completed")

    best_params = dict(study.best_params)
    models_dir = fold_dir / "models"
    models_dir.mkdir(exist_ok=True)
    rows = []
    for target, label in TARGET_LABELS.items():
        xtr, ytr = subset(matrices["train"], outcomes["train"][target])
        xva, yva = subset(matrices["val"], outcomes["val"][target])
        model = fit_model(best_params, xtr, ytr, xva, yva)
        probability = model.predict_proba(xva)[:, 1]
        rows.append(
            {
                "fold": fold,
                "target": target,
                "label": label,
                "validation_n": len(yva),
                "validation_prevalence": float(yva.mean()),
                "validation_auroc": float(roc_auc_score(yva, probability)),
                "validation_auprc": float(
                    average_precision_score(yva, probability)
                ),
                "best_iteration": int(model.best_iteration),
            }
        )
        model.save_model(models_dir / f"{target}.json")
    scores = pd.DataFrame(rows)
    if not np.isfinite(
        scores[["validation_auroc", "validation_auprc"]].to_numpy()
    ).all():
        raise RuntimeError(f"fold={fold}: non-finite validation result")
    scores.to_csv(fold_dir / "validation_scores.csv", index=False)
    result = {
        "fold": fold,
        "best_trial": int(study.best_trial.number),
        "best_params": best_params,
        "validation_macro_auroc": float(scores.validation_auroc.mean()),
        "validation_macro_auprc": float(scores.validation_auprc.mean()),
        "n_complete_trials": N_TRIALS,
        "physical_features": 149,
        "test_loaded_or_scored": False,
    }
    (fold_dir / "VALIDATION_DONE").write_text(json.dumps(result, indent=2))
    print(
        f"fold={fold}: validation complete, AUROC "
        f"{result['validation_macro_auroc']:.6f}, AUPRC "
        f"{result['validation_macro_auprc']:.6f}",
        flush=True,
    )
    return result


def score_test_fold(fold: int) -> dict:
    fold_dir = OUT / f"fold{fold}"
    result_path = fold_dir / "TEST_DONE"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if not (OUT / "VALIDATION_COMPLETE").exists():
        raise RuntimeError("all validation tuning must complete before any test scoring")

    matrices, outcomes, info = load_fold(fold, include_test=True)
    rows = []
    for target, label in TARGET_LABELS.items():
        xte, yte = subset(matrices["test"], outcomes["test"][target])
        model_path = fold_dir / "models" / f"{target}.json"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        probability = model.predict_proba(xte)[:, 1]
        rows.append(
            {
                "fold": fold,
                "target": target,
                "label": label,
                "test_n": len(yte),
                "test_prevalence": float(yte.mean()),
                "test_auroc": float(roc_auc_score(yte, probability)),
                "test_auprc": float(average_precision_score(yte, probability)),
            }
        )
    scores = pd.DataFrame(rows)
    if not np.isfinite(scores[["test_auroc", "test_auprc"]].to_numpy()).all():
        raise RuntimeError(f"fold={fold}: non-finite test result")
    scores.to_csv(fold_dir / "test_scores.csv", index=False)
    validation = json.loads((fold_dir / "VALIDATION_DONE").read_text())
    result = {
        "protocol": "validation_winner_scored_once_on_held_out_test",
        "fold": fold,
        "best_trial": validation["best_trial"],
        "test_macro_auroc": float(scores.test_auroc.mean()),
        "test_macro_auprc": float(scores.test_auprc.mean()),
        "physical_features": 149,
        "test_rows": info["test_rows"],
        "test_loaded_or_scored": True,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(
        f"fold={fold}: test AUROC {result['test_macro_auroc']:.6f}, "
        f"AUPRC {result['test_macro_auprc']:.6f}",
        flush=True,
    )
    return result


def aggregate() -> dict:
    validation_rows, test_rows, validation_tasks, test_tasks = [], [], [], []
    for fold in FOLDS:
        validation = json.loads((OUT / f"fold{fold}/VALIDATION_DONE").read_text())
        test = json.loads((OUT / f"fold{fold}/TEST_DONE").read_text())
        validation_rows.append(validation)
        test_rows.append(test)
        validation_tasks.append(pd.read_csv(OUT / f"fold{fold}/validation_scores.csv"))
        test_tasks.append(pd.read_csv(OUT / f"fold{fold}/test_scores.csv"))

    validation = pd.DataFrame(validation_rows)
    test = pd.DataFrame(test_rows)
    validation_long = pd.concat(validation_tasks, ignore_index=True)
    test_long = pd.concat(test_tasks, ignore_index=True)
    validation.to_csv(OUT / "validation_macro_by_fold.csv", index=False)
    test.to_csv(OUT / "test_macro_by_fold.csv", index=False)
    validation_long.to_csv(OUT / "validation_downstream_by_fold.csv", index=False)
    test_long.to_csv(OUT / "test_downstream_by_fold.csv", index=False)

    summary = {
        "protocol": "clean_v5_raw149_xgboost_fivefold_optuna25",
        "folds": 5,
        "trials_per_fold": N_TRIALS,
        "validation_macro_auroc_mean": float(
            validation.validation_macro_auroc.mean()
        ),
        "validation_macro_auroc_sd": float(
            validation.validation_macro_auroc.std(ddof=1)
        ),
        "validation_macro_auprc_mean": float(
            validation.validation_macro_auprc.mean()
        ),
        "validation_macro_auprc_sd": float(
            validation.validation_macro_auprc.std(ddof=1)
        ),
        "test_macro_auroc_mean": float(test.test_macro_auroc.mean()),
        "test_macro_auroc_sd": float(test.test_macro_auroc.std(ddof=1)),
        "test_macro_auprc_mean": float(test.test_macro_auprc.mean()),
        "test_macro_auprc_sd": float(test.test_macro_auprc.std(ddof=1)),
        "selection_used_test": False,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2))

    task_summary = (
        test_long.groupby(["target", "label"])
        .agg(
            folds=("fold", "count"),
            test_auroc_mean=("test_auroc", "mean"),
            test_auroc_sd=("test_auroc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_sd=("test_auprc", "std"),
        )
        .reset_index()
    )
    task_summary.to_csv(OUT / "test_downstream_summary.csv", index=False)

    if (SGMET_TEST / "DONE").exists():
        sgmet = pd.read_csv(SGMET_TEST / "macro_metrics_by_fold.csv")
        comparison = sgmet.merge(
            test[["fold", "test_macro_auroc", "test_macro_auprc"]],
            on="fold",
            suffixes=("_sgmet_k7", "_xgboost"),
        )
        comparison["xgboost_minus_sgmet_auroc"] = (
            comparison.test_macro_auroc_xgboost
            - comparison.test_macro_auroc_sgmet_k7
        )
        comparison["xgboost_minus_sgmet_auprc"] = (
            comparison.test_macro_auprc_xgboost
            - comparison.test_macro_auprc_sgmet_k7
        )
        comparison.to_csv(OUT / "sgmet_k7_vs_xgboost_macro_by_fold.csv", index=False)
        plot_comparison(comparison, task_summary)

    (OUT / "DONE").write_text(json.dumps(summary, indent=2))
    return summary


def plot_comparison(comparison: pd.DataFrame, xgb_tasks: pd.DataFrame) -> None:
    SHARE.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, metric, title in [
        (axes[0], "auroc", "Held-out test macro AUROC"),
        (axes[1], "auprc", "Held-out test macro AUPRC"),
    ]:
        sgmet = comparison[f"test_macro_{metric}_sgmet_k7"].to_numpy()
        xgb_values = comparison[f"test_macro_{metric}_xgboost"].to_numpy()
        for fold, left, right in zip(comparison.fold, sgmet, xgb_values):
            ax.plot([0, 1], [left, right], color="#DDE3E7", lw=1.3)
            ax.annotate(f"F{fold}", (1, right), xytext=(5, 0), textcoords="offset points")
        ax.scatter(np.zeros(5), sgmet, color="#167C91", s=45, zorder=2)
        ax.scatter(np.ones(5), xgb_values, color="#C45A1A", s=45, zorder=2)
        ax.set_xticks([0, 1], ["SGMET K=7", "Tuned XGBoost"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E7EBEE")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Clean v5 five-fold held-out test comparison",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(SHARE / "sgmet_k7_vs_xgboost_test_macro.png", dpi=240)
    plt.close(fig)

    sgmet_tasks = pd.read_csv(SGMET_TEST / "downstream_metrics_summary.csv")
    task_order = list(TARGET_LABELS)
    labels = [TARGET_LABELS[target] for target in task_order]
    fig, axes = plt.subplots(2, 1, figsize=(15, 5.8))
    for ax, metric, title, cmap, limits in [
        (axes[0], "auroc", "Test AUROC", "YlGnBu", (0.5, 0.9)),
        (axes[1], "auprc", "Test AUPRC", "YlOrBr", (0.0, 0.7)),
    ]:
        sgmet_values = (
            sgmet_tasks.set_index("target")
            .loc[task_order, f"test_{metric}_mean"]
            .to_numpy()
        )
        xgb_values = (
            xgb_tasks.set_index("target")
            .loc[task_order, f"test_{metric}_mean"]
            .to_numpy()
        )
        values = np.vstack([sgmet_values, xgb_values])
        image = ax.imshow(
            values,
            aspect="auto",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        ax.grid(False)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks([0, 1], ["SGMET K=7", "Tuned XGBoost"])
        ax.set_title(title, loc="left", fontweight="bold")
        for row in range(2):
            for column in range(len(labels)):
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
        fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
    fig.suptitle(
        "Mean held-out test performance across five folds",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(SHARE / "sgmet_k7_vs_xgboost_downstream_heatmaps.png", dpi=240)
    plt.close(fig)

    for path in [
        OUT / "SUMMARY.json",
        OUT / "test_macro_by_fold.csv",
        OUT / "test_downstream_summary.csv",
        OUT / "sgmet_k7_vs_xgboost_macro_by_fold.csv",
    ]:
        shutil.copy2(path, SHARE / path.name)
    archive = shutil.make_archive(str(SHARE), "zip", root_dir=SHARE)
    Path(f"{archive}.sha256").write_text(
        f"{sha256(Path(archive))}  {Path(archive).name}\n"
    )


def preflight() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["physical_feature_count"] != 149 or manifest["folds"] != list(FOLDS):
        raise ValueError("clean-v5 manifest mismatch")
    for fold in FOLDS:
        matrices, outcomes, info = load_fold(fold, include_test=False)
        if matrices["train"].shape[1] != 149 or matrices["val"].shape[1] != 149:
            raise ValueError(f"fold={fold}: feature count mismatch")
        if len(outcomes["train"]) != 11 or len(outcomes["val"]) != 11:
            raise ValueError(f"fold={fold}: target count mismatch")
        print(
            f"fold={fold}: train={matrices['train'].shape}, "
            f"val={matrices['val'].shape}, test not loaded",
            flush=True,
        )
    print("preflight passed: five clean-v5 folds, 149 features, 11 binary targets")


def smoke_test() -> None:
    matrices, outcomes, _ = load_fold(0, include_test=False)
    xtr, ytr = subset(matrices["train"], outcomes["train"]["arthritis"])
    xva, yva = subset(matrices["val"], outcomes["val"]["arthritis"])
    params = {
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 5.0,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
    }
    model = fit_model(params, xtr, ytr, xva, yva, n_estimators=20)
    score = roc_auc_score(yva, model.predict_proba(xva)[:, 1])
    if not np.isfinite(score) or not 0.5 < score < 1.0:
        raise RuntimeError("smoke-test metric failed")
    print(f"smoke test passed: fold=0 arthritis validation AUROC={score:.4f}")


def status() -> None:
    if (OUT / "DONE").exists():
        print((OUT / "DONE").read_text())
        return
    waiting = OUT / "WAITING.json"
    if waiting.exists():
        print(waiting.read_text())
    for fold in FOLDS:
        fold_dir = OUT / f"fold{fold}"
        if (fold_dir / "TEST_DONE").exists():
            result = json.loads((fold_dir / "TEST_DONE").read_text())
            print(
                f"fold={fold}: test done, AUROC {result['test_macro_auroc']:.4f}, "
                f"AUPRC {result['test_macro_auprc']:.4f}"
            )
        elif (fold_dir / "VALIDATION_DONE").exists():
            result = json.loads((fold_dir / "VALIDATION_DONE").read_text())
            print(
                f"fold={fold}: validation done, AUROC "
                f"{result['validation_macro_auroc']:.4f}"
            )
        elif (fold_dir / "STATUS.json").exists():
            result = json.loads((fold_dir / "STATUS.json").read_text())
            print(
                f"fold={fold}: tuning {result['complete_trials']}/{N_TRIALS}, "
                f"best {result['best_validation_macro_auroc']}"
            )
        else:
            print(f"fold={fold}: pending")


def wait_for_sgmet() -> None:
    marker = SGMET_TRAINING / "K10_COMPLETE"
    while not marker.exists():
        payload = {
            "state": "waiting_for_sgmet_k10_completion",
            "required_marker": str(marker),
            "updated_at": time.strftime("%F %T"),
        }
        (OUT / "WAITING.json").write_text(json.dumps(payload, indent=2))
        print("waiting for clean-v5 SGMET K=10 sequence to finish", flush=True)
        time.sleep(WAIT_SECONDS)
    (OUT / "WAITING.json").unlink(missing_ok=True)
    print("SGMET K=10 complete; starting XGBoost five-fold tuning", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--wait-for-sgmet", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return 0
    if args.smoke_test:
        smoke_test()
        return 0
    if args.status:
        status()
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another clean-v5 five-fold XGBoost runner is active")
    lock.write(str(os.getpid()))
    lock.flush()

    if args.wait_for_sgmet:
        wait_for_sgmet()
    preflight()
    started = time.time()
    validation_results = [tune_fold(fold) for fold in FOLDS]
    (OUT / "VALIDATION_COMPLETE").write_text(
        json.dumps(
            {
                "folds": list(FOLDS),
                "trials_per_fold": N_TRIALS,
                "test_loaded_or_scored": False,
                "results": validation_results,
            },
            indent=2,
        )
    )
    test_results = [score_test_fold(fold) for fold in FOLDS]
    summary = aggregate()
    summary["wall_minutes"] = round((time.time() - started) / 60, 1)
    summary["test_results"] = test_results
    (OUT / "DONE").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
