#!/usr/bin/env python3
"""Score the 15 locked FT checkpoints once on their held-out test folds."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
TRAIN_SOURCE = Path(__file__).with_name("ft_transformer_v5.py")
TRAIN_ROOT = ROOT / "outputs/ft_transformer_v5_fivefold_final_seed42"
OUT = ROOT / "outputs/ft_transformer_v5_fivefold_test_seed42"
ARCHITECTURES = ("reference", "matched", "matched_h4")
FOLDS = range(5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_module():
    spec = importlib.util.spec_from_file_location("ft_transformer_v5", TRAIN_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TRAIN_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FT = load_training_module()


def evaluate(architecture: str, fold: int) -> tuple[dict, pd.DataFrame]:
    run = TRAIN_ROOT / architecture / f"fold{fold}"
    train_config = json.loads((TRAIN_ROOT / architecture / "CONFIG.json").read_text())
    done = json.loads((run / "DONE").read_text())
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if done["test_scored"] is not False:
        raise RuntimeError(f"training marker already says test scored: {run}")
    if done["stopping_reason"] != "early_stopping":
        raise RuntimeError(f"non-early-stopped run is not eligible: {run}")
    if checkpoint["epoch"] != done["selected_epoch"]:
        raise RuntimeError(f"checkpoint-selection mismatch: {run}")
    if checkpoint["architecture"] != architecture:
        raise RuntimeError(f"checkpoint architecture mismatch: {run}")
    if checkpoint["config"] != train_config:
        raise RuntimeError(f"checkpoint configuration mismatch: {run}")
    if train_config["source_sha256"] != sha256(TRAIN_SOURCE):
        raise RuntimeError("training source hash changed after the locked runs")
    if train_config["dataset_manifest_sha256"] != sha256(FT.DATA / "MANIFEST.json"):
        raise RuntimeError("dataset manifest hash mismatch")

    metadata = FT.metadata_for_fold(fold)
    target_names = list(metadata["target_columns"])
    dataset = FT.load_split(fold, "test", target_names)
    loader = DataLoader(dataset, batch_size=FT.PROTOCOL["batch_size"], shuffle=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FT.model_for(architecture, metadata).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    ys, masks, probabilities = [], [], []
    started = time.time()
    with torch.inference_mode():
        for numeric, categorical, missing, y, y_mask in loader:
            logits = model(
                numeric.to(device), categorical.to(device), missing.to(device)
            )
            ys.append(y)
            masks.append(y_mask)
            probabilities.append(torch.sigmoid(logits).cpu())
    y = torch.cat(ys).numpy()
    mask = torch.cat(masks).numpy()
    probability = torch.cat(probabilities).numpy()
    metrics = FT.metrics_from_arrays(y, mask, probability, target_names)

    rows = []
    for index, target in enumerate(target_names):
        valid = mask[:, index].astype(bool)
        short = target.removeprefix("label_")
        rows.append(
            {
                "architecture": architecture,
                "fold": fold,
                "target": short,
                "test_n": int(valid.sum()),
                "test_prevalence": float(y[valid, index].mean()),
                "test_auroc": metrics[f"{short}_auroc"],
                "test_auprc": metrics[f"{short}_auprc"],
            }
        )
    scores = pd.DataFrame(rows)
    if len(scores) != 11 or not np.isfinite(
        scores[["test_auroc", "test_auprc"]].to_numpy()
    ).all():
        raise RuntimeError(f"invalid test metrics: {architecture} fold {fold}")

    result = {
        "architecture": architecture,
        "fold": fold,
        "protocol": "locked_validation_checkpoint_scored_once_on_held_out_test",
        "selected_epoch": int(checkpoint["epoch"]),
        "test_macro_auroc": float(scores.test_auroc.mean()),
        "test_macro_auprc": float(scores.test_auprc.mean()),
        "test_patients": int(len(dataset)),
        "runtime_seconds": time.time() - started,
        "checkpoint_sha256": sha256(checkpoint_path),
        "test_tokens_sha256": sha256(
            FT.DATA / "cv_splits" / f"fold{fold}" / "test_tokens.pt"
        ),
        "test_targets_sha256": sha256(
            FT.DATA / "cv_splits" / f"fold{fold}" / "test_targets.pt"
        ),
        "training_source_sha256": train_config["source_sha256"],
        "dataset_manifest_sha256": train_config["dataset_manifest_sha256"],
        "used_for_model_selection": False,
    }
    return result, scores


def score_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for architecture in ARCHITECTURES:
        for fold in FOLDS:
            destination = OUT / architecture / f"fold{fold}"
            marker = destination / "TEST_DONE.json"
            if marker.exists():
                print(f"{architecture} fold={fold}: already scored", flush=True)
                continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"refusing incomplete test directory: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            result, scores = evaluate(architecture, fold)
            scores.to_csv(destination / "test_scores.csv", index=False)
            marker.write_text(json.dumps(result, indent=2))
            print(
                f"{architecture} fold={fold}: test AUROC "
                f"{result['test_macro_auroc']:.6f}, AUPRC "
                f"{result['test_macro_auprc']:.6f}",
                flush=True,
            )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    macro, task = [], []
    for architecture in ARCHITECTURES:
        for fold in FOLDS:
            destination = OUT / architecture / f"fold{fold}"
            macro.append(json.loads((destination / "TEST_DONE.json").read_text()))
            task.append(pd.read_csv(destination / "test_scores.csv"))
    macro = pd.DataFrame(macro)
    tasks = pd.concat(task, ignore_index=True)
    macro.to_csv(OUT / "test_macro_by_fold.csv", index=False)
    tasks.to_csv(OUT / "test_downstream_by_fold.csv", index=False)
    summary = (
        macro.groupby("architecture", sort=False)
        .agg(
            test_macro_auroc_mean=("test_macro_auroc", "mean"),
            test_macro_auroc_sd=("test_macro_auroc", "std"),
            test_macro_auprc_mean=("test_macro_auprc", "mean"),
            test_macro_auprc_sd=("test_macro_auprc", "std"),
            runtime_seconds_total=("runtime_seconds", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "test_macro_summary.csv", index=False)
    (OUT / "DONE").write_text(
        json.dumps(
            {
                "architectures": list(ARCHITECTURES),
                "folds": list(FOLDS),
                "checkpoint_policy": "locked validation winners",
                "used_for_model_selection": False,
                "test_scoring_complete": True,
                "evaluator_sha256": sha256(Path(__file__).resolve()),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False), flush=True)


def status() -> None:
    for architecture in ARCHITECTURES:
        for fold in FOLDS:
            marker = OUT / architecture / f"fold{fold}" / "TEST_DONE.json"
            if marker.exists():
                row = json.loads(marker.read_text())
                print(
                    f"{architecture} fold={fold}: done, "
                    f"AUROC={row['test_macro_auroc']:.4f}, "
                    f"AUPRC={row['test_macro_auprc']:.4f}"
                )
            else:
                print(f"{architecture} fold={fold}: pending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    status() if args.status else score_all()
