#!/usr/bin/env python3
"""Train an exact-matched FT-Transformer with SGMET-style missingness.

Training corruption is generated online for every mini-batch. Validation stays
clean for checkpoint selection, and this script never loads held-out test data.
The shipped clean FT implementation and its checkpoints are read-only controls.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import ft_transformer_v5 as clean_ft


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/tokenized_nhanes_v5"
CLUSTER_MAP = DATA / "cluster_maps/feature_clusters_biolord_v5_k7_leiden.csv"
CLEAN_OUTPUT = ROOT / "outputs/ft_transformer_v5_fivefold_final_seed42/matched"
OUTPUT = ROOT / "outputs/ft_transformer_v5_matched_missingness_uniform_groups_seed42"
ARCHITECTURE = "matched"
N_FEATURES = 149
N_TARGETS = 11
N_GROUPS = 7
FEATURE_DROPOUT_MAX = 0.10

AUGMENTATION = {
    "scope": "training batches only",
    "feature_dropout": "sample p ~ Uniform(0, 0.10) per batch, then remove each observed patient-feature cell independently with probability p",
    "group_dropout": "sample n ~ DiscreteUniform({0,...,6}) per batch, then remove n of the 7 semantic groups uniformly without replacement for every patient in the batch",
    "natural_missingness": "preserved",
    "numeric_removal": "numeric value = 0 and missing_mask = true",
    "categorical_removal": "categorical code = 0 (explicit MISSING category) and missing_mask = true",
    "targets": "unchanged",
    "validation": "clean and unaugmented",
    "selection": "clean validation macro AUROC",
}


def cluster_ids(feature_names: list[str]) -> torch.Tensor:
    frame = pd.read_csv(CLUSTER_MAP)
    if list(frame.columns) != ["feature_name", "cluster_id"]:
        raise ValueError(f"unexpected columns in {CLUSTER_MAP}")
    if frame.feature_name.duplicated().any():
        raise ValueError("cluster map contains duplicate feature names")
    mapping = frame.set_index("feature_name").cluster_id.to_dict()
    missing = [name for name in feature_names if name not in mapping]
    extra = sorted(set(mapping) - set(feature_names))
    if missing or extra:
        raise ValueError(f"cluster-map mismatch: missing={missing}, extra={extra}")
    result = torch.tensor([int(mapping[name]) for name in feature_names], dtype=torch.long)
    if set(result.tolist()) != set(range(N_GROUPS)):
        raise ValueError(f"expected group ids 0..6, found {sorted(set(result.tolist()))}")
    return result


def corruption_seed(fold: int) -> int:
    return 42_000 + fold


def corrupt_batch(
    numeric: torch.Tensor,
    categorical: torch.Tensor,
    missing: torch.Tensor,
    groups: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int | list[int]]]:
    """Apply one reproducible online corruption draw to a CPU mini-batch."""
    if numeric.device.type != "cpu" or groups.device.type != "cpu":
        raise ValueError("corruption must be sampled on CPU for reproducibility")
    if numeric.shape != categorical.shape or numeric.shape != missing.shape:
        raise ValueError("token tensors have incompatible shapes")
    if numeric.shape[1] != groups.numel():
        raise ValueError("feature and group dimensions differ")

    numeric = numeric.clone()
    categorical = categorical.clone()
    missing = missing.clone()
    naturally_observed = ~missing

    p_feature = float(torch.rand((), generator=generator).item() * FEATURE_DROPOUT_MAX)
    cell_removed = (
        torch.rand(numeric.shape, generator=generator) < p_feature
    ) & naturally_observed

    n_groups_removed = int(torch.randint(0, N_GROUPS, (), generator=generator).item())
    removed_group_ids = (
        torch.randperm(N_GROUPS, generator=generator)[:n_groups_removed]
        if n_groups_removed
        else torch.empty(0, dtype=torch.long)
    )
    removed_columns = torch.zeros(groups.shape, dtype=torch.bool)
    for group_id in removed_group_ids.tolist():
        removed_columns |= groups == group_id
    group_removed = naturally_observed & removed_columns.unsqueeze(0)
    artificially_removed = cell_removed | group_removed

    numeric[artificially_removed] = 0.0
    categorical[artificially_removed] = 0
    missing[artificially_removed] = True

    observed_count = int(naturally_observed.sum().item())
    audit: dict[str, float | int | list[int]] = {
        "feature_p": p_feature,
        "groups_removed": n_groups_removed,
        "removed_group_ids": removed_group_ids.tolist(),
        "observed_cells": observed_count,
        "feature_removed_cells": int(cell_removed.sum().item()),
        "combined_removed_cells": int(artificially_removed.sum().item()),
        "combined_removed_fraction_of_observed": (
            float(artificially_removed.sum().item() / observed_count)
            if observed_count
            else 0.0
        ),
    }
    return numeric, categorical, missing, audit


def run_augmented_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_names: list[str],
    optimizer: torch.optim.Optimizer,
    groups: torch.Tensor,
    corruption_generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    losses, ys, masks, probabilities = [], [], [], []
    feature_ps, removed_counts, removed_fractions = [], [], []
    group_histogram = np.zeros(N_GROUPS, dtype=np.int64)

    for numeric, categorical, missing, y, y_mask in loader:
        numeric, categorical, missing, audit = corrupt_batch(
            numeric, categorical, missing, groups, corruption_generator
        )
        feature_ps.append(float(audit["feature_p"]))
        removed_counts.append(int(audit["groups_removed"]))
        removed_fractions.append(float(audit["combined_removed_fraction_of_observed"]))
        group_histogram[int(audit["groups_removed"])] += 1

        numeric = numeric.to(device)
        categorical = categorical.to(device)
        missing = missing.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(numeric, categorical, missing)
        loss = clean_ft.focal_loss(logits, y, y_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        ys.append(y.detach().cpu())
        masks.append(y_mask.detach().cpu())
        probabilities.append(torch.sigmoid(logits.detach()).cpu())

    result = clean_ft.metrics_from_arrays(
        torch.cat(ys).numpy(),
        torch.cat(masks).numpy(),
        torch.cat(probabilities).numpy(),
        target_names,
    )
    result.update(
        {
            "loss": float(np.mean(losses)),
            "sampled_feature_p_mean": float(np.mean(feature_ps)),
            "sampled_groups_removed_mean": float(np.mean(removed_counts)),
            "removed_observed_fraction_mean": float(np.mean(removed_fractions)),
            **{
                f"batches_with_{n}_groups_removed": int(group_histogram[n])
                for n in range(N_GROUPS)
            },
        }
    )
    return result


def experiment_config() -> dict:
    metadata = clean_ft.metadata_for_fold(0)
    model = clean_ft.model_for(ARCHITECTURE, metadata)
    clean_controls = {}
    for fold in range(5):
        done = CLEAN_OUTPUT / f"fold{fold}/DONE"
        best = CLEAN_OUTPUT / f"fold{fold}/best.pt"
        if not done.exists() or not best.exists():
            raise FileNotFoundError(f"missing clean FT control for fold {fold}")
        clean_controls[str(fold)] = {
            "done": str(done),
            "done_sha256": clean_ft.sha256(done),
            "checkpoint": str(best),
            "checkpoint_sha256": clean_ft.sha256(best),
        }
    return {
        "experiment": "FT-Transformer matched missingness augmentation",
        "dataset": "tokenized_nhanes_v5",
        "physical_feature_count": N_FEATURES,
        "target_count": N_TARGETS,
        "architecture_name": ARCHITECTURE,
        "architecture": asdict(clean_ft.ARCHITECTURES[ARCHITECTURE]),
        "trainable_parameters": clean_ft.parameter_count(model),
        "base_training_protocol": clean_ft.PROTOCOL,
        "augmentation": AUGMENTATION,
        "semantic_group_count": N_GROUPS,
        "cluster_map": str(CLUSTER_MAP),
        "cluster_map_sha256": clean_ft.sha256(CLUSTER_MAP),
        "dataset_manifest_sha256": clean_ft.sha256(DATA / "MANIFEST.json"),
        "base_source_sha256": clean_ft.sha256(Path(clean_ft.__file__).resolve()),
        "source_sha256": clean_ft.sha256(Path(__file__).resolve()),
        "clean_controls": clean_controls,
        "test_policy": "test tensors are not loaded by this script",
    }


def write_or_validate_config() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "CONFIG.json"
    expected = experiment_config()
    if path.exists() and json.loads(path.read_text()) != expected:
        raise RuntimeError(f"configuration guard failed for {path}")
    if not path.exists():
        path.write_text(json.dumps(expected, indent=2))


def train_fold(fold: int) -> None:
    write_or_validate_config()
    destination = OUTPUT / f"fold{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "DONE").exists():
        print(f"fold {fold} already complete")
        return
    if any(destination.iterdir()):
        raise RuntimeError(f"refusing to overwrite incomplete run directory: {destination}")

    clean_ft.seed_all(clean_ft.PROTOCOL["seed"])
    metadata = clean_ft.metadata_for_fold(fold)
    target_names = list(metadata["target_columns"])
    groups = cluster_ids(list(metadata["feature_names"]))
    train_data = clean_ft.load_split(fold, "train", target_names)
    val_data = clean_ft.load_split(fold, "val", target_names)
    shuffle_generator = torch.Generator().manual_seed(clean_ft.PROTOCOL["seed"])
    corruption_generator = torch.Generator().manual_seed(corruption_seed(fold))
    train_loader = DataLoader(
        train_data,
        batch_size=clean_ft.PROTOCOL["batch_size"],
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_data, batch_size=clean_ft.PROTOCOL["batch_size"], shuffle=False
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = clean_ft.model_for(ARCHITECTURE, metadata).to(device)
    optimizer = clean_ft.optimizer_for(model)

    metrics_path = destination / "metrics.csv"
    best_auroc, best_record, bad_epochs = -math.inf, None, 0
    started = time.time()
    for epoch in range(clean_ft.PROTOCOL["epochs"]):
        train_metrics = run_augmented_train_epoch(
            model,
            train_loader,
            device,
            target_names,
            optimizer,
            groups,
            corruption_generator,
        )
        with torch.no_grad():
            val_metrics = clean_ft.run_epoch(model, val_loader, device, target_names, None)

        row: dict[str, float | int] = {
            "epoch": epoch,
            "elapsed_minutes": (time.time() - started) / 60,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        with metrics_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

        improved = np.isfinite(val_metrics["macro_auroc"]) and (
            val_metrics["macro_auroc"] > best_auroc + 1e-6
        )
        if improved:
            best_auroc = val_metrics["macro_auroc"]
            best_record = row
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "architecture": ARCHITECTURE,
                    "model_state_dict": model.state_dict(),
                    "target_names": target_names,
                    "best_record": best_record,
                    "config": experiment_config(),
                },
                destination / "best.pt",
            )
        else:
            bad_epochs += 1
        torch.save(
            {
                "epoch": epoch,
                "architecture": ARCHITECTURE,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "corruption_generator_state": corruption_generator.get_state(),
                "bad_epochs": bad_epochs,
                "best_record": best_record,
            },
            destination / "last.pt",
        )
        print(
            f"matched+missingness fold={fold} epoch={epoch} "
            f"train={train_metrics['macro_auroc']:.4f} "
            f"val={val_metrics['macro_auroc']:.4f} "
            f"val_pr={val_metrics['macro_auprc']:.4f} "
            f"groups={train_metrics['sampled_groups_removed_mean']:.2f}",
            flush=True,
        )
        if bad_epochs >= clean_ft.PROTOCOL["patience"]:
            break

    if best_record is None:
        raise RuntimeError("no finite validation checkpoint was produced")
    done = {
        "architecture": ARCHITECTURE,
        "fold": fold,
        "selected_epoch": int(best_record["epoch"]),
        "epochs_run": epoch + 1,
        "selected_val_macro_auroc": best_record["val_macro_auroc"],
        "selected_val_macro_auprc": best_record["val_macro_auprc"],
        "runtime_minutes": (time.time() - started) / 60,
        "stopping_reason": (
            "early_stopping"
            if bad_epochs >= clean_ft.PROTOCOL["patience"]
            else "epoch_cap"
        ),
        "test_scored": False,
        "trainable_parameters": clean_ft.parameter_count(model),
        "clean_validation_selection": True,
    }
    (destination / "DONE").write_text(json.dumps(done, indent=2))


def preflight() -> None:
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    if manifest["physical_feature_count"] != N_FEATURES:
        raise ValueError("dataset feature-count contract changed")
    reference_names = None
    for fold in range(5):
        metadata = clean_ft.metadata_for_fold(fold)
        names = list(metadata["feature_names"])
        if names != manifest["feature_order"] or len(names) != N_FEATURES:
            raise ValueError(f"fold {fold}: feature contract mismatch")
        if len(metadata["target_columns"]) != N_TARGETS:
            raise ValueError(f"fold {fold}: target contract mismatch")
        if reference_names is None:
            reference_names = names
        if names != reference_names:
            raise ValueError(f"fold {fold}: feature order changed")
        categorical_names = [
            names[index]
            for index in torch.where(metadata["feature_type_ids"].long() == 1)[0].tolist()
        ]
        value_texts = metadata["category_value_texts"]
        for name in categorical_names:
            descriptions = value_texts[name]["0"]
            if not descriptions or "Answer: MISSING." not in descriptions[0]:
                raise ValueError(f"fold {fold}: category code 0 is not MISSING for {name}")
        cluster_ids(names)
        for split in ("train", "val"):
            dataset = clean_ft.load_split(fold, split, list(metadata["target_columns"]))
            numeric, categorical, missing, y, y_mask = dataset.tensors
            if numeric.shape[1] != N_FEATURES or len(dataset) == 0:
                raise ValueError(f"fold {fold} {split}: token shape mismatch")
            if numeric.shape != categorical.shape or numeric.shape != missing.shape:
                raise ValueError(f"fold {fold} {split}: channel shape mismatch")
            for target in range(N_TARGETS):
                observed = y[y_mask[:, target], target]
                if set(observed.unique().tolist()) != {0.0, 1.0}:
                    raise ValueError(f"fold {fold} {split}: target {target} lacks both classes")
        if not (CLEAN_OUTPUT / f"fold{fold}/DONE").exists():
            raise FileNotFoundError(f"fold {fold}: clean control is missing")

    metadata = clean_ft.metadata_for_fold(0)
    model = clean_ft.model_for(ARCHITECTURE, metadata)
    count = clean_ft.parameter_count(model)
    if count != 1_547_093:
        raise ValueError(f"unexpected matched parameter count: {count}")
    print(
        "PREFLIGHT PASS: 5 folds, 149 features, 11 targets, exact matched architecture, "
        "K=7 map, clean controls present, test tensors not loaded"
    )


def smoke_test() -> None:
    preflight()
    metadata = clean_ft.metadata_for_fold(0)
    target_names = list(metadata["target_columns"])
    groups = cluster_ids(list(metadata["feature_names"]))
    batch = [tensor[:32] for tensor in clean_ft.load_split(0, "train", target_names).tensors]
    numeric, categorical, missing, y, y_mask = batch

    first = corrupt_batch(
        numeric, categorical, missing, groups, torch.Generator().manual_seed(123)
    )
    repeated = corrupt_batch(
        numeric, categorical, missing, groups, torch.Generator().manual_seed(123)
    )
    changed = corrupt_batch(
        numeric, categorical, missing, groups, torch.Generator().manual_seed(124)
    )
    for left, right in zip(first[:3], repeated[:3]):
        if not torch.equal(left, right):
            raise AssertionError("same corruption seed was not deterministic")
    if all(torch.equal(left, right) for left, right in zip(first[:3], changed[:3])):
        raise AssertionError("different corruption seeds produced identical batches")

    aug_numeric, aug_categorical, aug_missing, _ = first
    artificial = aug_missing & ~missing
    if not torch.equal(aug_numeric[missing], numeric[missing]):
        raise AssertionError("naturally missing numeric payloads changed")
    if not torch.equal(aug_categorical[missing], categorical[missing]):
        raise AssertionError("naturally missing categorical payloads changed")
    if not torch.equal(aug_missing[missing], missing[missing]):
        raise AssertionError("natural missingness was not preserved")
    if artificial.any():
        if not torch.all(aug_numeric[artificial] == 0):
            raise AssertionError("removed numeric values were not neutralized")
        if not torch.all(aug_categorical[artificial] == 0):
            raise AssertionError("removed categorical codes were not neutralized")

    histogram = np.zeros(N_GROUPS, dtype=np.int64)
    generator = torch.Generator().manual_seed(999)
    for _ in range(1000):
        *_, audit = corrupt_batch(numeric, categorical, missing, groups, generator)
        histogram[int(audit["groups_removed"])] += 1
    if np.any(histogram == 0):
        raise AssertionError(f"group-count support incomplete: {histogram.tolist()}")

    model = clean_ft.model_for(ARCHITECTURE, metadata)
    optimizer = clean_ft.optimizer_for(model)
    optimizer.zero_grad(set_to_none=True)
    logits = model(aug_numeric, aug_categorical, aug_missing)
    loss = clean_ft.focal_loss(logits, y, y_mask)
    loss.backward()
    if not torch.isfinite(loss) or not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise AssertionError("non-finite smoke-test loss or gradient")
    print(
        "SMOKE PASS: deterministic online corruption, all n=0..6 group-count states sampled, "
        "removed values neutralized, natural missingness preserved, finite backward pass"
    )


def status() -> None:
    for fold in range(5):
        destination = OUTPUT / f"fold{fold}"
        if (destination / "DONE").exists():
            record = json.loads((destination / "DONE").read_text())
            print(
                f"fold={fold}: done, val AUROC={record['selected_val_macro_auroc']:.4f}, "
                f"val AUPRC={record['selected_val_macro_auprc']:.4f}, "
                f"epoch={record['selected_epoch']}, runtime={record['runtime_minutes']:.1f}m"
            )
        elif (destination / "metrics.csv").exists():
            with (destination / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            last = rows[-1]
            print(
                f"fold={fold}: active/incomplete, epoch={last['epoch']}, "
                f"val AUROC={float(last['val_macro_auroc']):.4f}"
            )
        else:
            print(f"fold={fold}: pending")


def with_run_lock(action) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT / "RUNNING.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another missingness-augmented FT run is active") from error
        lock.write(str(os.getpid()))
        lock.flush()
        action()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--fivefold", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.smoke_test:
        smoke_test()
    elif args.status:
        status()
    elif args.fold is not None:
        with_run_lock(lambda: train_fold(args.fold))
    elif args.fivefold:
        with_run_lock(lambda: [train_fold(fold) for fold in range(5)])
    else:
        parser.error("use --preflight, --smoke-test, --status, --fold N, or --fivefold")


if __name__ == "__main__":
    main()
