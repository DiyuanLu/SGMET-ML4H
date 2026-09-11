#!/usr/bin/env python3
"""Create balanced inner train/validation splits for all five existing outer folds.

The outer test folds from ``cv_splits/v4_cv_folds.npz`` are immutable. For each
outer fold, the remaining 80% is multilabel-stratified into five equal buckets;
one bucket becomes validation and four become training. This is the same
correction already used by ``173_recarve_stratified_val.py`` for fold 0.

Existing fold directories are never overwritten. A matching directory is
verified and skipped; a mismatching directory causes an error.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

V4: Path
CV: Path
SPLIT_ORDER = ("train", "val", "test")
TOKEN_KEYS = (
    "numeric_values",
    "continuous_bin_codes",
    "categorical_codes",
    "feature_context_codes",
    "missing_reason_codes",
    "missing_mask",
    "observed_mask",
)
SHARED = (
    "tokenizer_metadata.pt",
    "feature_bge_embeddings.pt",
    "category_bge_embeddings.pt",
    "feature_clusters_biolord_v4_k8_leiden.csv",
)
SEED = 2026


def iterative_stratification(
    labels: np.ndarray, proportions: list[float], seed: int
) -> np.ndarray:
    """Dependency-free first-order multilabel stratification used by script 92."""
    n, n_labels = labels.shape
    rng = np.random.default_rng(seed)
    p = np.asarray(proportions, dtype=float)
    desired = p * n
    label_totals = labels.sum(0)
    desired_pos = np.outer(p, label_totals)
    assignment = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)
    rows_by_label = [np.where(labels[:, j] == 1)[0] for j in range(n_labels)]
    remaining_pos = label_totals.astype(int).copy()

    def place(row: int) -> None:
        active_labels = np.where(labels[row] == 1)[0]
        score = desired_pos[:, active_labels].sum(1) if len(active_labels) else desired.copy()
        candidates = np.where(score >= score.max() - 1e-9)[0]
        if len(candidates) > 1:
            remaining = desired[candidates]
            candidates = candidates[remaining >= remaining.max() - 1e-9]
        bucket = int(candidates[rng.integers(len(candidates))])
        assignment[row] = bucket
        unassigned[row] = False
        desired[bucket] -= 1
        for label in active_labels:
            desired_pos[bucket, label] -= 1
            remaining_pos[label] -= 1

    while True:
        available = [j for j in range(n_labels) if remaining_pos[j] > 0]
        if not available:
            break
        label = min(available, key=lambda j: (remaining_pos[j], label_totals[j]))
        for row in rows_by_label[label]:
            if unassigned[row]:
                place(int(row))

    leftovers = np.where(unassigned)[0]
    rng.shuffle(leftovers)
    for row in leftovers:
        bucket = int(np.argmax(desired))
        assignment[row] = bucket
        desired[bucket] -= 1
    return assignment


def pool_inputs(targets: list[str]):
    pooled_tokens = {
        key: torch.cat(
            [
                torch.load(V4 / f"{split}_tokens.pt", map_location="cpu", weights_only=False)[key]
                for split in SPLIT_ORDER
            ],
            dim=0,
        )
        for key in TOKEN_KEYS
    }
    pooled_targets: dict[str, list] = {}
    for target in targets:
        values = []
        for split in SPLIT_ORDER:
            values.extend(
                torch.load(
                    V4 / f"{split}_targets.pt", map_location="cpu", weights_only=False
                )[target]
            )
        pooled_targets[target] = values
    return pooled_tokens, pooled_targets


def main() -> None:
    global V4, CV
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-root",
        type=Path,
        default=Path("data/processed/tokenized_nhanes_v4"),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    V4 = args.token_root.resolve()
    CV = V4 / "cv_splits"
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("--folds must be drawn from 0,1,2,3,4")

    metadata = torch.load(V4 / "tokenizer_metadata.pt", map_location="cpu", weights_only=False)
    targets = list(metadata["target_columns"])
    pooled_tokens, pooled_targets = pool_inputs(targets)
    n_rows = int(pooled_tokens["numeric_values"].shape[0])
    y_real = np.column_stack(
        [
            np.array(
                [np.nan if value is None else float(value) for value in pooled_targets[target]],
                dtype=float,
            )
            for target in targets
        ]
    )
    y_strat = np.nan_to_num(y_real, nan=0.0).astype(int)
    outer = np.load(CV / "v4_cv_folds.npz")
    summary = []

    for fold in args.folds:
        test_idx = np.sort(outer[f"fold{fold}_test"])
        pool_idx = np.sort(
            np.concatenate([outer[f"fold{fold}_train"], outer[f"fold{fold}_val"]])
        )
        buckets = iterative_stratification(
            y_strat[pool_idx], [0.2] * 5, seed=SEED + 100 + fold
        )
        val_idx = np.sort(pool_idx[buckets == 0])
        train_idx = np.sort(pool_idx[buckets != 0])

        train_set, val_set, test_set = map(set, (train_idx, val_idx, test_idx))
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert len(train_set | val_set | test_set) == n_rows
        assert test_set == set(outer[f"fold{fold}_test"])

        rates = {}
        for column, target in enumerate(targets):
            train_rate = float(np.nanmean(y_real[train_idx, column]))
            val_rate = float(np.nanmean(y_real[val_idx, column]))
            test_rate = float(np.nanmean(y_real[test_idx, column]))
            rates[target] = (train_rate, val_rate, test_rate)
        worst_ratio_error = max(
            abs(val_rate / train_rate - 1.0)
            for train_rate, val_rate, _ in rates.values()
            if train_rate > 0
        )
        if worst_ratio_error > 0.15:
            raise RuntimeError(
                f"fold {fold}: validation prevalence parity failed ({worst_ratio_error:.3f})"
            )

        output = CV / f"fold{fold}_restrat"
        index_path = output / "restrat_indices.npz"
        if index_path.exists():
            prior = np.load(index_path)
            if not (
                np.array_equal(prior["train"], train_idx)
                and np.array_equal(prior["val"], val_idx)
                and np.array_equal(prior["test"], test_idx)
            ):
                raise RuntimeError(f"{output} exists with different indices; refusing to overwrite")
            print(f"fold {fold}: verified existing {output}")
        else:
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"{output} is non-empty without indices; refusing to overwrite")
            output.mkdir(parents=True, exist_ok=True)
            for split, indices in (
                ("train", train_idx),
                ("val", val_idx),
                ("test", test_idx),
            ):
                tensor_indices = torch.as_tensor(indices, dtype=torch.long)
                torch.save(
                    {
                        key: pooled_tokens[key].index_select(0, tensor_indices).contiguous()
                        for key in TOKEN_KEYS
                    },
                    output / f"{split}_tokens.pt",
                )
                torch.save(
                    {
                        target: [pooled_targets[target][int(i)] for i in indices]
                        for target in targets
                    },
                    output / f"{split}_targets.pt",
                )
            for shared in SHARED:
                shutil.copy2(V4 / shared, output / shared)
            np.savez_compressed(
                index_path,
                train=train_idx.astype(np.int32),
                val=val_idx.astype(np.int32),
                test=test_idx.astype(np.int32),
            )
            print(f"fold {fold}: wrote {output}")

        for target, (train_rate, val_rate, test_rate) in rates.items():
            summary.append(
                {
                    "fold": fold,
                    "target": target.replace("label_", ""),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "train_prevalence": train_rate,
                    "val_prevalence": val_rate,
                    "test_prevalence": test_rate,
                    "val_train_ratio": val_rate / train_rate if train_rate else np.nan,
                }
            )

    summary_frame = pd.DataFrame(summary)
    summary_path = CV / "restratified_fivefold_prevalence.csv"
    summary_frame.to_csv(summary_path, index=False)
    manifest = {
        "outer_split": "cv_splits/v4_cv_folds.npz",
        "outer_test_folds_changed": False,
        "inner_method": "equal five-bucket iterative multilabel stratification; bucket 0 is validation",
        "seed_per_fold": {str(fold): SEED + 100 + fold for fold in args.folds},
        "proportions": {"train": 0.64, "validation": 0.16, "test": 0.20},
        "folds": args.folds,
        "prevalence_table": summary_path.name,
    }
    (CV / "restratified_fivefold_config.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
