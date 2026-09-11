#!/usr/bin/env python3
"""Validate an extracted literal-149 tokenized_nhanes_v5 package."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

TOKEN_KEYS = (
    "numeric_values",
    "continuous_bin_codes",
    "categorical_codes",
    "feature_context_codes",
    "missing_reason_codes",
    "missing_mask",
    "observed_mask",
)
MAPS = {
    "feature_clusters_biolord_v5_k7_leiden.csv": 7,
    "feature_clusters_biolord_v5_k10_clinical.csv": 10,
}


def validate_map(path: Path, names: list[str], k: int) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapped = [row["feature_name"] for row in rows]
    assignments = [int(row["cluster_id"]) for row in rows]
    assert mapped == names
    assert len(mapped) == len(set(mapped)) == 149
    assert set(assignments) == set(range(k))
    assert min(assignments.count(cluster) for cluster in range(k)) >= 2


def validate_targets(path: Path, target_columns: list[str]) -> None:
    targets = torch.load(path, map_location="cpu", weights_only=False)
    assert set(target_columns).issubset(targets)
    for target in target_columns:
        observed = {int(value) for value in targets[target] if value is not None}
        assert observed == {0, 1}, f"{path}: {target} lacks both classes"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_package.py TOKENIZED_NHANES_V5")
    root = Path(sys.argv[1]).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert manifest["physical_feature_count"] == 149
    names = list(manifest["feature_order"])
    assert len(names) == len(set(names)) == 149

    for map_name, k in MAPS.items():
        validate_map(root / "cluster_maps" / map_name, names, k)

    reference_metadata = None
    for fold in range(5):
        fold_dir = root / "cv_splits" / f"fold{fold}"
        metadata = torch.load(
            fold_dir / "tokenizer_metadata.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert metadata["feature_names"] == names
        if reference_metadata is None:
            reference_metadata = metadata
        assert metadata["feature_context_cardinalities"] == reference_metadata[
            "feature_context_cardinalities"
        ]
        expected_contexts = sum(metadata["feature_context_cardinalities"])
        feature_embeddings = torch.load(
            fold_dir / "feature_bge_embeddings.pt",
            map_location="cpu",
            weights_only=False,
        )
        category_embeddings = torch.load(
            fold_dir / "category_bge_embeddings.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert feature_embeddings["feature_names"] == names
        assert feature_embeddings["embeddings"].shape[0] == expected_contexts
        assert category_embeddings["feature_names"] == names
        assert all(0 <= int(index) < 149 for index in category_embeddings["embeddings"])

        for split in ("train", "val", "test"):
            tokens = torch.load(
                fold_dir / f"{split}_tokens.pt",
                map_location="cpu",
                weights_only=False,
            )
            shapes = [tuple(tokens[key].shape) for key in TOKEN_KEYS]
            assert len(set(shapes)) == 1 and shapes[0][1] == 149
            validate_targets(
                fold_dir / f"{split}_targets.pt",
                metadata["target_columns"],
            )
        assert (fold_dir / "restrat_indices.npz").is_file()

    print("PASS: 5 folds, 149 physical features, clean K=7 and clinical K=10 maps")


if __name__ == "__main__":
    main()
