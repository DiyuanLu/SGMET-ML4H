#!/usr/bin/env python3
"""Build the literal 149-feature tokenized_nhanes_v5 archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
TOKEN_KEYS = (
    "numeric_values",
    "continuous_bin_codes",
    "categorical_codes",
    "feature_context_codes",
    "missing_reason_codes",
    "missing_mask",
    "observed_mask",
)
SPLITS = ("train", "val", "test")
MAPS = {
    "feature_clusters_biolord_v5_k7_leiden.csv": (
        "feature_clusters_act149_t7_k7raw.csv",
        7,
    ),
    "feature_clusters_biolord_v5_k10_clinical.csv": (
        "feature_clusters_act149_clinical_k10_PROPOSED.csv",
        10,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_names(metadata: dict) -> list[str]:
    base = json.loads((HERE / "config/active_set.json").read_text())
    spec = json.loads((HERE / "config/active_149.json").read_text())
    active = set(base["active_features"]) - set(spec["exclude_additional"])
    names = [name for name in metadata["feature_names"] if name in active]
    assert len(names) == len(set(names)) == spec["n_active"] == 149
    return names


def slice_metadata(source: dict, names: list[str]) -> tuple[dict, list[int], list[int]]:
    old_names = list(source["feature_names"])
    indices = [old_names.index(name) for name in names]
    old_offsets = torch.as_tensor(source["feature_context_offsets"]).long()
    old_cards = [int(value) for value in source["feature_context_cardinalities"]]
    context_indices = [
        row
        for index in indices
        for row in range(int(old_offsets[index]), int(old_offsets[index]) + old_cards[index])
    ]
    cards = [old_cards[index] for index in indices]
    offsets = torch.tensor(
        [sum(cards[:index]) for index in range(len(cards))], dtype=torch.long
    )
    keep = set(names)
    metadata = {
        **source,
        "feature_names": names,
        "feature_texts": [source["feature_texts"][index] for index in context_indices],
        "feature_context_texts": [
            source["feature_context_texts"][index] for index in context_indices
        ],
        "feature_context_texts_by_feature": {
            name: source["feature_context_texts_by_feature"][name] for name in names
        },
        "feature_context_cardinalities": cards,
        "feature_context_offsets": offsets,
        "feature_context_cycle_maps": {
            name: source["feature_context_cycle_maps"][name] for name in names
        },
        "category_value_texts": {
            name: source["category_value_texts"][name]
            for name in names
            if name in source["category_value_texts"]
        },
        "feature_type_ids": torch.as_tensor(source["feature_type_ids"]).index_select(
            0, torch.tensor(indices)
        ),
        "categorical_cardinalities": [
            source["categorical_cardinalities"][index] for index in indices
        ],
        "feature_classification": {
            key: [name for name in values if name in keep]
            for key, values in source["feature_classification"].items()
        },
        "v5_feature_contract": {
            "physical_feature_count": 149,
            "source_feature_count": len(old_names),
            "feature_axis_is_physically_sliced": True,
        },
    }
    return metadata, indices, context_indices


def slice_embeddings(
    source_dir: Path,
    metadata: dict,
    names: list[str],
    indices: list[int],
    context_indices: list[int],
) -> tuple[dict, dict]:
    feature = torch.load(
        source_dir / "feature_bge_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    feature = {
        **feature,
        "feature_names": names,
        "feature_texts": metadata["feature_texts"],
        "feature_context_texts": metadata["feature_context_texts"],
        "feature_context_offsets": metadata["feature_context_offsets"],
        "embeddings": feature["embeddings"]
        .index_select(0, torch.tensor(context_indices))
        .contiguous(),
    }
    category = torch.load(
        source_dir / "category_bge_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    old_to_new = {old: new for new, old in enumerate(indices)}
    category = {
        **category,
        "feature_names": names,
        "category_value_texts": {
            name: category["category_value_texts"].get(name, []) for name in names
        },
        "embeddings": {
            old_to_new[int(old_index)]: tensor
            for old_index, tensor in category["embeddings"].items()
            if int(old_index) in old_to_new
        },
    }
    return feature, category


def write_clean_maps(destination: Path, names: list[str]) -> None:
    destination.mkdir()
    for output_name, (source_name, k) in MAPS.items():
        source = pd.read_csv(HERE / "cluster_maps" / source_name).set_index(
            "feature_name"
        )
        clean = pd.DataFrame(
            {
                "feature_name": names,
                "cluster_id": [int(source.at[name, "cluster_id"]) for name in names],
            }
        )
        assert set(clean.cluster_id) == set(range(k))
        assert int(clean.groupby("cluster_id").size().min()) >= 2
        clean.to_csv(destination / output_name, index=False)


def build(cv_root: Path, destination: Path) -> None:
    source0 = cv_root / "fold0_restrat"
    source_metadata = torch.load(
        source0 / "tokenizer_metadata.pt", map_location="cpu", weights_only=False
    )
    names = active_names(source_metadata)
    metadata, indices, context_indices = slice_metadata(source_metadata, names)
    feature_embeddings, category_embeddings = slice_embeddings(
        source0, metadata, names, indices, context_indices
    )
    index_tensor = torch.tensor(indices)

    (destination / "cv_splits").mkdir(parents=True)
    write_clean_maps(destination / "cluster_maps", names)
    for fold in range(5):
        source = cv_root / f"fold{fold}_restrat"
        target = destination / "cv_splits" / f"fold{fold}"
        target.mkdir()
        for split in SPLITS:
            tokens = torch.load(
                source / f"{split}_tokens.pt",
                map_location="cpu",
                weights_only=False,
            )
            torch.save(
                {
                    key: tokens[key].index_select(1, index_tensor).contiguous()
                    for key in TOKEN_KEYS
                },
                target / f"{split}_tokens.pt",
            )
            shutil.copy2(source / f"{split}_targets.pt", target)
        torch.save(metadata, target / "tokenizer_metadata.pt")
        torch.save(feature_embeddings, target / "feature_bge_embeddings.pt")
        torch.save(category_embeddings, target / "category_bge_embeddings.pt")
        shutil.copy2(source / "restrat_indices.npz", target)

    config = destination / "config"
    config.mkdir()
    for name in (
        "v4_cv_folds.npz",
        "restratified_fivefold_config.json",
        "restratified_fivefold_prevalence.csv",
    ):
        shutil.copy2(cv_root / name, config)
    for path in sorted((HERE / "config").iterdir()):
        shutil.copy2(path, config)
    shutil.copy2(HERE / "README.md", destination)
    removed = [
        name for name in source_metadata["feature_names"] if name not in set(names)
    ]
    (destination / "MANIFEST.json").write_text(
        json.dumps(
            {
                "dataset": "tokenized_nhanes_v5",
                "physical_feature_count": 149,
                "source_feature_count": len(source_metadata["feature_names"]),
                "removed_features": removed,
                "outer_test_folds_changed": False,
                "cluster_maps": list(MAPS),
                "feature_order": names,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenized_nhanes_v5_literal149.tar.gz"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    with tempfile.TemporaryDirectory() as temp:
        dataset = Path(temp) / "tokenized_nhanes_v5"
        build(args.cv_root.resolve(), dataset)
        with tarfile.open(args.output, "w:gz", compresslevel=6) as archive:
            archive.add(dataset, arcname=dataset.name)

    checksum = args.output.with_suffix(args.output.suffix + ".sha256")
    checksum.write_text(f"{sha256(args.output)}  {args.output.name}\n")
    print(args.output)
    print(checksum)


if __name__ == "__main__":
    main()
