from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .metadata import DEFAULT_CATEGORY_LABEL_MAPS_PATH, DEFAULT_METADATA_PATH, apply_value_labels, load_feature_metadata
from .preprocessing import NameValuePreprocessor, transformed_batch_to_dict
from .tokenization_artifacts import (
    write_tokenization_check_parquet,
    write_tokenization_plan_parquet,
)
from .columns import (
    IDENTIFIER_COLUMNS,
    infer_columns_from_value_labels,
    load_registry_drop_columns,
    load_value_label_specs,
    load_registry_target_columns,
    load_missing_reason_codes,
    model_feature_columns,
    sanitize_range_feature_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize stacked NHANES rows for the feature-token transformer.")
    parser.add_argument(
        "--stacked-parquet",
        type=Path,
        default=Path("data/processed/nhanes_2011_2023.parquet"),
        help="Stacked NHANES parquet with split, cycle, feature, and feature__missing_reason columns.",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--category-label-maps", type=Path, default=DEFAULT_CATEGORY_LABEL_MAPS_PATH)
    parser.add_argument(
        "--task-registry",
        type=Path,
        default=Path("src/downstream_tasks/task_registry.json"),
        help="Task registry whose target_column entries define which columns are treated as labels.",
    )
    parser.add_argument(
        "--missing-reason-codes",
        type=Path,
        default=Path("data/processed/nhanes_2011_2023_missing_reason_codes.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/name_value_tokens/nhanes_2011_2023"))
    parser.add_argument("--continuous-quantile-bins", type=int, default=10)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.stacked_parquet)
    if "split" not in df.columns:
        raise ValueError(f"Stacked parquet must contain a split column: {args.stacked_parquet}")

    split_frames = {}
    for split in ["train", "val", "test"]:
        frame = df[df["split"].astype(str).eq(split)].copy()
        if args.max_rows_per_split is not None:
            frame = frame.head(args.max_rows_per_split).copy()
        split_frames[split] = frame
    if split_frames["train"].empty:
        raise ValueError("Training split is empty; cannot fit tokenizer.")

    registry_target_columns = load_registry_target_columns(args.task_registry)
    registry_drop_columns = load_registry_drop_columns(args.task_registry)
    target_columns = [column for column in registry_target_columns if column in df.columns]
    missing_target_columns = [column for column in registry_target_columns if column not in df.columns]
    if missing_target_columns:
        print(
            "[WARN] registry target columns not found in input parquet and will not be saved: "
            + ", ".join(missing_target_columns)
        )
    model_columns = model_feature_columns(
        split_frames["train"],
        split_frames["val"],
        split_frames["test"],
        target_columns=target_columns,
        drop_columns=registry_drop_columns,
    )
    if not model_columns:
        raise ValueError("No model feature columns found after excluding identifiers and missing-reason columns.")

    missing_reason_codes = load_missing_reason_codes(args.missing_reason_codes)
    value_label_specs = load_value_label_specs(args.category_label_maps)
    model_column_set = set(model_columns)
    model_value_label_specs = {
        column: spec for column, spec in value_label_specs.items() if column in model_column_set
    }
    sanitize_range_feature_values(
        split_frames.values(),
        value_label_specs=model_value_label_specs,
        missing_reason_codes=missing_reason_codes,
    )

    metadata = load_feature_metadata(args.metadata, model_columns, args.category_label_maps)
    split_frames["train"], split_frames["val"], split_frames["test"] = apply_value_labels(
        [split_frames["train"], split_frames["val"], split_frames["test"]],
        metadata.value_labels,
        metadata.cycle_value_labels,
    )

    numerical_columns, categorical_columns, classification_report = infer_columns_from_value_labels(
        split_frames["train"][model_columns],
        value_label_specs=model_value_label_specs,
    )
    if classification_report["missing_label_map_columns"]:
        missing = classification_report["missing_label_map_columns"]
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(
            "Category label-map coverage is required for NHANES tokenization. "
            f"Missing label-map entries for feature columns: {preview}{suffix}. "
            "Fix nhanes_category_label_maps.json before tokenizing."
        )

    preprocessor = NameValuePreprocessor.infer_from_dataframe(
        split_frames["train"][model_columns],
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        continuous_quantile_bins=args.continuous_quantile_bins,
        missing_reason_codes=missing_reason_codes,
        descriptions=metadata.descriptions,
        feature_context_texts_by_feature={
            feature_name: metadata.feature_context_texts[feature_name]
            for feature_name in model_columns
            if feature_name in metadata.feature_context_texts
        },
        units=metadata.units,
    )
    feature_context_frames = {
        split: build_feature_context_code_frame(frame, model_columns, metadata.feature_context_cycle_maps)
        for split, frame in split_frames.items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.fit(split_frames["train"], feature_context_frames["train"])
    write_tokenization_plan_parquet(
        args.out_dir / "tokenization_plan.parquet",
        preprocessor=preprocessor,
        metadata=metadata,
        model_columns=model_columns,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        classification_report=classification_report,
        target_columns=target_columns,
        split_frames=split_frames,
    )
    batches = {
        "train": preprocessor.transform(split_frames["train"], feature_context_frames["train"]),
        "val": preprocessor.transform(split_frames["val"], feature_context_frames["val"]),
        "test": preprocessor.transform(split_frames["test"], feature_context_frames["test"]),
    }
    write_tokenization_check_parquet(
        args.out_dir / "nhanes_2011_2023_tokenization_check.parquet",
        preprocessor=preprocessor,
        batches=batches,
        split_frames=split_frames,
        model_columns=model_columns,
        numerical_columns=numerical_columns,
    )
    for split, batch in batches.items():
        torch.save(transformed_batch_to_dict(batch), args.out_dir / f"{split}_tokens.pt")
        if target_columns:
            torch.save(
                {
                    column: series_to_serializable_list(split_frames[split][column])
                    for column in target_columns
                    if column in split_frames[split].columns
                },
                args.out_dir / f"{split}_targets.pt",
            )
    feature_texts = preprocessor.feature_texts()
    feature_texts_by_name = {
        feature_name: {
            "contexts": [
                {"context_code": context_code, "feature_semantic": feature_text}
                for context_code, feature_text in enumerate(preprocessor.feature_context_texts_by_feature[feature_name])
            ]
        }
        for feature_name in preprocessor.feature_names
    }
    category_value_texts = preprocessor.categorical_value_texts()
    (args.out_dir / "feature_semantics.json").write_text(
        json.dumps(feature_texts_by_name, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "category_value_texts.json").write_text(
        json.dumps(category_value_texts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    torch.save(
        {
            "feature_names": preprocessor.feature_names,
            "feature_texts": feature_texts,
            "feature_context_texts": feature_texts,
            "feature_context_texts_by_feature": preprocessor.feature_context_texts_by_feature,
            "feature_context_cardinalities": preprocessor.feature_context_cardinalities(),
            "feature_context_offsets": preprocessor.feature_context_offsets(),
            "feature_context_cycle_maps": {
                feature_name: metadata.feature_context_cycle_maps.get(feature_name, {})
                for feature_name in preprocessor.feature_names
            },
            "category_value_texts": category_value_texts,
            "feature_type_ids": preprocessor.feature_type_ids(),
            "categorical_cardinalities": preprocessor.categorical_cardinalities(),
            "continuous_bin_cardinality": preprocessor.continuous_bin_cardinality(),
            "missing_reason_cardinality": preprocessor.missing_reason_cardinality(),
            "identifier_columns": sorted(IDENTIFIER_COLUMNS),
            "target_columns": target_columns,
            "feature_classification": classification_report,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        args.out_dir / "tokenizer_metadata.pt",
    )
    print(f"[OK] saved NHANES token tensors and text artifacts to {args.out_dir}")


def build_feature_context_code_frame(
    frame: pd.DataFrame,
    feature_names: list[str],
    feature_context_cycle_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    if "cycle" not in frame.columns:
        raise ValueError("Input frame must contain a cycle column for feature context coding.")
    cycle_values = frame["cycle"].astype(str)
    context_codes: dict[str, list[int]] = {}
    for feature_name in feature_names:
        cycle_map = feature_context_cycle_maps.get(feature_name)
        if not cycle_map:
            raise ValueError(f"Missing feature context metadata for {feature_name}.")
        missing_cycles = sorted(set(cycle_values) - set(cycle_map))
        if missing_cycles:
            invalid_cycles = [
                cycle
                for cycle in missing_cycles
                if frame.loc[cycle_values.eq(cycle), feature_name].notna().any()
            ]
        else:
            invalid_cycles = []
        if invalid_cycles:
            preview = ", ".join(invalid_cycles[:8])
            suffix = "" if len(invalid_cycles) <= 8 else f", ... ({len(invalid_cycles)} total)"
            raise ValueError(f"Missing feature context metadata for {feature_name} cycles: {preview}{suffix}.")
        context_codes[feature_name] = [int(cycle_map[cycle]) if cycle in cycle_map else 0 for cycle in cycle_values]
    return pd.DataFrame(context_codes, index=frame.index)


def series_to_serializable_list(series: pd.Series) -> list[object]:
    values = []
    for value in series.tolist():
        if pd.isna(value):
            values.append(None)
        elif hasattr(value, "item"):
            values.append(value.item())
        else:
            values.append(value)
    return values


if __name__ == "__main__":
    main()
