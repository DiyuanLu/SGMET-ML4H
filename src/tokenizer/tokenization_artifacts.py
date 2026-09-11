from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import NameValuePreprocessor


def build_tokenization_plan(
    *,
    preprocessor: NameValuePreprocessor,
    metadata,
    model_columns: list[str],
    numerical_columns: list[str],
    categorical_columns: list[str],
    classification_report: dict[str, object],
    target_columns: list[str],
    split_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    numerical_set = set(numerical_columns)
    categorical_set = set(categorical_columns)
    label_map_numerical = set(classification_report.get("label_map_numerical_columns", []))
    label_map_categorical = set(classification_report.get("label_map_categorical_columns", []))
    target_column_set = set(target_columns)
    rows: list[dict[str, object]] = []
    cardinalities = preprocessor.categorical_cardinalities()
    feature_types = preprocessor.feature_type_ids().tolist()
    category_value_texts = preprocessor.categorical_value_texts()
    for feature_idx, feature_name in enumerate(model_columns):
        contexts = preprocessor.feature_context_texts_by_feature.get(feature_name, [])
        cycle_map = metadata.feature_context_cycle_maps.get(feature_name, {})
        cycles_by_context: dict[int, list[str]] = {}
        for cycle, context_code in cycle_map.items():
            cycles_by_context.setdefault(int(context_code), []).append(str(cycle))
        category_texts = category_value_texts.get(feature_name, {})
        split_non_missing = {
            split: int(frame[feature_name].notna().sum()) if feature_name in frame.columns else 0
            for split, frame in split_frames.items()
        }
        for context_code, feature_text in enumerate(contexts):
            context_category_texts = category_texts.get(str(context_code), []) if isinstance(category_texts, dict) else []
            rows.append(
                {
                    "feature_idx": feature_idx,
                    "feature_name": feature_name,
                    "feature_type_id": int(feature_types[feature_idx]),
                    "feature_type": "numerical" if feature_name in numerical_set else "categorical",
                    "context_code": context_code,
                    "n_contexts_for_feature": len(contexts),
                    "cycles_for_context": json.dumps(sorted(cycles_by_context.get(context_code, []))),
                    "feature_context_text": feature_text,
                    "is_numerical_from_label_map": feature_name in label_map_numerical,
                    "is_categorical_from_label_map": feature_name in label_map_categorical,
                    "has_missing_reason_column": any(
                        f"{feature_name}__missing_reason" in frame.columns for frame in split_frames.values()
                    ),
                    "is_target_column": feature_name in target_column_set,
                    "categorical_cardinality": int(cardinalities[feature_idx]) if feature_name in categorical_set else 0,
                    "numeric_mean": preprocessor.numeric_mean(feature_name),
                    "numeric_std": preprocessor.numeric_std(feature_name),
                    "quantile_edges": json.dumps(preprocessor.quantile_edges(feature_name)),
                    "context_category_texts": json.dumps(context_category_texts),
                    "n_context_category_texts": len(context_category_texts),
                    "train_non_missing_rows": split_non_missing.get("train", 0),
                    "val_non_missing_rows": split_non_missing.get("val", 0),
                    "test_non_missing_rows": split_non_missing.get("test", 0),
                }
            )
    return pd.DataFrame(rows)


def write_tokenization_plan_parquet(
    path: Path,
    *,
    preprocessor: NameValuePreprocessor,
    metadata,
    model_columns: list[str],
    numerical_columns: list[str],
    categorical_columns: list[str],
    classification_report: dict[str, object],
    target_columns: list[str],
    split_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    tokenization_plan = build_tokenization_plan(
        preprocessor=preprocessor,
        metadata=metadata,
        model_columns=model_columns,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        classification_report=classification_report,
        target_columns=target_columns,
        split_frames=split_frames,
    )
    tokenization_plan.to_parquet(path, index=False)
    return tokenization_plan


def build_tokenization_check(
    *,
    preprocessor: NameValuePreprocessor,
    batches: dict[str, object],
    split_frames: dict[str, pd.DataFrame],
    model_columns: list[str],
    numerical_columns: list[str],
) -> pd.DataFrame:
    frames = list(iter_tokenization_check_frames(
        preprocessor=preprocessor,
        batches=batches,
        split_frames=split_frames,
        model_columns=model_columns,
        numerical_columns=numerical_columns,
    ))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_tokenization_check_parquet(
    path: Path,
    *,
    preprocessor: NameValuePreprocessor,
    batches: dict[str, object],
    split_frames: dict[str, pd.DataFrame],
    model_columns: list[str],
    numerical_columns: list[str],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        build_tokenization_check(
            preprocessor=preprocessor,
            batches=batches,
            split_frames=split_frames,
            model_columns=model_columns,
            numerical_columns=numerical_columns,
        ).to_parquet(path, index=False)
        return

    writer = None
    try:
        for frame in iter_tokenization_check_frames(
            preprocessor=preprocessor,
            batches=batches,
            split_frames=split_frames,
            model_columns=model_columns,
            numerical_columns=numerical_columns,
        ):
            frame = normalize_tokenization_check_frame(frame)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def normalize_tokenization_check_frame(frame: pd.DataFrame) -> pd.DataFrame:
    string_columns = [
        "split",
        "SEQN",
        "cycle",
        "feature_name",
        "feature_type",
        "raw_value_after_labeling",
        "feature_context_text",
        "quantile_edges",
        "category_text",
    ]
    normalized = frame.copy()
    for column in string_columns:
        normalized[column] = normalized[column].where(normalized[column].notna(), None).astype("string")
    return normalized


def iter_tokenization_check_frames(
    *,
    preprocessor: NameValuePreprocessor,
    batches: dict[str, object],
    split_frames: dict[str, pd.DataFrame],
    model_columns: list[str],
    numerical_columns: list[str],
):
    numerical_set = set(numerical_columns)
    feature_types = preprocessor.feature_type_ids().tolist()
    context_texts_by_feature = preprocessor.feature_context_texts_by_feature
    category_labels_by_feature = {
        feature_name: preprocessor.categorical_labels(feature_name)
        for feature_name in model_columns
    }
    for split in ["train", "val", "test"]:
        if split not in split_frames or split not in batches:
            continue
        frame = split_frames[split]
        batch = batches[split]
        n_rows = len(frame)
        row_idx_in_split = np.arange(n_rows, dtype=np.int64)
        row_ids = (
            frame["SEQN"].astype("object").where(frame["SEQN"].notna(), None).to_numpy()
            if "SEQN" in frame.columns
            else frame.index.astype("object").to_numpy()
        )
        cycles = (
            frame["cycle"].astype("object").where(frame["cycle"].notna(), None).astype(str).to_numpy()
            if "cycle" in frame.columns
            else np.full(n_rows, "", dtype=object)
        )
        for feature_idx, feature_name in enumerate(model_columns):
            raw_values = frame[feature_name].astype("object").where(frame[feature_name].notna(), None).to_numpy()
            labels = category_labels_by_feature.get(feature_name, [])
            context_texts = context_texts_by_feature.get(feature_name, [])
            category_codes = batch.categorical_codes[:, feature_idx].cpu().numpy().astype(np.int64, copy=False)
            feature_context_codes = (
                batch.feature_context_codes[:, feature_idx].cpu().numpy().astype(np.int64, copy=False)
            )
            category_texts: list[str | None]
            if labels:
                category_text_by_code = []
                for context_code, label in labels:
                    context_text = context_texts[context_code] if context_code < len(context_texts) else feature_name
                    category_text_by_code.append(f"Question:\n{context_text}\nAnswer: {label}.")
                category_texts = [category_text_by_code[int(code)] for code in category_codes]
            else:
                category_texts = [None] * n_rows
            feature_context_texts = [
                context_texts[int(code)] if int(code) < len(context_texts) else None
                for code in feature_context_codes
            ]
            numeric_mean = preprocessor.numeric_mean(feature_name)
            numeric_std = preprocessor.numeric_std(feature_name)
            yield pd.DataFrame(
                {
                    "split": split,
                    "row_idx_in_split": row_idx_in_split,
                    "SEQN": row_ids,
                    "cycle": cycles,
                    "feature_idx": np.full(n_rows, feature_idx, dtype=np.int64),
                    "feature_name": feature_name,
                    "feature_type_id": np.full(n_rows, int(feature_types[feature_idx]), dtype=np.int64),
                    "feature_type": "numerical" if feature_name in numerical_set else "categorical",
                    "raw_value_after_labeling": raw_values,
                    "feature_context_code": feature_context_codes,
                    "feature_context_text": feature_context_texts,
                    "numeric_mean": np.full(
                        n_rows,
                        np.nan if numeric_mean is None else float(numeric_mean),
                        dtype=np.float64,
                    ),
                    "numeric_std": np.full(
                        n_rows,
                        np.nan if numeric_std is None else float(numeric_std),
                        dtype=np.float64,
                    ),
                    "quantile_edges": json.dumps(preprocessor.quantile_edges(feature_name)),
                    "numeric_value_standardized": (
                        batch.numeric_values[:, feature_idx].cpu().numpy().astype(np.float64, copy=False)
                    ),
                    "continuous_bin_code": (
                        batch.continuous_bin_codes[:, feature_idx].cpu().numpy().astype(np.int64, copy=False)
                    ),
                    "categorical_code": category_codes,
                    "category_text": category_texts,
                    "missing_reason_code": (
                        batch.missing_reason_codes[:, feature_idx].cpu().numpy().astype(np.int64, copy=False)
                    ),
                    "missing_mask": batch.missing_mask[:, feature_idx].cpu().numpy().astype(bool, copy=False),
                    "observed_mask": batch.observed_mask[:, feature_idx].cpu().numpy().astype(bool, copy=False),
                }
            )
