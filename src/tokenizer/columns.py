from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {"SEQN", "cycle", "SEQN_cycle", "split"}
CODEBOOK_LOOKUP_PREFIXES = ("GLU_", "TRIGLY_")
RANGE_LABEL = "range of values"
PROCEDURAL_MISSING_MARKERS = (
    "could not obtain",
    "provider did not specify",
    "never had",
    "never heard",
)


@dataclass(frozen=True)
class ValueLabelSpec:
    labels_by_cycle: dict[str, dict[str, str]]

    @property
    def has_range(self) -> bool:
        return any(has_range_label(labels) for labels in self.labels_by_cycle.values())


def load_registry_target_columns(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task registry must contain a JSON object: {path}")

    target_columns = []
    seen = set()
    for task_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Task {task_name!r} must contain a JSON object in {path}")
        target_column = entry.get("target_column")
        if not target_column:
            raise ValueError(f"Task {task_name!r} is missing target_column in {path}")
        target_column = str(target_column)
        if target_column not in seen:
            target_columns.append(target_column)
            seen.add(target_column)
    return target_columns


def load_registry_drop_columns(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task registry must contain a JSON object: {path}")

    drop_columns = []
    seen = set()
    for task_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Task {task_name!r} must contain a JSON object in {path}")
        for column in entry.get("cols_to_drop", []):
            column = str(column)
            if column not in seen:
                drop_columns.append(column)
                seen.add(column)
    return drop_columns


def load_value_label_specs(path: Path | str | None) -> dict[str, ValueLabelSpec]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"[WARN] NHANES category label map file not found: {path}")
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Category label map must contain a JSON object: {path}")

    labels_by_feature: dict[str, dict[str, dict[str, str]]] = {}
    for key, labels in raw.items():
        if not isinstance(labels, dict):
            continue
        parts = str(key).split("|", 2)
        if len(parts) != 3:
            continue
        cycle, _file_name, feature_name = parts
        normalized = {
            normalize_code(code): str(label)
            for code, label in labels.items()
            if normalize_code(code) is not None and str(label).strip()
        }
        if normalized:
            labels_by_feature.setdefault(feature_name, {})[cycle] = normalized
    specs = {
        feature_name: ValueLabelSpec(labels_by_cycle=labels_by_cycle)
        for feature_name, labels_by_cycle in labels_by_feature.items()
    }
    for feature_name in list(labels_by_feature):
        for prefix in CODEBOOK_LOOKUP_PREFIXES:
            specs[f"{prefix}{feature_name}"] = ValueLabelSpec(labels_by_cycle=labels_by_feature[feature_name])
    return specs


def model_feature_columns(
    *frames: pd.DataFrame,
    target_columns: Iterable[str] = (),
    drop_columns: Iterable[str] = (),
) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    target_column_set = set(target_columns)
    drop_column_set = set(drop_columns)
    return [
        column
        for column in frames[0].columns
        if (
            column in common
            and column not in IDENTIFIER_COLUMNS
            and not column.endswith("__missing_reason")
            and column not in target_column_set
            and column not in drop_column_set
        )
    ]


def infer_categorical_and_binary_columns(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = []
    binary_columns = []
    for column in X_train.columns:
        series = X_train[column]
        non_missing = series.dropna()
        unique_values = set(non_missing.unique().tolist())
        if pd.api.types.is_bool_dtype(series) or unique_values <= {0, 1, 0.0, 1.0}:
            categorical_columns.append(column)
        elif not pd.api.types.is_numeric_dtype(series):
            categorical_columns.append(column)
    return categorical_columns, binary_columns


def infer_columns_from_value_labels(
    X_train: pd.DataFrame,
    *,
    value_label_specs: dict[str, ValueLabelSpec],
) -> tuple[list[str], list[str], dict[str, object]]:
    numerical_columns = []
    categorical_columns = []
    missing_label_map_columns = []
    for column in X_train.columns:
        spec = value_label_specs.get(column)
        if spec is None:
            missing_label_map_columns.append(column)
        elif spec.has_range:
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)

    report = {
        "label_map_numerical_columns": sorted(
            column for column in X_train.columns if value_label_specs.get(column) is not None and value_label_specs[column].has_range
        ),
        "label_map_categorical_columns": sorted(
            column for column in X_train.columns if value_label_specs.get(column) is not None and not value_label_specs[column].has_range
        ),
        "missing_label_map_columns": sorted(missing_label_map_columns),
    }
    return numerical_columns, categorical_columns, report


def sanitize_range_feature_values(
    frames: Iterable[pd.DataFrame],
    *,
    value_label_specs: dict[str, ValueLabelSpec],
    missing_reason_codes: dict[str, int] | None,
) -> None:
    codes = missing_reason_codes or {}
    not_missing_code = codes.get("not_missing", 0)
    unknown_missing_code = codes.get("unknown_missing", 6)
    refused_code = codes.get("refused", unknown_missing_code)
    dont_know_code = codes.get("dont_know", unknown_missing_code)

    range_specs = {
        feature_name: spec
        for feature_name, spec in value_label_specs.items()
        if spec.has_range
    }
    for frame in frames:
        for feature_name, spec in range_specs.items():
            if feature_name not in frame.columns:
                continue
            reason_col = f"{feature_name}__missing_reason"
            if reason_col not in frame.columns:
                frame[reason_col] = np.where(frame[feature_name].isna(), unknown_missing_code, not_missing_code)

            if "cycle" in frame.columns:
                cycle_values = frame["cycle"].astype(str)
                cycle_keys = sorted(cycle_values.dropna().unique().tolist())
            else:
                cycle_values = pd.Series("", index=frame.index)
                cycle_keys = [""]

            for cycle in cycle_keys:
                labels = spec.labels_by_cycle.get(cycle) or merged_cycle_labels(spec)
                idx = frame.index[cycle_values.eq(cycle)] if cycle else frame.index
                sanitize_range_feature_for_rows(
                    frame,
                    idx,
                    feature_name=feature_name,
                    labels=labels,
                    reason_col=reason_col,
                    not_missing_code=not_missing_code,
                    unknown_missing_code=unknown_missing_code,
                    refused_code=refused_code,
                    dont_know_code=dont_know_code,
                )


def sanitize_range_feature_for_rows(
    frame: pd.DataFrame,
    idx: pd.Index,
    *,
    feature_name: str,
    labels: dict[str, str],
    reason_col: str,
    not_missing_code: int,
    unknown_missing_code: int,
    refused_code: int,
    dont_know_code: int,
) -> None:
    if len(idx) == 0:
        return
    ranges = [parsed for code in labels for parsed in [parse_range_code(code)] if parsed is not None]
    values = pd.to_numeric(frame.loc[idx, feature_name], errors="coerce")
    normalized_values = frame.loc[idx, feature_name].map(normalize_code)
    reason = pd.to_numeric(frame.loc[idx, reason_col], errors="coerce").fillna(unknown_missing_code).astype(int)
    already_missing = reason.ne(not_missing_code) | values.isna()
    handled = already_missing.copy()

    for raw_code, label in labels.items():
        if parse_range_code(raw_code) is not None:
            continue
        normalized_code = normalize_code(raw_code)
        if normalized_code is None:
            continue
        match = normalized_values.eq(normalized_code)
        if not match.any():
            continue
        label_reason = missing_reason_for_label(label, refused_code, dont_know_code, unknown_missing_code)
        if label_reason is not None:
            frame.loc[idx[match], reason_col] = label_reason
            frame.loc[idx[match], feature_name] = np.nan
            handled.loc[match] = True
            continue
        cap_value = numeric_cap_from_label(label)
        if cap_value is not None:
            frame.loc[idx[match], feature_name] = cap_value
            frame.loc[idx[match], reason_col] = not_missing_code
            handled.loc[match] = True

    values_after = pd.to_numeric(frame.loc[idx, feature_name], errors="coerce")
    in_range = pd.Series(False, index=idx)
    for lower, upper in ranges:
        in_range |= values_after.between(lower, upper, inclusive="both")
    invalid_observed = ~handled & ~in_range
    if invalid_observed.any():
        frame.loc[idx[invalid_observed], reason_col] = unknown_missing_code
        frame.loc[idx[invalid_observed], feature_name] = np.nan


def merged_cycle_labels(spec: ValueLabelSpec) -> dict[str, str]:
    merged = {}
    for labels in spec.labels_by_cycle.values():
        merged.update(labels)
    return merged


def has_range_label(labels: dict[str, str]) -> bool:
    return any(parse_range_code(code) is not None or str(label).strip().lower() == RANGE_LABEL for code, label in labels.items())


def parse_range_code(value: object) -> tuple[float, float] | None:
    text = str(value).strip()
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s+to\s+([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    lower = float(match.group(1))
    upper = float(match.group(2))
    return (lower, upper)


def missing_reason_for_label(label: object, refused_code: int, dont_know_code: int, unknown_missing_code: int) -> int | None:
    text = str(label).strip().lower().replace("’", "'")
    if "refused" in text:
        return refused_code
    if "don't know" in text or "dont know" in text:
        return dont_know_code
    if any(marker in text for marker in PROCEDURAL_MISSING_MARKERS):
        return unknown_missing_code
    return None


def numeric_cap_from_label(label: object) -> float | None:
    text = str(label).strip().lower()
    if text in {"none", "never"}:
        return 0.0
    if not any(marker in text for marker in ("less than", "or less", "or more", "or older", "and over", "more than", "greater than or equal")):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def normalize_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value).rstrip("0").rstrip(".")
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return str(int(float(text)))
    return text or None


def load_missing_reason_codes(path: Path) -> dict[str, int] | None:
    if not path.exists():
        print(f"[WARN] missing reason code file not found: {path}")
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Missing reason code map must contain a JSON object: {path}")
    return {str(key): int(value) for key, value in raw.items()}
