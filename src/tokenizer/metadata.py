from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


DEFAULT_METADATA_PATH = Path("data/external/codebooks/nhanes_metadata.parquet")
DEFAULT_CATEGORY_LABEL_MAPS_PATH = Path("data/external/codebooks/nhanes_category_label_maps.json")
CODEBOOK_LOOKUP_PREFIXES = ("GLU_", "TRIGLY_")

FEATURE_CONTEXT_COLUMNS = [
    "variable_name",
    "sas_label",
    "english_text",
    "variable_description",
    "target",
    "unit",
    "hard_edits",
    "value_labels_json",
]

VALID_UNIT_PATTERN = re.compile(
    r"^(cm|kg|kg/m\*\*2|mg/dL|mmol/L|mm Hg|pmol/L|uU/mL|day/week/month/year|month/year|"
    r"30 sec\. pulse \* 2)$",
    re.IGNORECASE,
)
NUMERIC_RANGE_PATTERN = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s+to\s+[-+]?\d+(?:\.\d+)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class FeatureMetadata:
    descriptions: dict[str, str]
    feature_context_texts: dict[str, list[str]]
    feature_context_cycle_maps: dict[str, dict[str, int]]
    units: dict[str, str]
    value_labels: dict[str, dict[str, str]]
    cycle_value_labels: dict[str, dict[str, dict[str, str]]]
    matched_features: set[str]


def load_feature_metadata(
    metadata_path: Path | str,
    feature_names: Sequence[str],
    category_label_maps_path: Path | str | None = DEFAULT_CATEGORY_LABEL_MAPS_PATH,
) -> FeatureMetadata:
    """Load NHANES metadata for the requested feature names only."""
    metadata_path = Path(metadata_path)
    feature_set = set(feature_names)
    if not metadata_path.exists():
        print(f"[WARN] NHANES metadata file not found: {metadata_path}")
        return FeatureMetadata({}, {}, {}, {}, {}, {}, set())

    metadata = read_metadata_frame(metadata_path)
    if "variable_name" not in metadata.columns:
        raise ValueError(f"Metadata file must contain a variable_name column: {metadata_path}")

    alias_to_canonical = {
        feature_name: canonical
        for feature_name in feature_set
        for canonical in [canonical_codebook_feature_name(feature_name)]
        if canonical != feature_name
    }
    metadata = metadata[metadata["variable_name"].isin(feature_set | set(alias_to_canonical.values()))].copy()
    alias_frames = []
    for alias, canonical in alias_to_canonical.items():
        alias_frame = metadata[metadata["variable_name"].eq(canonical)].copy()
        module_prefix = alias_module_prefix(alias)
        if module_prefix is not None and "file_name" in alias_frame.columns:
            alias_frame = alias_frame[
                alias_frame["file_name"].astype(str).str.upper().str.contains(module_prefix)
            ].copy()
        if alias_frame.empty:
            continue
        alias_frame["variable_name"] = alias
        alias_frames.append(alias_frame)
    if alias_frames:
        metadata = pd.concat([metadata, *alias_frames], ignore_index=True)
    metadata = metadata[metadata["variable_name"].isin(feature_set)].copy()
    category_label_maps = read_category_label_maps(category_label_maps_path)
    descriptions: dict[str, str] = {}
    feature_context_texts: dict[str, list[str]] = {}
    feature_context_cycle_maps: dict[str, dict[str, int]] = {}
    units: dict[str, str] = {}
    value_labels: dict[str, dict[str, str]] = {}
    cycle_value_labels: dict[str, dict[str, dict[str, str]]] = {}

    for feature_name, group in metadata.groupby("variable_name", sort=False):
        descriptions[feature_name] = build_feature_description(group)
        contexts, cycle_map = build_feature_contexts(feature_name, group)
        feature_context_texts[feature_name] = contexts
        feature_context_cycle_maps[feature_name] = cycle_map
        unit = most_common_nonempty(group.get("unit"))
        if is_valid_unit(unit):
            units[feature_name] = unit
        labels = merge_value_labels(group.get("value_labels_json"))
        if labels:
            value_labels[feature_name] = labels
        per_cycle = build_cycle_value_labels(group, category_label_maps)
        if per_cycle:
            cycle_value_labels[feature_name] = per_cycle

    missing_count = len(feature_set - set(metadata["variable_name"].unique()))
    print(
        f"[INFO] loaded NHANES metadata for {metadata['variable_name'].nunique()} "
        f"of {len(feature_set)} features from {metadata_path}"
    )
    if missing_count:
        print(f"[INFO] {missing_count} feature columns had no NHANES metadata match")

    return FeatureMetadata(
        descriptions=descriptions,
        feature_context_texts=feature_context_texts,
        feature_context_cycle_maps=feature_context_cycle_maps,
        units=units,
        value_labels=value_labels,
        cycle_value_labels=cycle_value_labels,
        matched_features=set(metadata["variable_name"].unique()),
    )


def canonical_codebook_feature_name(feature_name: str) -> str:
    for prefix in CODEBOOK_LOOKUP_PREFIXES:
        if feature_name.startswith(prefix):
            return feature_name.removeprefix(prefix)
    return feature_name


def alias_module_prefix(feature_name: str) -> str | None:
    for prefix in CODEBOOK_LOOKUP_PREFIXES:
        if feature_name.startswith(prefix):
            return prefix.rstrip("_").upper()
    return None


def read_metadata_frame(metadata_path: Path) -> pd.DataFrame:
    if metadata_path.suffix.lower() == ".parquet":
        return pd.read_parquet(metadata_path).astype(str).replace("nan", "")
    return pd.read_csv(metadata_path, dtype=str, keep_default_na=False)


def read_category_label_maps(path: Path | str | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"[WARN] NHANES category label map file not found: {path}")
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Category label map must contain a JSON object: {path}")
    maps: dict[str, dict[str, str]] = {}
    for key, labels in raw.items():
        if not isinstance(labels, dict):
            continue
        normalized = {
            normalize_category_code(code): clean(label)
            for code, label in labels.items()
            if normalize_category_code(code) and clean(label)
        }
        if normalized and not has_numeric_range_label(normalized):
            maps[str(key)] = normalized
    return maps


def build_feature_description(group: pd.DataFrame) -> str:
    """Build compact structured semantic text without NHANES code identifiers or boilerplate."""
    row = representative_row(group)
    parts = []

    def add_labeled(label: str, value: str | None) -> None:
        value = clean(value)
        if value:
            parts.append(f"{label}: {format_sentence(value)}")

    english_text = strip_english_instructions(row.get("english_text"))
    variable_description = clean(row.get("variable_description"))
    sas_label = clean(row.get("sas_label"))
    primary_parts = []
    if sas_label:
        primary_parts.append(format_sentence(sas_label))
    if english_text and is_distinct_text(english_text, sas_label):
        primary_parts.append(format_sentence(english_text))
    primary = " ".join(primary_parts) or variable_description or readable_variable_name(row.get("variable_name"))
    if primary:
        add_labeled("Description", primary)
    add_labeled("Target population", row.get("target"))
    hard_edit = clean(row.get("hard_edits"))
    if is_numeric_range(hard_edit):
        add_labeled("Valid numeric range", hard_edit)
    unit = clean(row.get("unit"))
    if is_valid_unit(unit):
        add_labeled("Unit", unit)

    return "\n".join(parts)


def build_feature_contexts(feature_name: str, group: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Build exact per-cycle semantic contexts for one feature."""
    contexts: list[str] = []
    context_index: dict[str, int] = {}
    cycle_map: dict[str, int] = {}
    for cycle, cycle_group in group.groupby("cycle", sort=False):
        descriptions = []
        for _, row in cycle_group.iterrows():
            description = build_feature_description(pd.DataFrame([row]))
            if description not in descriptions:
                descriptions.append(description)
        description = descriptions[0] if len(descriptions) == 1 else build_feature_description(cycle_group)
        if description not in context_index:
            context_index[description] = len(contexts)
            contexts.append(description)
        cycle_map[clean(cycle)] = context_index[description]
    return contexts or [build_feature_description(group)], cycle_map


def representative_row(group: pd.DataFrame) -> Mapping[str, str]:
    values = {}
    for column in FEATURE_CONTEXT_COLUMNS:
        values[column] = most_common_nonempty(group.get(column)) or ""
    return values


def merge_value_labels(values: pd.Series | None) -> dict[str, str]:
    if values is None:
        return {}

    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for raw_json in values:
        raw_json = clean(raw_json)
        if not raw_json:
            continue
        try:
            labels = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(labels, dict):
            continue
        for raw_code, raw_label in labels.items():
            code = normalize_category_code(raw_code)
            label = clean(raw_label)
            if code and label:
                votes[code][label] += 1

    merged = {}
    for code, counter in votes.items():
        label, _ = counter.most_common(1)[0]
        merged[code] = label
    if has_numeric_range_label(merged):
        return {}
    return merged


def apply_value_labels(
    frames: Sequence[pd.DataFrame],
    value_labels: Mapping[str, Mapping[str, str]],
    cycle_value_labels: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
) -> list[pd.DataFrame]:
    """Replace categorical NHANES codes with human-readable labels in each frame."""
    transformed = [frame.copy() for frame in frames]
    cycle_value_labels = cycle_value_labels or {}
    for column, labels in value_labels.items():
        cycle_labels = cycle_value_labels.get(column, {})
        if not labels and not cycle_labels:
            continue
        for frame in transformed:
            if column not in frame.columns:
                continue
            if cycle_labels and "cycle" in frame.columns:
                frame[column] = [
                    label_value_by_cycle(value, cycle, labels, cycle_labels)
                    for value, cycle in zip(frame[column], frame["cycle"])
                ]
            else:
                frame[column] = frame[column].map(lambda value: label_value(value, labels))
    return transformed


def label_value_by_cycle(
    value: object,
    cycle: object,
    fallback_labels: Mapping[str, str],
    cycle_labels: Mapping[str, Mapping[str, str]],
) -> object:
    labels = cycle_labels.get(clean(cycle), fallback_labels)
    return label_value(value, labels)


def label_value(value: object, labels: Mapping[str, str]) -> object:
    if pd.isna(value):
        return value
    key = normalize_category_code(value)
    return labels.get(key, value)


def normalize_category_code(value: object) -> str:
    text = clean(str(value))
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def has_numeric_range_label(labels: Mapping[str, str]) -> bool:
    for code, label in labels.items():
        if code == ".":
            continue
        if " to " in code.lower() or label.lower() == "range of values":
            return True
    return False


def most_common_nonempty(values: pd.Series | None) -> str | None:
    candidates = unique_nonempty(values)
    if not candidates:
        return None
    counts = Counter(clean(value) for value in values if clean(value)) if values is not None else Counter()
    return counts.most_common(1)[0][0]


def unique_nonempty(values: pd.Series | None) -> list[str]:
    if values is None:
        return []
    seen = []
    for value in values:
        value = clean(value)
        if value and value not in seen:
            seen.append(value)
    return seen


def clean(value: object | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def strip_english_instructions(value: object | None) -> str:
    text = clean(value)
    if not text:
        return ""
    return clean(re.split(r"\bEnglish Instructions\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0])


def build_cycle_value_labels(
    group: pd.DataFrame,
    category_label_maps: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    per_cycle: dict[str, dict[str, str]] = {}
    for _, row in group.iterrows():
        cycle = clean(row.get("cycle"))
        file_name = clean(row.get("file_name"))
        variable_name = clean(row.get("variable_name"))
        if not cycle or not variable_name:
            continue
        labels = None
        if file_name:
            labels = category_label_maps.get(f"{cycle}|{file_name}|{variable_name}")
        if labels is None:
            labels = merge_value_labels(pd.Series([row.get("value_labels_json")]))
        if labels:
            per_cycle[cycle] = dict(labels)
    return per_cycle


def is_distinct_text(candidate: str, primary: str) -> bool:
    candidate_norm = normalize_text(candidate)
    primary_norm = normalize_text(primary)
    return bool(candidate_norm and candidate_norm != primary_norm and candidate_norm not in primary_norm)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", clean(value).lower())


def looks_like_instructional_noise(value: str) -> bool:
    text = normalize_text(value)
    noise_markers = [
        "read hand card",
        "english instructions",
        "enter highest",
        "display quantity",
    ]
    return any(marker in text for marker in noise_markers) and len(text) > 180


def is_valid_unit(value: object | None) -> bool:
    value = clean(value)
    if not value:
        return False
    return bool(VALID_UNIT_PATTERN.match(value))


def is_numeric_range(value: object | None) -> bool:
    return bool(NUMERIC_RANGE_PATTERN.match(clean(value)))


def readable_variable_name(value: object | None) -> str:
    text = clean(value)
    if not text:
        return ""
    if "_" in text:
        return clean(text.replace("_", " ").lower())
    return text


def format_sentence(value: str) -> str:
    value = clean(value)
    if not value:
        return ""
    return value if value.endswith((".", "?", "!")) else f"{value}."
