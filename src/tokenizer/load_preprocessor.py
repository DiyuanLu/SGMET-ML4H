"""Rebuild a fitted NameValuePreprocessor from NHANES tokenization artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .preprocessing import MISSING_CATEGORY, OTHER_CATEGORY, NameValuePreprocessor
from .schema import FeatureSpec, FeatureType


def load_preprocessor_from_token_dir(token_dir: Path | str) -> NameValuePreprocessor:
    """Load the NHANES-fitted preprocessor saved alongside token tensors."""
    token_dir = Path(token_dir)
    metadata_path = token_dir / "tokenizer_metadata.pt"
    plan_path = token_dir / "tokenization_plan.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing tokenizer metadata: {metadata_path}")
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Missing tokenization plan: {plan_path}. "
            "Run tokenize_nhanes.py on the reference cohort first."
        )

    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    plan = pd.read_parquet(plan_path)
    feature_names = list(metadata["feature_names"])
    missing_reason_path = token_dir.parent / "nhanes_2011_2023_missing_reason_codes.json"
    if missing_reason_path.exists():
        missing_reason_codes = json.loads(missing_reason_path.read_text(encoding="utf-8"))
    else:
        missing_reason_codes = {
            "not_missing": 0,
            "skipped": 1,
            "item_missing": 2,
            "not_eligible": 3,
            "variable_not_in_cycle": 4,
            "unknown_missing": 5,
            "refused": 6,
            "dont_know": 7,
            "masked": 8,
        }

    context_texts_by_feature: dict[str, list[str]] = {}
    for feature_name in feature_names:
        rows = plan.loc[plan["feature_name"].eq(feature_name)].sort_values("context_code")
        context_texts_by_feature[feature_name] = rows["feature_context_text"].astype(str).tolist()

    feature_specs: list[FeatureSpec] = []
    for feature_name in feature_names:
        row = plan.loc[plan["feature_name"].eq(feature_name)].iloc[0]
        feature_type = FeatureType.CATEGORICAL if row["feature_type"] == "categorical" else FeatureType.NUMERICAL
        feature_specs.append(FeatureSpec(name=feature_name, feature_type=feature_type))

    continuous_bins = int(metadata.get("continuous_bin_cardinality", 11)) - 1
    preprocessor = NameValuePreprocessor(
        feature_specs,
        continuous_quantile_bins=max(continuous_bins, 1),
        missing_reason_codes=missing_reason_codes,
        feature_context_texts_by_feature=context_texts_by_feature,
    )

    preprocessor._means = {}
    preprocessor._stds = {}
    preprocessor._quantile_edges = {}
    preprocessor._category_maps = {}
    preprocessor._category_labels_by_feature = {}

    for feature_name in feature_names:
        feature_rows = plan.loc[plan["feature_name"].eq(feature_name)].sort_values("context_code")
        first = feature_rows.iloc[0]
        if first["feature_type"] == "numerical":
            preprocessor._means[feature_name] = float(first["numeric_mean"])
            preprocessor._stds[feature_name] = float(first["numeric_std"]) if float(first["numeric_std"]) > 0 else 1.0
            edges = json.loads(first["quantile_edges"]) if pd.notna(first["quantile_edges"]) else []
            preprocessor._quantile_edges[feature_name] = np.asarray(edges, dtype=np.float64)
        else:
            labels: list[tuple[int, str]] = []
            category_map: dict[tuple[int, str], int] = {}
            for _, row in feature_rows.iterrows():
                context_code = int(row["context_code"])
                context_texts = json.loads(row["context_category_texts"])
                for text_idx, text in enumerate(context_texts):
                    answer = _parse_answer_label(str(text))
                    key = (context_code, answer)
                    if key not in category_map:
                        category_map[key] = len(labels)
                        labels.append(key)
            preprocessor._category_labels_by_feature[feature_name] = labels
            preprocessor._category_maps[feature_name] = category_map

    preprocessor._fitted = True
    return preprocessor


def _parse_answer_label(category_text: str) -> str:
    marker = "\nAnswer: "
    if marker in category_text:
        return category_text.rsplit(marker, maxsplit=1)[-1].strip().rstrip(".")
    if category_text.endswith("."):
        return category_text[:-1]
    return category_text


def default_feature_context_frame(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    default_context_code: int = 0,
) -> pd.DataFrame:
    """Build a zero context-code frame for external cohorts without NHANES cycle maps."""
    codes = np.full((len(frame), len(feature_names)), default_context_code, dtype=np.int64)
    return pd.DataFrame(codes, columns=feature_names, index=frame.index)
