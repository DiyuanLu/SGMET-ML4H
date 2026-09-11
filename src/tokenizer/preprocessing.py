from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .schema import FeatureSpec, FeatureType


MISSING_CATEGORY = "MISSING"
OTHER_CATEGORY = "OTHER"
MASKED_MISSING_REASON = "masked"
DEFAULT_MISSING_REASON_CODES = {
    "not_missing": 0,
    "skipped": 1,
    "item_missing": 2,
    "not_eligible": 3,
    "variable_not_in_cycle": 4,
    "unknown_missing": 5,
    "refused": 6,
    "dont_know": 7,
    MASKED_MISSING_REASON: 8,
}


@dataclass(frozen=True)
class TransformedBatch:
    """Tensor batch aligned to the preprocessor's feature order."""

    numeric_values: torch.Tensor
    continuous_bin_codes: torch.Tensor
    categorical_codes: torch.Tensor
    feature_context_codes: torch.Tensor
    missing_reason_codes: torch.Tensor
    missing_mask: torch.Tensor
    observed_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "TransformedBatch":
        return TransformedBatch(
            numeric_values=self.numeric_values.to(device),
            continuous_bin_codes=self.continuous_bin_codes.to(device),
            categorical_codes=self.categorical_codes.to(device),
            feature_context_codes=self.feature_context_codes.to(device),
            missing_reason_codes=self.missing_reason_codes.to(device),
            missing_mask=self.missing_mask.to(device),
            observed_mask=self.observed_mask.to(device),
        )


def transformed_batch_to_dict(batch: TransformedBatch) -> dict[str, torch.Tensor]:
    return {
        "numeric_values": batch.numeric_values,
        "continuous_bin_codes": batch.continuous_bin_codes,
        "categorical_codes": batch.categorical_codes,
        "feature_context_codes": batch.feature_context_codes,
        "missing_reason_codes": batch.missing_reason_codes,
        "missing_mask": batch.missing_mask,
        "observed_mask": batch.observed_mask,
    }


def transformed_batch_from_dict(payload: Mapping[str, torch.Tensor]) -> TransformedBatch:
    required = {
        "numeric_values",
        "continuous_bin_codes",
        "categorical_codes",
        "missing_mask",
        "observed_mask",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Token payload is missing required tensors: {missing}")
    feature_context_codes = payload.get("feature_context_codes")
    if feature_context_codes is None:
        feature_context_codes = torch.zeros_like(payload["categorical_codes"], dtype=torch.long)
    missing_reason_codes = payload.get("missing_reason_codes")
    if missing_reason_codes is None:
        raise ValueError("Token payload is missing required tensors: ['missing_reason_codes']")
    return TransformedBatch(
        numeric_values=payload["numeric_values"],
        continuous_bin_codes=payload["continuous_bin_codes"],
        categorical_codes=payload["categorical_codes"],
        feature_context_codes=feature_context_codes,
        missing_reason_codes=missing_reason_codes,
        missing_mask=payload["missing_mask"],
        observed_mask=payload["observed_mask"],
    )


def slice_transformed_batch(batch: TransformedBatch, max_rows: int | None) -> TransformedBatch:
    if max_rows is None:
        return batch
    return TransformedBatch(
        numeric_values=batch.numeric_values[:max_rows],
        continuous_bin_codes=batch.continuous_bin_codes[:max_rows],
        categorical_codes=batch.categorical_codes[:max_rows],
        feature_context_codes=batch.feature_context_codes[:max_rows],
        missing_reason_codes=batch.missing_reason_codes[:max_rows],
        missing_mask=batch.missing_mask[:max_rows],
        observed_mask=batch.observed_mask[:max_rows],
    )


def index_transformed_batch(batch: TransformedBatch, index: torch.Tensor) -> TransformedBatch:
    return TransformedBatch(
        numeric_values=batch.numeric_values[index],
        continuous_bin_codes=batch.continuous_bin_codes[index],
        categorical_codes=batch.categorical_codes[index],
        feature_context_codes=batch.feature_context_codes[index],
        missing_reason_codes=batch.missing_reason_codes[index],
        missing_mask=batch.missing_mask[index],
        observed_mask=batch.observed_mask[index],
    )


def sample_value_mask(
    batch: TransformedBatch,
    mask_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample reconstruction mask positions over originally observed values."""
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be between 0 and 1.")
    random_values = torch.rand(batch.observed_mask.shape, generator=generator, device=batch.observed_mask.device)
    return (random_values < mask_probability) & batch.observed_mask & ~batch.missing_mask


def mask_batch_values(batch: TransformedBatch, mask_positions: torch.Tensor) -> TransformedBatch:
    """Hide selected values while preserving the original batch as reconstruction targets."""
    if mask_positions.shape != batch.missing_mask.shape:
        raise ValueError("mask_positions must have the same shape as batch.missing_mask.")
    numeric_values = batch.numeric_values.clone()
    continuous_bin_codes = batch.continuous_bin_codes.clone()
    categorical_codes = batch.categorical_codes.clone()
    feature_context_codes = batch.feature_context_codes.clone()
    missing_reason_codes = batch.missing_reason_codes.clone()
    missing_mask = batch.missing_mask.clone()

    numeric_values[mask_positions] = 0.0
    continuous_bin_codes[mask_positions] = 0
    categorical_codes[mask_positions] = 0
    missing_reason_codes[mask_positions] = DEFAULT_MISSING_REASON_CODES[MASKED_MISSING_REASON]
    missing_mask[mask_positions] = True

    return TransformedBatch(
        numeric_values=numeric_values,
        continuous_bin_codes=continuous_bin_codes,
        categorical_codes=categorical_codes,
        feature_context_codes=feature_context_codes,
        missing_reason_codes=missing_reason_codes,
        missing_mask=missing_mask,
        observed_mask=batch.observed_mask,
    )


class NameValuePreprocessor:
    """
    Train-split-only preprocessing for NHANES name-value feature tokens.

    Numerical features are standardized with statistics fitted on training
    rows only. Missing standardized values are set to 0. Categorical missing
    values map to MISSING and unseen values at transform time map to OTHER.
    """

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        continuous_quantile_bins: int = 10,
        missing_reason_codes: Mapping[str, int] | None = None,
        feature_context_texts_by_feature: Mapping[str, Sequence[str]] | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.feature_specs = list(feature_specs)
        self.continuous_quantile_bins = continuous_quantile_bins
        self.missing_reason_codes = dict(missing_reason_codes or DEFAULT_MISSING_REASON_CODES)
        if MASKED_MISSING_REASON not in self.missing_reason_codes:
            self.missing_reason_codes[MASKED_MISSING_REASON] = max(self.missing_reason_codes.values(), default=-1) + 1
        self.eps = eps
        self.feature_names = [spec.name for spec in self.feature_specs]
        context_texts = feature_context_texts_by_feature or {}
        self.feature_context_texts_by_feature = {
            name: list(context_texts.get(name) or [self.feature_specs[idx].to_feature_text()])
            for idx, name in enumerate(self.feature_names)
        }
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._quantile_edges: dict[str, np.ndarray] = {}
        self._category_maps: dict[str, dict[tuple[int, str], int]] = {}
        self._category_labels_by_feature: dict[str, list[tuple[int, str]]] = {}
        self._fitted = False

    @classmethod
    def infer_from_dataframe(
        cls,
        df_train: pd.DataFrame,
        numerical_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] | None = None,
        descriptions: Mapping[str, str] | None = None,
        units: Mapping[str, str] | None = None,
        feature_context_texts_by_feature: Mapping[str, Sequence[str]] | None = None,
        log_transform_columns: Sequence[str] | None = None,
        **kwargs: object,
    ) -> "NameValuePreprocessor":
        numerical = set(numerical_columns or [])
        categorical = set(categorical_columns or [])
        log_columns = set(log_transform_columns or [])
        descriptions = descriptions or {}
        units = units or {}

        assigned = numerical | categorical
        for column in df_train.columns:
            if column in assigned:
                continue
            series = df_train[column]
            non_missing = series.dropna()
            unique_values = set(non_missing.unique().tolist())
            if pd.api.types.is_bool_dtype(series) or unique_values <= {0, 1, 0.0, 1.0}:
                categorical.add(column)
            elif pd.api.types.is_numeric_dtype(series):
                numerical.add(column)
            else:
                categorical.add(column)

        specs = []
        for column in df_train.columns:
            if column in categorical:
                feature_type = FeatureType.CATEGORICAL
            else:
                feature_type = FeatureType.NUMERICAL
            specs.append(
                FeatureSpec(
                    name=column,
                    feature_type=feature_type,
                    unit=units.get(column),
                    description=descriptions.get(column),
                    apply_log=column in log_columns,
                )
            )
        return cls(specs, feature_context_texts_by_feature=feature_context_texts_by_feature, **kwargs)

    def fit(self, df_train: pd.DataFrame, feature_context_codes: pd.DataFrame | np.ndarray | torch.Tensor | None = None) -> "NameValuePreprocessor":
        self._validate_columns(df_train)
        context_codes = self._feature_context_code_array(df_train, feature_context_codes)
        not_missing_code = self.missing_reason_codes.get("not_missing", 0)
        for idx, spec in enumerate(self.feature_specs):
            series = df_train[spec.name]
            raw_missing = series.isna().to_numpy()
            reasons = self._missing_reason_series(df_train, spec.name, raw_missing)
            value_series = series.mask(raw_missing | (reasons != not_missing_code))
            if spec.feature_type == FeatureType.NUMERICAL:
                values = self._numeric_series(value_series, spec.apply_log)
                self._means[spec.name] = float(values.mean(skipna=True)) if values.notna().any() else 0.0
                std = float(values.std(skipna=True, ddof=0)) if values.notna().any() else 1.0
                self._stds[spec.name] = std if std > self.eps else 1.0
                self._quantile_edges[spec.name] = self._fit_quantile_edges(values)
            elif spec.feature_type == FeatureType.CATEGORICAL:
                normalized = value_series.astype("object").where(value_series.notna(), MISSING_CATEGORY).astype(str)
                labels: list[tuple[int, str]] = []
                for context_code in range(self.feature_context_cardinality(spec.name)):
                    context_mask = context_codes[:, idx] == context_code
                    context_values = normalized[context_mask]
                    kept = sorted(context_values.unique().tolist())
                    categories = [MISSING_CATEGORY, OTHER_CATEGORY]
                    categories.extend(c for c in kept if c not in categories)
                    labels.extend((context_code, category) for category in categories)
                self._category_labels_by_feature[spec.name] = labels
                self._category_maps[spec.name] = {label: label_idx for label_idx, label in enumerate(labels)}
        self._fitted = True
        return self

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        feature_context_codes: pd.DataFrame | np.ndarray | torch.Tensor | None = None,
    ) -> TransformedBatch:
        return self.fit(df_train, feature_context_codes).transform(df_train, feature_context_codes)

    def transform(
        self,
        df: pd.DataFrame,
        feature_context_codes: pd.DataFrame | np.ndarray | torch.Tensor | None = None,
    ) -> TransformedBatch:
        if not self._fitted:
            raise RuntimeError("NameValuePreprocessor must be fitted before transform().")
        self._validate_columns(df)
        context_codes = self._feature_context_code_array(df, feature_context_codes)

        n_rows = len(df)
        n_features = len(self.feature_specs)
        numeric_values = np.zeros((n_rows, n_features), dtype=np.float32)
        continuous_bin_codes = np.zeros((n_rows, n_features), dtype=np.int64)
        categorical_codes = np.zeros((n_rows, n_features), dtype=np.int64)
        missing_reason_codes = np.zeros((n_rows, n_features), dtype=np.int64)
        missing_mask = np.zeros((n_rows, n_features), dtype=bool)
        observed_mask = np.ones((n_rows, n_features), dtype=bool)
        not_missing_code = self.missing_reason_codes.get("not_missing", 0)

        for idx, spec in enumerate(self.feature_specs):
            series = df[spec.name]
            raw_missing = series.isna().to_numpy()
            reasons = self._missing_reason_series(df, spec.name, raw_missing)
            missing = raw_missing | (reasons != not_missing_code)
            missing_mask[:, idx] = missing
            feature_context = context_codes[:, idx]
            missing_reason_codes[:, idx] = reasons
            value_series = series.mask(missing)
            if spec.feature_type == FeatureType.NUMERICAL:
                values = self._numeric_series(value_series, spec.apply_log)
                standardized = (values - self._means[spec.name]) / self._stds[spec.name]
                numeric_values[:, idx] = standardized.fillna(0.0).astype(np.float32).to_numpy()
                continuous_bin_codes[:, idx] = self._quantile_bin_codes(spec.name, values)
            elif spec.feature_type == FeatureType.CATEGORICAL:
                category_map = self._category_maps[spec.name]
                normalized = value_series.astype("object").where(value_series.notna(), MISSING_CATEGORY).astype(str)
                categorical_codes[:, idx] = [
                    category_map.get((int(context_code), value), category_map[(int(context_code), OTHER_CATEGORY)])
                    for context_code, value in zip(feature_context, normalized)
                ]

        return TransformedBatch(
            numeric_values=torch.from_numpy(numeric_values),
            continuous_bin_codes=torch.from_numpy(continuous_bin_codes),
            categorical_codes=torch.from_numpy(categorical_codes),
            feature_context_codes=torch.from_numpy(context_codes),
            missing_reason_codes=torch.from_numpy(missing_reason_codes),
            missing_mask=torch.from_numpy(missing_mask),
            observed_mask=torch.from_numpy(observed_mask),
        )

    def feature_type_ids(self) -> torch.Tensor:
        return torch.tensor([int(spec.feature_type) for spec in self.feature_specs], dtype=torch.long)

    def categorical_cardinalities(self) -> list[int]:
        cardinalities = []
        for spec in self.feature_specs:
            if spec.feature_type == FeatureType.CATEGORICAL:
                cardinalities.append(len(self._category_maps.get(spec.name, {(0, MISSING_CATEGORY): 0, (0, OTHER_CATEGORY): 1})))
            else:
                cardinalities.append(1)
        return cardinalities

    def feature_context_cardinality(self, feature_name: str) -> int:
        return len(self.feature_context_texts_by_feature.get(feature_name) or [""])

    def feature_context_cardinalities(self) -> list[int]:
        return [self.feature_context_cardinality(name) for name in self.feature_names]

    def feature_context_offsets(self) -> torch.Tensor:
        offsets = []
        current = 0
        for cardinality in self.feature_context_cardinalities():
            offsets.append(current)
            current += cardinality
        return torch.tensor(offsets, dtype=torch.long)

    def numeric_mean(self, feature_name: str) -> float | None:
        return self._means.get(feature_name)

    def numeric_std(self, feature_name: str) -> float | None:
        return self._stds.get(feature_name)

    def quantile_edges(self, feature_name: str) -> list[float]:
        return [float(value) for value in self._quantile_edges.get(feature_name, np.array([], dtype=np.float64))]

    def categorical_labels(self, feature_name: str) -> list[tuple[int, str]]:
        return list(self._category_labels_by_feature.get(feature_name, []))

    def flat_feature_context_texts(self) -> list[str]:
        texts: list[str] = []
        for feature_name in self.feature_names:
            texts.extend(self.feature_context_texts_by_feature[feature_name])
        return texts

    def continuous_bin_cardinality(self) -> int:
        return max(int(self.continuous_quantile_bins), 1) + 1

    def missing_reason_cardinality(self) -> int:
        return max(self.missing_reason_codes.values(), default=0) + 1

    def feature_texts(self) -> list[str]:
        return self.flat_feature_context_texts()

    def categorical_value_texts(self) -> dict[str, dict[str, list[str]]]:
        if not self._fitted:
            raise RuntimeError("NameValuePreprocessor must be fitted before categorical_value_texts().")
        texts: dict[str, dict[str, list[str]]] = {}
        for feature_name, category_map in self._category_maps.items():
            labels = sorted(category_map, key=category_map.get)
            feature_texts = self.feature_context_texts_by_feature[feature_name]
            by_context: dict[str, list[str]] = {}
            for context_code, category in labels:
                context_text = feature_texts[context_code]
                category_prefix = f"Question:\n{context_text}" if context_text else f"Question: {feature_name}."
                by_context.setdefault(str(context_code), []).append(f"{category_prefix}\nAnswer: {category}.")
            texts[feature_name] = by_context
        return texts

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [name for name in self.feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Dataframe is missing required feature columns: {missing}")

    def _feature_context_code_array(
        self,
        df: pd.DataFrame,
        feature_context_codes: pd.DataFrame | np.ndarray | torch.Tensor | None,
    ) -> np.ndarray:
        n_rows = len(df)
        n_features = len(self.feature_specs)
        if feature_context_codes is None:
            return np.zeros((n_rows, n_features), dtype=np.int64)
        if isinstance(feature_context_codes, pd.DataFrame):
            missing = [name for name in self.feature_names if name not in feature_context_codes.columns]
            if missing:
                raise ValueError(f"Feature context codes are missing feature columns: {missing}")
            array = feature_context_codes[self.feature_names].to_numpy(dtype=np.int64)
        elif isinstance(feature_context_codes, torch.Tensor):
            array = feature_context_codes.detach().cpu().numpy().astype(np.int64)
        else:
            array = np.asarray(feature_context_codes, dtype=np.int64)
        array = array.copy()
        if array.shape != (n_rows, n_features):
            raise ValueError(f"feature_context_codes must have shape [{n_rows}, {n_features}], got {array.shape}.")
        for idx, feature_name in enumerate(self.feature_names):
            cardinality = self.feature_context_cardinality(feature_name)
            invalid = (array[:, idx] < 0) | (array[:, idx] >= cardinality)
            if invalid.any():
                raise ValueError(f"Feature context code out of range for {feature_name}.")
        return array

    @staticmethod
    def _numeric_series(series: pd.Series, apply_log: bool) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        if apply_log:
            values = np.log1p(values.clip(lower=0.0))
        return values

    def _fit_quantile_edges(self, values: pd.Series) -> np.ndarray:
        clean_values = values.dropna().astype(float)
        if clean_values.empty or self.continuous_quantile_bins <= 1:
            return np.array([], dtype=np.float64)
        quantiles = np.linspace(0.0, 1.0, self.continuous_quantile_bins + 1)[1:-1]
        edges = np.quantile(clean_values.to_numpy(), quantiles)
        return np.unique(edges.astype(np.float64))

    def _quantile_bin_codes(self, feature_name: str, values: pd.Series) -> np.ndarray:
        edges = self._quantile_edges.get(feature_name, np.array([], dtype=np.float64))
        codes = np.zeros(len(values), dtype=np.int64)
        observed = values.notna().to_numpy()
        if observed.any():
            codes[observed] = np.searchsorted(edges, values[observed].astype(float).to_numpy(), side="right") + 1
        return codes

    def _missing_reason_series(self, df: pd.DataFrame, feature_name: str, missing: np.ndarray) -> np.ndarray:
        reason_col = f"{feature_name}__missing_reason"
        not_missing = self.missing_reason_codes.get("not_missing", 0)
        unknown_missing = self.missing_reason_codes.get("unknown_missing", not_missing)
        reasons = np.where(missing, unknown_missing, not_missing).astype(np.int64)
        if reason_col not in df.columns:
            return reasons
        numeric = pd.to_numeric(df[reason_col], errors="coerce")
        parsed = numeric.to_numpy(dtype=np.float64)
        invalid = np.isnan(parsed)
        parsed[invalid] = reasons[invalid]
        return parsed.astype(np.int64)
