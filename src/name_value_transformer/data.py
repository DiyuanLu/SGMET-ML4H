from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from src.clinical_cluster_experts.utils import load_embeddings


FEATURE_BATCH_KEYS = (
    "numeric_values",
    "continuous_bin_codes",
    "categorical_codes",
    "missing_reason_codes",
    "missing_mask",
    "observed_mask",
    "feature_context_codes",
)


@dataclass(frozen=True)
class LeidenFeatureView:
    """Fixed input view used for the flat comparison model."""

    global_indices: torch.Tensor
    feature_names: tuple[str, ...]
    ignored_cluster_ids: tuple[int, ...]
    n_global_features: int

    @property
    def n_features(self) -> int:
        return int(self.global_indices.numel())


@dataclass(frozen=True)
class FlatModelInputs:
    feature_view: LeidenFeatureView
    name_embeddings: torch.Tensor
    feature_context_offsets: torch.Tensor | None
    feature_type_ids: torch.Tensor
    categorical_cardinalities: tuple[int, ...]
    continuous_bin_cardinality: int
    missing_reason_cardinality: int
    categorical_embedding_weights: dict[int, torch.Tensor] | None

    def model_kwargs(self, *, missing_reason_cardinality: int | None = None) -> dict[str, Any]:
        return {
            "name_embeddings": self.name_embeddings,
            "feature_context_offsets": self.feature_context_offsets,
            "feature_type_ids": self.feature_type_ids,
            "categorical_cardinalities": list(self.categorical_cardinalities),
            "continuous_bin_cardinality": self.continuous_bin_cardinality,
            "missing_reason_cardinality": int(
                self.missing_reason_cardinality
                if missing_reason_cardinality is None
                else missing_reason_cardinality
            ),
            "categorical_embedding_weights": self.categorical_embedding_weights,
        }


def load_leiden_feature_view(
    cluster_csv: str | Path,
    feature_names: list[str] | tuple[str, ...],
    *,
    ignored_cluster_ids: tuple[int, ...] = (7,),
) -> LeidenFeatureView:
    frame = pd.read_csv(cluster_csv)
    required = {"feature_name", "cluster_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Leiden feature CSV is missing columns: {sorted(missing)}")
    if frame["feature_name"].duplicated().any():
        duplicates = frame.loc[frame["feature_name"].duplicated(), "feature_name"].tolist()
        raise ValueError(f"Duplicate feature names in Leiden CSV: {duplicates[:10]}")
    mapping = dict(zip(frame["feature_name"].astype(str), frame["cluster_id"].astype(int)))
    absent = [name for name in feature_names if name not in mapping]
    if absent:
        raise ValueError(f"Leiden feature CSV is missing tokenizer features: {absent[:10]}")
    ignored = {int(value) for value in ignored_cluster_ids}
    indices = [idx for idx, name in enumerate(feature_names) if int(mapping[name]) not in ignored]
    selected_names = tuple(str(feature_names[idx]) for idx in indices)
    if not indices:
        raise ValueError("The Leiden feature view selected no features.")
    return LeidenFeatureView(
        global_indices=torch.tensor(indices, dtype=torch.long),
        feature_names=selected_names,
        ignored_cluster_ids=tuple(sorted(ignored)),
        n_global_features=len(feature_names),
    )


def _slice_context_embeddings(
    name_embeddings: torch.Tensor,
    feature_context_offsets: torch.Tensor | None,
    feature_indices: torch.Tensor,
    n_global_features: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if feature_context_offsets is None:
        return name_embeddings.index_select(0, feature_indices).clone(), None

    offsets = torch.as_tensor(feature_context_offsets).long()
    blocks: list[torch.Tensor] = []
    local_offsets: list[int] = []
    cursor = 0
    for global_idx in feature_indices.tolist():
        start = int(offsets[global_idx])
        end = (
            int(offsets[global_idx + 1])
            if global_idx + 1 < n_global_features
            else int(name_embeddings.shape[0])
        )
        if end <= start:
            raise ValueError(f"Invalid feature-context range {start}:{end} for feature {global_idx}.")
        local_offsets.append(cursor)
        blocks.append(name_embeddings[start:end])
        cursor += end - start
    return torch.cat(blocks, dim=0), torch.tensor(local_offsets, dtype=torch.long)


def build_flat_model_inputs(
    token_dir: str | Path,
    cluster_csv: str | Path,
    metadata: Mapping[str, Any],
    *,
    ignored_cluster_ids: tuple[int, ...] = (7,),
) -> FlatModelInputs:
    feature_names = list(metadata["feature_names"])
    view = load_leiden_feature_view(
        cluster_csv,
        feature_names,
        ignored_cluster_ids=ignored_cluster_ids,
    )
    name_embeddings, context_offsets, category_weights = load_embeddings(token_dir, metadata)
    local_names, local_offsets = _slice_context_embeddings(
        name_embeddings,
        context_offsets,
        view.global_indices,
        len(feature_names),
    )
    global_types = torch.as_tensor(metadata["feature_type_ids"]).long()
    global_cards = [int(value) for value in metadata["categorical_cardinalities"]]
    local_cards = tuple(global_cards[idx] for idx in view.global_indices.tolist())
    local_category_weights = None
    if category_weights:
        local_category_weights = {
            local_idx: category_weights[global_idx]
            for local_idx, global_idx in enumerate(view.global_indices.tolist())
            if global_idx in category_weights
        }
    return FlatModelInputs(
        feature_view=view,
        name_embeddings=local_names,
        feature_context_offsets=local_offsets,
        feature_type_ids=global_types.index_select(0, view.global_indices),
        categorical_cardinalities=local_cards,
        continuous_bin_cardinality=int(metadata["continuous_bin_cardinality"]),
        missing_reason_cardinality=int(metadata["missing_reason_cardinality"]),
        categorical_embedding_weights=local_category_weights,
    )


def select_batch_features(
    batch: Mapping[str, Any],
    feature_view: LeidenFeatureView,
) -> dict[str, Any]:
    """Select the fixed flat feature view from a collated global batch."""

    selected: dict[str, Any] = dict(batch)
    for key in FEATURE_BATCH_KEYS:
        value = batch.get(key)
        if not torch.is_tensor(value):
            continue
        if value.ndim != 2:
            raise ValueError(f"batch[{key!r}] must have shape [B, N], got {tuple(value.shape)}.")
        index = feature_view.global_indices.to(value.device)
        selected[key] = value.index_select(1, index)
    return selected


def validate_checkpoint_feature_view(
    checkpoint: Mapping[str, Any],
    feature_view: LeidenFeatureView,
) -> None:
    saved_names = tuple(str(value) for value in checkpoint.get("feature_names", ()))
    if saved_names != feature_view.feature_names:
        raise ValueError("Checkpoint feature names/order do not match the selected Leiden feature view.")
    saved_indices = torch.as_tensor(checkpoint.get("feature_indices", [])).long()
    if not torch.equal(saved_indices.cpu(), feature_view.global_indices.cpu()):
        raise ValueError("Checkpoint feature indices do not match the selected Leiden feature view.")
