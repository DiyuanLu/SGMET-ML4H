from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from src.clinical_cluster_experts.model import DownstreamPredictionHead, TransformerSetEncoder
from src.clinical_cluster_experts.token_builder import FeatureTokenBuilder
from src.tokenizer.schema import FeatureType


DEFAULT_D_MODEL = 160
DEFAULT_N_HEADS = 4
DEFAULT_N_LAYERS = 4
DEFAULT_DIM_FEEDFORWARD = 640
DEFAULT_DROPOUT = 0.1
CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class TransformerOutput:
    """Outputs from the flat all-feature patient encoder."""

    patient_embedding: torch.Tensor
    token_embeddings: torch.Tensor
    feature_available_mask: torch.Tensor


class FeatureTokenTransformer(nn.Module):
    """One flat Transformer over all selected name-value feature tokens.

    Every selected feature is tokenized by one shared :class:`FeatureTokenBuilder`
    and passed to the same Transformer in a single sequence.  There are no
    cluster assignments, expert branches, group tokens, cluster embeddings, or
    fusion Transformer.  Like the clinical expert blocks, the encoder is
    permutation invariant because it adds no positional encoding.
    """

    def __init__(
        self,
        *,
        name_embeddings: torch.Tensor,
        feature_type_ids: torch.Tensor,
        categorical_cardinalities: list[int] | tuple[int, ...],
        feature_context_offsets: torch.Tensor | None = None,
        continuous_bin_cardinality: int = 11,
        missing_reason_cardinality: int = 10,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        dim_feedforward: int | None = DEFAULT_DIM_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
        categorical_embedding_weights: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if dim_feedforward is None:
            dim_feedforward = 4 * int(d_model)

        self.n_features = int(feature_type_ids.numel())
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)

        self.token_builder = FeatureTokenBuilder(
            name_embeddings=name_embeddings,
            feature_type_ids=feature_type_ids,
            categorical_cardinalities=categorical_cardinalities,
            feature_context_offsets=feature_context_offsets,
            continuous_bin_cardinality=continuous_bin_cardinality,
            missing_reason_cardinality=missing_reason_cardinality,
            token_dim=self.d_model,
            dropout=self.dropout,
            categorical_embedding_weights=categorical_embedding_weights,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.encoder = TransformerSetEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        nn.init.normal_(self.cls_token, std=0.02)

    @property
    def feature_type_ids(self) -> torch.Tensor:
        return self.token_builder.feature_type_ids

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        feature_available_mask: torch.Tensor | None = None,
    ) -> TransformerOutput:
        if "numeric_values" not in batch:
            raise KeyError("batch must contain 'numeric_values'.")
        numeric_values = batch["numeric_values"]
        if numeric_values.ndim != 2:
            raise ValueError("batch['numeric_values'] must have shape [B, N].")
        batch_size, n_features = numeric_values.shape
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}.")

        device = numeric_values.device
        if feature_available_mask is None:
            if "observed_mask" in batch:
                feature_available_mask = batch["observed_mask"]
            else:
                feature_available_mask = torch.ones(
                    batch_size,
                    n_features,
                    dtype=torch.bool,
                    device=device,
                )
        feature_available_mask = feature_available_mask.to(device=device, dtype=torch.bool)
        if feature_available_mask.shape != (batch_size, n_features):
            raise ValueError(
                "feature_available_mask must have shape "
                f"{(batch_size, n_features)}, got {tuple(feature_available_mask.shape)}."
            )

        feature_tokens = self.token_builder(batch)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, feature_tokens], dim=1)
        cls_available = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        token_available = torch.cat([cls_available, feature_available_mask], dim=1)
        hidden = self.encoder(tokens, key_padding_mask=~token_available)
        return TransformerOutput(
            patient_embedding=hidden[:, 0],
            token_embeddings=hidden[:, 1:],
            feature_available_mask=feature_available_mask,
        )


BinaryPredictionHead = DownstreamPredictionHead


def parameter_counts(
    encoder: FeatureTokenTransformer,
    head: nn.Module | None = None,
) -> dict[str, int]:
    """Count stored and forward-used parameters like clinical inference does.

    ``FeatureTokenBuilder`` keeps one category table per feature for stable
    indexing, including cardinality-one tables for numerical features.  Those
    numerical tables are stored but never read by the forward pass, so active
    counts exclude them to stay comparable to clinical_cluster_experts.
    """

    def counts(module: nn.Module | None) -> tuple[int, int]:
        if module is None:
            return 0, 0
        total = sum(parameter.numel() for parameter in module.parameters())
        trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        return int(total), int(trainable)

    stored_total, stored_trainable = counts(encoder)
    head_total, head_trainable = counts(head)
    unused_total = 0
    unused_trainable = 0
    type_ids = encoder.token_builder.feature_type_ids.detach().cpu().long()
    for feature_idx, embedding in enumerate(encoder.token_builder.category_embeddings):
        if int(type_ids[feature_idx]) == int(FeatureType.CATEGORICAL):
            continue
        total, trainable = counts(embedding)
        unused_total += total
        unused_trainable += trainable

    active_total = stored_total - unused_total + head_total
    active_trainable = stored_trainable - unused_trainable + head_trainable
    return {
        "active_total": int(active_total),
        "active_trainable": int(active_trainable),
        "active_frozen": int(active_total - active_trainable),
        "stored_total": int(stored_total + head_total),
        "stored_trainable": int(stored_trainable + head_trainable),
        "stored_frozen": int(stored_total + head_total - stored_trainable - head_trainable),
        "stored_unused": int(unused_total),
        "encoder_stored": int(stored_total),
        "head": int(head_total),
    }
