from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from src.tokenizer.schema import FeatureType


class FeatureTokenBuilder(nn.Module):
    """
    Build dense table-cell tokens z_ij from tokenizer channels.
    This follows the same token construction as src/name_value_transformer/model.py:

        z_ij =
            context-specific feature semantic token_ij
          + value token_ij
          + feature type token_j
          + missing reason token_ij

    Inputs:
        batch["numeric_values"]           [B, N]
        batch["continuous_bin_codes"]     [B, N]
        batch["categorical_codes"]        [B, N]
        batch["missing_reason_codes"]     [B, N]
        batch["missing_mask"]             [B, N]   # means patient-level missingness: for patient i, feature j is schema-present but the actual value is missing.

    Optional:
        batch["feature_context_codes"]    [B, N]

    Output:
        z                                [B, N, token_dim]

    Notes:
        - Patient-level missingness is encoded through missing_mask and missing_reason_codes.
    """

    def __init__(
        self,
        name_embeddings: torch.Tensor,
        feature_type_ids: torch.Tensor,
        categorical_cardinalities: list[int] | tuple[int, ...],
        feature_context_offsets: torch.Tensor | None = None,
        continuous_bin_cardinality: int = 11,
        missing_reason_cardinality: int = 10,
        token_dim: int = 128,
        dropout: float = 0.1,
        categorical_embedding_weights: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        
        super().__init__()

        if name_embeddings.ndim != 2:
            raise ValueError("name_embeddings must have shape [n_context_embeddings, name_embedding_dim].")
        if feature_type_ids.ndim != 1:
            raise ValueError("feature_type_ids must have shape [n_features].")

        n_features = int(feature_type_ids.shape[0])
        if len(categorical_cardinalities) != n_features:
            raise ValueError("categorical_cardinalities must have one entry per feature.")

        if feature_context_offsets is None:
            if int(name_embeddings.shape[0]) != n_features:
                raise ValueError("name_embeddings must have one row per feature when feature_context_offsets is not set.")
            feature_context_offsets = torch.arange(n_features, dtype=torch.long)
        else:
            feature_context_offsets = torch.as_tensor(feature_context_offsets).long()

        if feature_context_offsets.ndim != 1 or int(feature_context_offsets.shape[0]) != n_features:
            raise ValueError("feature_context_offsets must have shape [n_features].")


        self.n_features = n_features
        self.token_dim = int(token_dim)

        self.register_buffer("frozen_name_embeddings", name_embeddings.float(), persistent=True)
        self.register_buffer("feature_type_ids", feature_type_ids.long(), persistent=True)
        self.register_buffer("feature_context_offsets", feature_context_offsets.long(), persistent=True)


        category_embedding_dim = self.token_dim
        if categorical_embedding_weights:
            dims = {int(weights.shape[1]) for weights in categorical_embedding_weights.values()}
            if len(dims) != 1:
                raise ValueError("All categorical_embedding_weights tensors must have the same embedding dimension.")
            category_embedding_dim = dims.pop()


        name_dim = int(name_embeddings.shape[1])

        self.name_projection = nn.Linear(name_dim, self.token_dim)
        #self.numeric_encoder = _value_mlp(2, self.token_dim, dropout)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(2, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.numeric_film = nn.Linear(self.token_dim, 2 * self.token_dim)
        self.continuous_bin_embedding = nn.Embedding(max(int(continuous_bin_cardinality), 1), self.token_dim)
        self.type_embedding = nn.Embedding(len(FeatureType), self.token_dim)
        self.missing_embedding = nn.Embedding(max(int(missing_reason_cardinality), 1), self.token_dim)
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(max(int(cardinality), 1), category_embedding_dim) for cardinality in categorical_cardinalities]
        )
        
        if categorical_embedding_weights:
            with torch.no_grad():
                for feature_idx, weights in categorical_embedding_weights.items():
                    feature_idx = int(feature_idx)
                    if feature_idx < 0 or feature_idx >= self.n_features:
                        raise ValueError(f"categorical_embedding_weights contains invalid feature index {feature_idx}.")

                    expected_shape = self.category_embeddings[feature_idx].weight.shape
                    if tuple(weights.shape) != tuple(expected_shape):
                        raise ValueError(
                            "categorical_embedding_weights entries must match the corresponding embedding shape. "
                            f"Feature {feature_idx}: expected {tuple(expected_shape)}, got {tuple(weights.shape)}."
                        )

                    self.category_embeddings[feature_idx].weight.copy_(weights.float())
            
            # Freeze the blue lookup tables after initializing them from BGE/category-text embeddings.
            for emb in self.category_embeddings:
                emb.weight.requires_grad_(False)
        
        # The orange category_projection trainable.
        self.category_projection = (
            nn.Linear(category_embedding_dim, self.token_dim) if category_embedding_dim != self.token_dim else nn.Identity()
        )

        self.token_norm = nn.LayerNorm(self.token_dim)

    
    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        numeric_values = batch["numeric_values"]
        continuous_bin_codes = batch["continuous_bin_codes"]
        categorical_codes = batch["categorical_codes"]
        missing_reason_codes = batch["missing_reason_codes"]
        missing_mask = batch["missing_mask"]
        feature_context_codes = batch.get("feature_context_codes")

        return self._build_feature_tokens(
            numeric_values=numeric_values,
            continuous_bin_codes=continuous_bin_codes,
            categorical_codes=categorical_codes,
            missing_reason_codes=missing_reason_codes,
            missing_mask=missing_mask,
            feature_context_codes=feature_context_codes,
        )


    def _build_feature_tokens(
        self,
        *,
        numeric_values: torch.Tensor,
        continuous_bin_codes: torch.Tensor,
        categorical_codes: torch.Tensor,
        missing_reason_codes: torch.Tensor,
        missing_mask: torch.Tensor,
        feature_context_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if numeric_values.ndim != 2:
            raise ValueError("numeric_values must have shape [B, N].")

        batch_size, n_features = numeric_values.shape
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}.")

        device = numeric_values.device

        continuous_bin_codes = continuous_bin_codes.to(device)
        categorical_codes = categorical_codes.to(device)
        missing_reason_codes = missing_reason_codes.to(device)
        missing_mask = missing_mask.to(device)

        if feature_context_codes is None:
            feature_context_codes = torch.zeros(batch_size, self.n_features, dtype=torch.long, device=device)
        #   feature_context_codes = torch.zeros_like(categorical_codes, dtype=torch.long)
        else:
            feature_context_codes = feature_context_codes.to(device=device, dtype=torch.long)
            if feature_context_codes.shape != (batch_size, self.n_features):
                raise ValueError(
                    "feature_context_codes must have shape "
                    f"[{batch_size}, {self.n_features}], got {tuple(feature_context_codes.shape)}."
                )

        type_ids = self.feature_type_ids.to(device)
        missing_ids = missing_reason_codes.clamp(min=0, max=self.missing_embedding.num_embeddings - 1).long()

        projected_contexts = self.name_projection(self.frozen_name_embeddings.to(device))
        context_ids = self.feature_context_offsets.to(device).unsqueeze(0) + feature_context_codes.long()
        context_ids = context_ids.clamp(min=0, max=projected_contexts.shape[0] - 1)
        name_tokens = projected_contexts[context_ids]
        type_tokens = self.type_embedding(type_ids).unsqueeze(0).expand(batch_size, -1, -1)
        missing_tokens = self.missing_embedding(missing_ids)

        value_tokens = torch.zeros(batch_size, self.n_features, self.token_dim, device=device, dtype=name_tokens.dtype)
        numerical = type_ids == int(FeatureType.NUMERICAL)
        categorical = type_ids == int(FeatureType.CATEGORICAL)

        
        if numerical.any():
            num_input = torch.stack([numeric_values[:, numerical], missing_mask[:, numerical].float()], dim=-1)
            bin_codes = continuous_bin_codes[:, numerical].clamp(
                min=0,
                max=self.continuous_bin_embedding.num_embeddings - 1,
            ).long()
            numeric_base = self.numeric_encoder(num_input)
            scale, shift = self.numeric_film(name_tokens[:, numerical, :]).chunk(2, dim=-1)
            numeric_value = numeric_base * (1.0 + torch.tanh(scale)) + shift

            value_tokens[:, numerical] = numeric_value + self.continuous_bin_embedding(bin_codes)

        for feature_idx in torch.where(categorical)[0].tolist():
            codes = categorical_codes[:, feature_idx].clamp(min=0, max=self.category_embeddings[feature_idx].num_embeddings - 1).long()
            value_tokens[:, feature_idx] = self.category_projection(self.category_embeddings[feature_idx](codes))


        tokens = name_tokens + value_tokens + type_tokens + missing_tokens
        return self.token_norm(tokens)
    

    def metadata_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        feature_context_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build metadata-only query tokens q_ij for the reconstruction decoder.

        q_ij =
            context-specific feature semantic token_ij
          + feature type token_j

        """
        if feature_context_codes is None:
            feature_context_codes = torch.zeros(
                batch_size,
                self.n_features,
                dtype=torch.long,
                device=device,
            )
        else:
            feature_context_codes = feature_context_codes.to(device=device, dtype=torch.long)
            if feature_context_codes.shape != (batch_size, self.n_features):
                raise ValueError(
                    "feature_context_codes must have shape "
                    f"[{batch_size}, {self.n_features}], got {tuple(feature_context_codes.shape)}."
                )

        projected_contexts = self.name_projection(self.frozen_name_embeddings.to(device))
        context_ids = self.feature_context_offsets.to(device).unsqueeze(0) + feature_context_codes
        context_ids = context_ids.clamp(min=0, max=projected_contexts.shape[0] - 1)

        name_tokens = projected_contexts[context_ids]

        type_ids = self.feature_type_ids.to(device)
        type_tokens = self.type_embedding(type_ids).unsqueeze(0).expand(
            batch_size,
            self.n_features,
            self.token_dim,
        )

        return self.token_norm(name_tokens + type_tokens)