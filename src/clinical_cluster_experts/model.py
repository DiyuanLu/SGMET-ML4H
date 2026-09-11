from __future__ import annotations
from typing import Mapping, Optional
import torch
from torch import nn
from src.clinical_cluster_experts.token_builder import FeatureTokenBuilder

TOKEN_BATCH_KEYS = (
    "numeric_values",
    "continuous_bin_codes",
    "categorical_codes",
    "missing_reason_codes",
    "missing_mask",
)

def _slice_context_embeddings_for_branch(
    *,
    name_embeddings: torch.Tensor,
    feature_context_offsets: torch.Tensor | None,
    feature_indices: torch.Tensor,
    n_global_features: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if feature_context_offsets is None:
        return name_embeddings[feature_indices].clone(), None

    blocks = []
    local_offsets = []
    cursor = 0

    offsets = feature_context_offsets.long()

    for global_idx in feature_indices.tolist():
        global_idx = int(global_idx)
        start = int(offsets[global_idx].item())

        if global_idx + 1 < n_global_features:
            end = int(offsets[global_idx + 1].item())
        else:
            end = int(name_embeddings.shape[0])

        if end <= start:
            raise ValueError(f"Invalid context offset range for feature {global_idx}: {start}:{end}")

        local_offsets.append(cursor)
        blocks.append(name_embeddings[start:end])
        cursor += end - start

    local_name_embeddings = torch.cat(blocks, dim=0)
    local_feature_context_offsets = torch.tensor(local_offsets, dtype=torch.long)

    return local_name_embeddings, local_feature_context_offsets


# Generic Transformer encoder over a set/sequence of tokens.
class TransformerSetEncoder(nn.Module):
    """Generic Transformer encoder without positional encodings.
   
    When requested, the final layer's self-attention is replayed once with
    attention dropout disabled. This exposes differentiable, head-averaged
    attention weights without changing the normal encoder output or checkpoint
    parameters. The extra replay is used only by attention-based auxiliary
    losses and optional diagnostics.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if int(n_layers) < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True, # means input shape is [B, L, d_model]
            norm_first=True, # pre-norm architecture, which is more stable for deeper transformers
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        *,
        return_last_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode 'tokens' and optionally return final-layer attention.

        Args:
            tokens: [B, L, d_model].
            key_padding_mask: optional bool mask [B, L] where True means ignored by attention.
            return_last_attention: if True, also return differentiable, head-averaged final-layer attention weights [B, L, L].
                The replay uses the exact final-layer inputs from this forward pass and disables attention dropout only for the returned map.
        """
        if not return_last_attention:
            encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
            return self.norm(encoded)

        # Capture the exact query/key/value tensors and masks received by the final Transformer layer. 
        # The hook also disables PyTorch's fused encoder fast path, ensuring the self-attention module is invoked normally.
        captured: dict[str, object] = {}
        self_attn = self.encoder.layers[-1].self_attn

        def _capture_attention_inputs(module, args, kwargs):
            captured["query"] = args[0]
            captured["key"] = args[1]
            captured["value"] = args[2]
            captured["key_padding_mask"] = kwargs.get("key_padding_mask")
            captured["attn_mask"] = kwargs.get("attn_mask")
            captured["is_causal"] = bool(kwargs.get("is_causal", False))

        handle = self_attn.register_forward_pre_hook( _capture_attention_inputs, with_kwargs=True)
        try:
            encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        finally:
            handle.remove()

        if "query" not in captured:
            raise RuntimeError("Could not capture final-layer attention inputs.")

        # Replay only the attention operation. The captured tensors remain in the
        # autograd graph, so attention-based auxiliary losses can update the expert.
        # Disable attention dropout during the replay to regularize the underlying
        # attention distribution rather than a second random dropout sample.
        original_dropout = self_attn.dropout
        self_attn.dropout = 0.0
        try:
            _, attention = self_attn(
                captured["query"],
                captured["key"],
                captured["value"],
                key_padding_mask=captured["key_padding_mask"],
                need_weights=True,
                attn_mask=captured["attn_mask"],
                average_attn_weights=True,
                is_causal=bool(captured["is_causal"]),
            )
        finally:
            self_attn.dropout = original_dropout

        if attention is None:
            raise RuntimeError("Final-layer attention replay returned no weights.")
        return self.norm(encoded), attention


class ClusterExpertEncoder(nn.Module):
    """ Transformer expert for one clinical feature group.

    Learns: 
        - A group-specific [CLS] token (or M>1 summary tokens) that attends to the feature tokens and summarizes the group.
        - Transformer weights that encode the feature tokens and CLS token(s) into a group representation.

    - num_summary_tokens=1 exactly preserves the original SGMET interface and state-dict shape. 
    - With num_summary_tokens M > 1, the expert input is [CLS_1, ..., CLS_M, z_1, ..., z_A] 
        and the first M outputs become the group representations.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        num_summary_tokens: int = 1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_summary_tokens = int(num_summary_tokens)
        if self.num_summary_tokens < 1:
            raise ValueError(f"num_summary_tokens must be >= 1, got {num_summary_tokens}.")

        # One parameter tensor is sampled element-wise, so every summary slot is independently initialized. 
        # For M=1 this retains the original cls_token name and [1, 1, d] checkpoint shape.
        self.cls_token = nn.Parameter(torch.empty(1, self.num_summary_tokens, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.encoder = TransformerSetEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def _empty_summary(self, feature_tokens: torch.Tensor) -> torch.Tensor:
        B = int(feature_tokens.shape[0])
        shape = (
            (B, self.d_model)
            if self.num_summary_tokens == 1
            else (B, self.num_summary_tokens, self.d_model)
        )
        return torch.zeros(
            *shape,
            device=feature_tokens.device,
            dtype=feature_tokens.dtype,
        )

    def forward(
        self,
        feature_tokens: torch.Tensor,
        feature_available_mask: Optional[torch.Tensor] = None,
        *,
        return_attention_overlap: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Summarize one feature group.
        Input: 
            feature_tokens: [B, A_c, d_model] feature tokens for this group.
            feature_available_mask: optional [B, A_c] bool mask for transformer attention. 
        Returns:
            feature-group representation token(s) of shape [B, M, d_model].
            If return_attention_overlap=True, also returns pairwise CLS-to-feature attention overlap [B, M, M]. 
                Entries are NaN for patients with fewer than two available features, where overlap is not informative.
        """
        if feature_tokens.ndim != 3:
            raise ValueError(f"feature_tokens must have shape [B, A_c, d_model], got {tuple(feature_tokens.shape)}.")

        B, A, d = feature_tokens.shape
        device = feature_tokens.device
        if d != self.d_model:
            raise ValueError(f"feature_tokens last dimension is {d}, expected {self.d_model}.")

        if A == 0:
            # Handle empty feature group: No features in this group, so return a zero summary and NaN overlap.
            summary = self._empty_summary(feature_tokens)
            if return_attention_overlap:
                overlap = torch.full(
                    (B, self.num_summary_tokens, self.num_summary_tokens),
                    float("nan"),
                    device=device,
                    dtype=feature_tokens.dtype,
                )
                feature_distribution = torch.zeros(
                    B, self.num_summary_tokens, 0, device=device, dtype=feature_tokens.dtype
                )
                return summary, overlap, feature_distribution
            return summary

        if feature_available_mask is None:
            # If no mask is provided, assume all features are available for attention.
            feature_available_mask = torch.ones(B, A, dtype=torch.bool, device=device)
        else:
            # Convert mask to bool and ensure it has the correct shape.
            feature_available_mask = feature_available_mask.to(device=device, dtype=torch.bool)
            if feature_available_mask.shape != (B, A):
                raise ValueError(f"feature_available_mask must have shape {(B, A)}, got {tuple(feature_available_mask.shape)}.")

        cls = self.cls_token.expand(B, -1, -1)
        # CLS token is always available
        cls_available = torch.ones(B, self.num_summary_tokens, dtype=torch.bool, device=device) 
        # tokens = [CLS_1, ..., CLS_M, z_1, ..., z_A] where M=num_summary_tokens and A=number of features in this group
        tokens = torch.cat([cls, feature_tokens], dim=1)
        token_available = torch.cat([cls_available, feature_available_mask], dim=1)
        # PyTorch convention: key_padding_mask=True means ignore this token.
        key_padding_mask = ~token_available

        if return_attention_overlap:
            encoded, attention = self.encoder(tokens, key_padding_mask=key_padding_mask, return_last_attention=True)
        else:
            encoded = self.encoder(tokens, key_padding_mask=key_padding_mask)
            attention = None

        summaries = encoded[:, : self.num_summary_tokens]
        summary_output = summaries[:, 0] if self.num_summary_tokens == 1 else summaries

        if not return_attention_overlap:
            return summary_output

        assert attention is not None

        # --- For return_attention_overlap = True ---
        # Keep only CLS-query -> feature-key attention, remove unavailable features, and renormalize over feature positions. 
        # Attention is already averaged over heads by TransformerSetEncoder.
        feature_attention = attention[:, : self.num_summary_tokens, self.num_summary_tokens :]
        feature_attention = feature_attention.masked_fill(~feature_available_mask.unsqueeze(1), 0.0)
        mass = feature_attention.sum(dim=-1, keepdim=True)
        feature_distribution = feature_attention / mass.clamp_min(1e-12)
        l2 = torch.linalg.vector_norm(feature_distribution, ord=2, dim=-1, keepdim=True)
        normalized = feature_distribution / l2.clamp_min(1e-12)
        overlap = torch.matmul(normalized, normalized.transpose(1, 2))

        valid = (
            feature_available_mask.sum(dim=1).ge(2)
            & mass.squeeze(-1).gt(1e-12).all(dim=1)
        )
        overlap = overlap.masked_fill(~valid[:, None, None], float("nan"))
        return summary_output, overlap, feature_distribution

# One complete expert branch: local token builder + expert encoder.
class ClusterBranch(nn.Module):
    """One complete branch: local token builder + expert encoder."""

    def __init__(
        self,
        *,
        feature_indices: torch.Tensor,
        name_embeddings: torch.Tensor,
        feature_type_ids: torch.Tensor,
        categorical_cardinalities: list[int],
        continuous_bin_cardinality: int,
        missing_reason_cardinality: int,
        feature_context_offsets: torch.Tensor | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        categorical_embedding_weights: Mapping[int, torch.Tensor] | None = None,
        num_summary_tokens: int = 1,
    ):
        super().__init__()
        if feature_indices.ndim != 1:
            raise ValueError(f"feature_indices must have shape [A_c], got {tuple(feature_indices.shape)}.")

        feature_indices = feature_indices.long()
        self.register_buffer("feature_indices", feature_indices, persistent=True)
        self.d_model = int(d_model)
        self.num_summary_tokens = int(num_summary_tokens)

        self.expert = ClusterExpertEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_summary_tokens=self.num_summary_tokens,
        )

        # Empty feature group: no features in this branch, so no token builder is needed.
        if feature_indices.numel() == 0:
            self.token_builder = None
            return

        global_indices = [int(idx) for idx in feature_indices.tolist()]
        local_name_embeddings, local_feature_context_offsets = (
            _slice_context_embeddings_for_branch(
                name_embeddings=name_embeddings,
                feature_context_offsets=feature_context_offsets,
                feature_indices=feature_indices,
                n_global_features=int(feature_type_ids.shape[0]),
            )
        )
        local_feature_type_ids = feature_type_ids[feature_indices].clone()
        local_categorical_cardinalities = [
            int(categorical_cardinalities[global_idx])
            for global_idx in global_indices
        ]

        local_category_weights: dict[int, torch.Tensor] = {}
        if categorical_embedding_weights is not None:
            for local_idx, global_idx in enumerate(global_indices):
                if global_idx in categorical_embedding_weights:
                    local_category_weights[local_idx] = (
                        categorical_embedding_weights[global_idx]
                    )

        self.token_builder = FeatureTokenBuilder(
            name_embeddings=local_name_embeddings,
            feature_type_ids=local_feature_type_ids,
            categorical_cardinalities=local_categorical_cardinalities,
            continuous_bin_cardinality=continuous_bin_cardinality,
            missing_reason_cardinality=missing_reason_cardinality,
            feature_context_offsets=local_feature_context_offsets,
            token_dim=d_model,
            dropout=dropout,
            categorical_embedding_weights=(
                local_category_weights if local_category_weights else None
            ),
        )

    def _zero_group_tokens(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (
            (batch_size, self.d_model)
            if self.num_summary_tokens == 1
            else (batch_size, self.num_summary_tokens, self.d_model)
        )
        return torch.zeros(*shape, device=device, dtype=dtype)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        feature_available_mask: torch.Tensor | None = None,
        *,
        return_attention_overlap: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Input: 
            batch: dictionary containing feature data for this branch. Must contain keys in TOKEN_BATCH_KEYS.
            feature_available_mask: optional [B, A_c] bool mask for transformer attention.
        Returns:
            group_tokens: [B, M, d_model] or [B, d_model] if num_summary_tokens=1.
            cluster_available: [B] bool mask indicating if any features were available for this branch.
        """
        
        missing_keys = [key for key in TOKEN_BATCH_KEYS if key not in batch]
        if missing_keys:
            raise KeyError(f"Batch is missing required keys {missing_keys}. Available keys: {list(batch.keys())}")

        B = int(batch["numeric_values"].shape[0])
        device = batch["numeric_values"].device
        dtype = batch["numeric_values"].dtype
        idx = self.feature_indices.to(device)

        if idx.numel() == 0:
            group_tokens = self._zero_group_tokens(B, device=device, dtype=dtype)
            cluster_available = torch.zeros(B, dtype=torch.bool, device=device)
            if return_attention_overlap:
                overlap = torch.full(
                    (B, self.num_summary_tokens, self.num_summary_tokens),
                    float("nan"),
                    device=device,
                    dtype=dtype,
                )
                feature_attention = torch.zeros(
                    B, self.num_summary_tokens, 0, device=device, dtype=dtype
                )
                local_available_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
                return group_tokens, cluster_available, overlap, feature_attention, local_available_mask
            return group_tokens, cluster_available

        local_batch = {
            key: batch[key].index_select(dim=1, index=idx)
            for key in TOKEN_BATCH_KEYS
        }
        if "feature_context_codes" in batch:
            local_batch["feature_context_codes"] = batch["feature_context_codes"].index_select(dim=1, index=idx)

        assert self.token_builder is not None
        z_c = self.token_builder(local_batch)

        if feature_available_mask is None:
            local_available_mask = torch.ones(B, idx.numel(), dtype=torch.bool, device=device)
        else:
            local_available_mask = feature_available_mask.index_select(dim=1, index=idx).to(device=device, dtype=torch.bool)

        cluster_available = local_available_mask.any(dim=1)
        if return_attention_overlap:
            group_tokens, attention_overlap, feature_attention = self.expert(
                feature_tokens=z_c,
                feature_available_mask=local_available_mask,
                return_attention_overlap=True,
            )
        else:
            group_tokens = self.expert(
                feature_tokens=z_c,
                feature_available_mask=local_available_mask,
            )
            attention_overlap = None

        availability_shape = (B,) + (1,) * (group_tokens.ndim - 1)
        group_tokens = torch.where(
            cluster_available.view(availability_shape),
            group_tokens,
            torch.zeros_like(group_tokens),
        )

        if return_attention_overlap:
            assert attention_overlap is not None
            attention_overlap = attention_overlap.masked_fill(~cluster_available[:, None, None], float("nan"))
            return group_tokens, cluster_available, attention_overlap, feature_attention, local_available_mask
        return group_tokens, cluster_available

# Bank of all expert branches.
class ExpertBank(nn.Module):
    """Bank of all expert branches."""

    def __init__(
        self,
        *,
        cluster_assignments: torch.Tensor,
        name_embeddings: torch.Tensor,
        feature_type_ids: torch.Tensor,
        categorical_cardinalities: list[int],
        continuous_bin_cardinality: int,
        missing_reason_cardinality: int,
        feature_context_offsets: torch.Tensor | None = None,
        num_clusters: int | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        categorical_embedding_weights: Mapping[int, torch.Tensor] | None = None,
        num_summary_tokens: int = 1,
    ):
        super().__init__()
        if cluster_assignments.ndim != 1:
            raise ValueError(f"cluster_assignments must have shape [N], got {tuple(cluster_assignments.shape)}.")
        cluster_assignments = cluster_assignments.long()
        n_features = int(cluster_assignments.numel())

        if name_embeddings.ndim != 2:
            raise ValueError("name_embeddings must have shape [n_context_embeddings, name_embedding_dim].")
        if feature_context_offsets is not None:
            feature_context_offsets = feature_context_offsets.long()
            if (feature_context_offsets.ndim != 1
                or int(feature_context_offsets.shape[0]) != n_features
            ):
                raise ValueError("feature_context_offsets must have shape [N].")
            if int(feature_context_offsets.min().item()) < 0:
                raise ValueError("feature_context_offsets must be non-negative.")
            if int(feature_context_offsets.max().item()) >= int(name_embeddings.shape[0]):
                raise ValueError("feature_context_offsets points outside name_embeddings.")
            if not torch.all(feature_context_offsets[1:] >= feature_context_offsets[:-1]):
                raise ValueError("feature_context_offsets must be sorted/non-decreasing.")
        if feature_type_ids.ndim != 1 or int(feature_type_ids.shape[0]) != n_features:
            raise ValueError("feature_type_ids must have shape [N] and match cluster_assignments.")
        if len(categorical_cardinalities) != n_features:
            raise ValueError("categorical_cardinalities must have one entry per feature.")

        if num_clusters is None:
            num_clusters = int(cluster_assignments.max().item()) + 1
        if int(cluster_assignments.min().item()) < 0:
            raise ValueError("cluster_assignments must be non-negative.")
        if int(cluster_assignments.max().item()) >= int(num_clusters):
            raise ValueError(
                f"cluster_assignments contains cluster id "
                f"{int(cluster_assignments.max().item())}, but num_clusters={num_clusters}."
            )

        self.n_features = n_features
        self.n_clusters = int(num_clusters)
        self.d_model = int(d_model)
        self.num_summary_tokens = int(num_summary_tokens)
        if self.num_summary_tokens < 1:
            raise ValueError("num_summary_tokens must be >= 1.")

        self.register_buffer("cluster_assignments", cluster_assignments, persistent=True)
        self.feature_indices = [
            torch.where(cluster_assignments == cluster_id)[0]
            for cluster_id in range(self.n_clusters)
        ]

        cluster_sizes = [int(idx.numel()) for idx in self.feature_indices]
        self.cluster_sizes = cluster_sizes
        self.empty_clusters = [
            cluster_id
            for cluster_id, size in enumerate(cluster_sizes)
            if size == 0
        ]
        self.n_non_empty_clusters = sum(size > 0 for size in cluster_sizes)

        print(f"[INFO] ExpertBank: {self.n_non_empty_clusters}/{self.n_clusters} clusters have schema features.")
        print(f"[INFO] ExpertBank: empty clusters = {self.empty_clusters}")
        print(f"[INFO] ExpertBank: summary tokens per expert = {self.num_summary_tokens}")
        print("[INFO] ExpertBank cluster sizes:")
        for cluster_id, size in enumerate(cluster_sizes):
            print(f"  cluster {cluster_id}: {size} features")

        self.branches = nn.ModuleList(
            [
                ClusterBranch(
                    feature_indices=self.feature_indices[cluster_id],
                    name_embeddings=name_embeddings,
                    feature_type_ids=feature_type_ids,
                    categorical_cardinalities=categorical_cardinalities,
                    continuous_bin_cardinality=continuous_bin_cardinality,
                    missing_reason_cardinality=missing_reason_cardinality,
                    feature_context_offsets=feature_context_offsets,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    categorical_embedding_weights=categorical_embedding_weights,
                    num_summary_tokens=self.num_summary_tokens,
                )
                for cluster_id in range(self.n_clusters)
            ]
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        feature_available_mask: torch.Tensor | None = None,
        *,
        return_attention_overlap: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]
    ]:
        """
        Input: 
            batch: dictionary containing feature data for all clusters. Must contain keys in TOKEN_BATCH_KEYS.
            feature_available_mask: optional [B, N] bool mask for transformer attention, where N is the total number of features.
        Returns:
            group_tokens: [B, K, M, d_model] or [B, K, d_model] if num_summary_tokens=1.
            cluster_available: [B, K] bool mask indicating if any features were available for each cluster.
            If return_attention_overlap=True, also returns [B, K, M, M] pairwise CLS-to-feature attention overlap for each cluster. 
                Entries are NaN for patients with fewer than two available features in a cluster.
        """
        
        if "numeric_values" not in batch:
            raise KeyError("batch must contain 'numeric_values'.")
        numeric_values = batch["numeric_values"]
        if numeric_values.ndim != 2:
            raise ValueError(f"batch['numeric_values'] must have shape [B, N], got {tuple(numeric_values.shape)}.")

        B, N = numeric_values.shape
        device = numeric_values.device
        if N != self.n_features:
            raise ValueError(f"Input batch has {N} features, but ExpertBank was initialized with {self.n_features} features.")
        if feature_available_mask is not None:
            feature_available_mask = feature_available_mask.to(device=device, dtype=torch.bool)
            if feature_available_mask.shape != (B, N):
                raise ValueError(f"feature_available_mask must have shape {(B, N)}, got {tuple(feature_available_mask.shape)}.")

        group_tokens = []
        cluster_available = []
        attention_overlaps = []
        feature_attention_by_cluster: dict[int, torch.Tensor] = {}
        feature_mask_by_cluster: dict[int, torch.Tensor] = {}
        for cluster_id, branch in enumerate(self.branches):
            if return_attention_overlap:
                token_c, available_c, overlap_c, feature_attention_c, feature_mask_c = branch(
                    batch=batch,
                    feature_available_mask=feature_available_mask,
                    return_attention_overlap=True,
                )
                attention_overlaps.append(overlap_c)
                feature_attention_by_cluster[cluster_id] = feature_attention_c
                feature_mask_by_cluster[cluster_id] = feature_mask_c
            else:
                token_c, available_c = branch(
                    batch=batch,
                    feature_available_mask=feature_available_mask,
                )
            group_tokens.append(token_c)
            cluster_available.append(available_c)

        group_tokens_stacked = torch.stack(group_tokens, dim=1)
        cluster_available_mask = torch.stack(cluster_available, dim=1)
        if return_attention_overlap:
            return (
                group_tokens_stacked,
                cluster_available_mask,
                torch.stack(attention_overlaps, dim=1),
                feature_attention_by_cluster,
                feature_mask_by_cluster,
            )
        return group_tokens_stacked, cluster_available_mask


class PatientFusionTransformer(nn.Module):
    """Fuse all expert summary tokens into one patient embedding."""

    def __init__(
        self,
        *,
        num_clusters: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        use_cluster_embedding: bool = True,
        num_summary_tokens: int = 1,
    ):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.d_model = int(d_model)
        self.use_cluster_embedding = bool(use_cluster_embedding)
        self.num_summary_tokens = int(num_summary_tokens)
        if self.num_summary_tokens < 1:
            raise ValueError("num_summary_tokens must be >= 1.")

        self.patient_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # If use_cluster_embedding is True, add a learnable identity embedding to each cluster's summary token(s) 
        # to help the fusion transformer distinguish clusters.
        self.cluster_embedding = (
            nn.Embedding(self.num_clusters, d_model)
            if self.use_cluster_embedding
            else None
        )
        self.encoder = TransformerSetEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        nn.init.normal_(self.patient_cls_token, std=0.02)
        if self.cluster_embedding is not None:
            nn.init.normal_(self.cluster_embedding.weight, std=0.02)

    def forward(
        self,
        group_tokens: torch.Tensor,
        cluster_available_mask: torch.Tensor | None = None,
        summary_slot_indices: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Fuse all expert summary tokens into one patient embedding.
        Input:
            group_tokens: [B, K, M, d_model] or [B, K, d_model] if num_summary_tokens=1, where K=num_clusters and M=num_summary_tokens
            cluster_available_mask: optional [B, K] bool mask indicating which clusters have any available features. If None, all clusters are assumed available.
            summary_slot_indices: optional list of summary token indices to retain for fusion. If None, all summary tokens are used.
        Returns:
            patient_embedding: [B, d_model] embedding for the patient.
            fused_group_tokens: [B, K, M, d_model] or [B, K, d_model] if num_summary_tokens=1, the fused summary tokens for each cluster.
        """
        input_was_single_summary = group_tokens.ndim == 3
        if input_was_single_summary:
            if self.num_summary_tokens != 1:
                raise ValueError(
                    "A 3D group_tokens tensor is valid only when "
                    "num_summary_tokens=1."
                )
            group_tokens_4d = group_tokens.unsqueeze(2)
        elif group_tokens.ndim == 4:
            group_tokens_4d = group_tokens
        else:
            raise ValueError(
                "group_tokens must have shape [B, K, d] or [B, K, M, d], "
                f"got {tuple(group_tokens.shape)}."
            )

        B, K, M, d = group_tokens_4d.shape
        device = group_tokens_4d.device

        # Check the full expert output against the checkpoint architecture first.
        if M != self.num_summary_tokens:
            raise ValueError(
                f"group_tokens has M={M}, expected {self.num_summary_tokens}."
            )

        # Optionally retain only selected summary slots for fusion.
        if summary_slot_indices is not None:
            slots = [int(slot) for slot in summary_slot_indices]

            if not slots:
                raise ValueError("summary_slot_indices must not be empty.")
            if len(set(slots)) != len(slots):
                raise ValueError(f"Duplicate summary slots: {slots}")
            if min(slots) < 0 or max(slots) >= M:
                raise ValueError(f"Invalid summary slots {slots}; expected indices 0..{M - 1}.")

            slot_idx = torch.tensor(slots, dtype=torch.long, device=device)
            group_tokens_4d = group_tokens_4d.index_select(2, slot_idx)
            M = len(slots)
            
        if K != self.num_clusters:
            raise ValueError(
                f"group_tokens has K={K}, expected {self.num_clusters}."
            )
        if d != self.d_model:
            raise ValueError(
                f"group_tokens last dimension is {d}, expected {self.d_model}."
            )

        if cluster_available_mask is None:
            cluster_available_mask = torch.ones(
                B, K, dtype=torch.bool, device=device
            )
        else:
            cluster_available_mask = cluster_available_mask.to(
                device=device, dtype=torch.bool
            )
            if cluster_available_mask.shape != (B, K):
                raise ValueError(
                    f"cluster_available_mask must have shape {(B, K)}, "
                    f"got {tuple(cluster_available_mask.shape)}."
                )

        cluster_tokens = group_tokens_4d
        if self.cluster_embedding is not None:
            cluster_ids = torch.arange(K, device=device)
            cluster_tokens = cluster_tokens + self.cluster_embedding(
                cluster_ids
            ).view(1, K, 1, d)

        flat_cluster_tokens = cluster_tokens.reshape(B, K * M, d)
        flat_available = cluster_available_mask.unsqueeze(-1).expand(B, K, M)
        flat_available = flat_available.reshape(B, K * M)

        patient_cls = self.patient_cls_token.expand(B, -1, -1)
        tokens = torch.cat([patient_cls, flat_cluster_tokens], dim=1)
        patient_cls_available = torch.ones(B, 1, dtype=torch.bool, device=device)
        token_available = torch.cat(
            [patient_cls_available, flat_available], dim=1
        )
        encoded = self.encoder(tokens, key_padding_mask=~token_available)

        patient_embedding = encoded[:, 0]
        fused_4d = encoded[:, 1:].reshape(B, K, M, d)
        fused_group_tokens = (
            fused_4d[:, :, 0]
            if input_was_single_summary
            else fused_4d
        )
        return patient_embedding, fused_group_tokens

# Full SGMET encoder from tokenized patient table to patient embedding.
class ClinicalClusterEncoder(nn.Module):
    """Full SGMET encoder from tokenized patient table to patient embedding."""

    def __init__(
        self,
        *,
        cluster_assignments: torch.Tensor,
        name_embeddings: torch.Tensor,
        feature_type_ids: torch.Tensor,
        categorical_cardinalities: list[int],
        continuous_bin_cardinality: int,
        missing_reason_cardinality: int,
        feature_context_offsets: torch.Tensor | None = None,
        num_clusters: int | None = None,
        d_model: int = 64,
        expert_n_heads: int = 4,
        expert_n_layers: int = 2,
        fusion_n_heads: int = 4,
        fusion_n_layers: int = 2,
        dropout: float = 0.1,
        categorical_embedding_weights: Mapping[int, torch.Tensor] | None = None,
        use_cluster_embedding: bool = True,
        num_summary_tokens: int = 1,
    ):
        super().__init__()
        if num_clusters is None:
            num_clusters = int(cluster_assignments.max().item()) + 1
        self.num_summary_tokens = int(num_summary_tokens)
        if self.num_summary_tokens < 1:
            raise ValueError("num_summary_tokens must be >= 1.")

        self.expert_bank = ExpertBank(
            cluster_assignments=cluster_assignments,
            name_embeddings=name_embeddings,
            feature_type_ids=feature_type_ids,
            categorical_cardinalities=categorical_cardinalities,
            continuous_bin_cardinality=continuous_bin_cardinality,
            missing_reason_cardinality=missing_reason_cardinality,
            feature_context_offsets=feature_context_offsets,
            num_clusters=num_clusters,
            d_model=d_model,
            n_heads=expert_n_heads,
            n_layers=expert_n_layers,
            dropout=dropout,
            categorical_embedding_weights=categorical_embedding_weights,
            num_summary_tokens=self.num_summary_tokens,
        )
        self.fusion = PatientFusionTransformer(
            num_clusters=int(num_clusters),
            d_model=d_model,
            n_heads=fusion_n_heads,
            n_layers=fusion_n_layers,
            dropout=dropout,
            use_cluster_embedding=use_cluster_embedding,
            num_summary_tokens=self.num_summary_tokens,
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        feature_available_mask: torch.Tensor | None = None,
        *,
        return_summary_attention: bool = False,
        fusion_summary_slot_indices: list[int] | tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        if return_summary_attention:
            (
                group_tokens, cluster_available_mask, attention_overlap,
                feature_attention_by_cluster, feature_mask_by_cluster,
            ) = self.expert_bank(
                batch=batch,
                feature_available_mask=feature_available_mask,
                return_attention_overlap=True,
            )
        else:
            group_tokens, cluster_available_mask = self.expert_bank(
                batch=batch,
                feature_available_mask=feature_available_mask,
            )
            attention_overlap = None
            feature_attention_by_cluster = None
            feature_mask_by_cluster = None

        patient_embedding, fused_group_tokens = self.fusion(
            group_tokens=group_tokens,
            cluster_available_mask=cluster_available_mask,
            summary_slot_indices=fusion_summary_slot_indices,
        )
        out: dict[str, object] = {
            "patient_embedding": patient_embedding,
            "group_tokens": group_tokens,
            "cluster_available_mask": cluster_available_mask,
            "fused_group_tokens": fused_group_tokens,
        }
        if return_summary_attention:
            out["summary_attention_overlap"] = attention_overlap
            out["summary_feature_attention"] = feature_attention_by_cluster
            out["summary_feature_available_mask"] = feature_mask_by_cluster
        return out

    def forward_active_clusters(
        self,
        batch: Mapping[str, torch.Tensor],
        active_clusters: list[int] | tuple[int, ...] | None = None,
        feature_available_mask: torch.Tensor | None = None,
        *,
        return_summary_attention: bool = False,
        fusion_summary_slot_indices: list[int] | tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        """ 
        Forward pass using only a subset of active clusters.
        Input:
            batch: dictionary containing feature data for all clusters. Must contain keys in TOKEN_BATCH_KEYS.
            active_clusters: list or tuple of cluster ids to use. If None, uses all clusters.
            feature_available_mask: optional [B, N] bool mask for transformer attention, where N is the total number of features. If None, all features are assumed available.
        Returns:
            patient_embedding: [B, d_model] embedding for the patient.
            group_tokens: [B, K, M, d_model] or [B, K, d_model] if num_summary_tokens=1, the summary tokens for each active cluster.
            cluster_available_mask: [B, K] bool mask indicating if any features were available for each active cluster.
            fused_group_tokens: [B, K, M, d_model] or [B, K, d_model] if num_summary_tokens=1, the fused summary tokens for each active cluster.
            If return_summary_attention=True, also returns [B, K, M, M] pairwise CLS-to-feature attention overlap for each active cluster. 
                Entries are NaN for patients with fewer than two available features in a cluster.
        """
        if active_clusters is None:
            return self.forward(
                batch=batch,
                feature_available_mask=feature_available_mask,
                return_summary_attention=return_summary_attention,
                fusion_summary_slot_indices=fusion_summary_slot_indices,
            )
        if "numeric_values" not in batch:
            raise KeyError("batch must contain 'numeric_values'.")

        numeric_values = batch["numeric_values"]
        if numeric_values.ndim != 2:
            raise ValueError(
                "batch['numeric_values'] must have shape [B, N], "
                f"got {tuple(numeric_values.shape)}."
            )
        B, N = numeric_values.shape
        bank = self.expert_bank
        K, M, d = bank.n_clusters, bank.num_summary_tokens, bank.d_model
        device, dtype = numeric_values.device, numeric_values.dtype
        if N != bank.n_features:
            raise ValueError(
                f"Input batch has {N} features, but ExpertBank was initialized "
                f"with {bank.n_features} features."
            )

        if feature_available_mask is None:
            feature_available_mask = torch.ones(
                B, N, dtype=torch.bool, device=device
            )
        else:
            feature_available_mask = feature_available_mask.to(
                device=device, dtype=torch.bool
            )
            if feature_available_mask.shape != (B, N):
                raise ValueError(
                    f"feature_available_mask must have shape {(B, N)}, "
                    f"got {tuple(feature_available_mask.shape)}."
                )

        if M == 1:
            group_tokens = torch.zeros(B, K, d, device=device, dtype=dtype)
        else:
            group_tokens = torch.zeros(B, K, M, d, device=device, dtype=dtype)
        cluster_available_mask = torch.zeros(
            B, K, dtype=torch.bool, device=device
        )
        attention_overlap = (
            torch.full(
                (B, K, M, M),
                float("nan"),
                device=device,
                dtype=dtype,
            )
            if return_summary_attention
            else None
        )
        feature_attention_by_cluster: dict[int, torch.Tensor] = {}
        feature_mask_by_cluster: dict[int, torch.Tensor] = {}

        for cluster_id in active_clusters:
            cluster_id = int(cluster_id)
            if cluster_id < 0 or cluster_id >= K:
                raise ValueError(
                    f"Invalid active cluster {cluster_id}; expected 0..{K - 1}."
                )
            if return_summary_attention:
                token_c, available_c, overlap_c, feature_attention_c, feature_mask_c = bank.branches[cluster_id](
                    batch=batch,
                    feature_available_mask=feature_available_mask,
                    return_attention_overlap=True,
                )
                assert attention_overlap is not None
                attention_overlap[:, cluster_id] = overlap_c
                feature_attention_by_cluster[cluster_id] = feature_attention_c
                feature_mask_by_cluster[cluster_id] = feature_mask_c
            else:
                token_c, available_c = bank.branches[cluster_id](
                    batch=batch,
                    feature_available_mask=feature_available_mask,
                )
            group_tokens[:, cluster_id] = token_c
            cluster_available_mask[:, cluster_id] = available_c

        patient_embedding, fused_group_tokens = self.fusion(
            group_tokens=group_tokens,
            cluster_available_mask=cluster_available_mask,
            summary_slot_indices=fusion_summary_slot_indices,
        )
        out: dict[str, object] = {
            "patient_embedding": patient_embedding,
            "group_tokens": group_tokens,
            "cluster_available_mask": cluster_available_mask,
            "fused_group_tokens": fused_group_tokens,
        }
        if return_summary_attention:
            out["summary_attention_overlap"] = attention_overlap
            out["summary_feature_attention"] = feature_attention_by_cluster
            out["summary_feature_available_mask"] = feature_mask_by_cluster
        return out


class DownstreamPredictionHead(nn.Module):
    """Prediction head for T binary tasks from one patient embedding.
    Input:
        patient_embedding: [B, d_model]
    Output:
        binary_logits: [B, T]
    """

    def __init__(self, d_model: int = 64, n_binary_targets: int = 11, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(hidden_dim or d_model)

        self.shared = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout)
        )
        self.binary_head = nn.Linear(hidden_dim, int(n_binary_targets))
    
    def forward(self, patient_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(patient_embedding)
        return {"binary_logits": self.binary_head(h)}

    @staticmethod
    def predictions(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"binary_probs": torch.sigmoid(outputs["binary_logits"])}

    @staticmethod
    def probabilities(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return DownstreamPredictionHead.predictions(outputs)