from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


SUMMARY_AUX_LOSS_CHOICES = (
    "none",
    "attention_margin",
    "attention_mi",
    "output_margin",
)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _upper_triangle_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
    )


def output_cosine_margin_loss(
    group_tokens: torch.Tensor,
    cluster_available_mask: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize only summary-output cosine similarities above ``margin``.

    Args:
        group_tokens: [B, K, M, D].
        cluster_available_mask: [B, K].

    Returns:
        (loss, mean_pairwise_cosine) over available cluster/patient pairs.
    """
    if group_tokens.ndim != 4:
        raise ValueError(
            "output cosine diversity requires group_tokens [B, K, M, D], "
            f"got {tuple(group_tokens.shape)}."
        )
    B, K, M, _ = group_tokens.shape
    if M < 2:
        raise ValueError("summary auxiliary losses require at least 2 summary tokens.")
    if cluster_available_mask.shape != (B, K):
        raise ValueError(
            f"cluster_available_mask must have shape {(B, K)}, "
            f"got {tuple(cluster_available_mask.shape)}."
        )

    z = F.normalize(group_tokens, p=2, dim=-1, eps=1e-8)
    cosine = torch.matmul(z, z.transpose(-1, -2))
    pair_mask = _upper_triangle_mask(M, group_tokens.device)
    valid = cluster_available_mask[..., None, None] & pair_mask
    values = cosine.masked_select(valid)
    if values.numel() == 0:
        zero = _zero(group_tokens)
        return zero, zero
    return F.relu(values - float(margin)).square().mean(), values.mean()


def attention_cosine_margin_loss(
    attention_by_cluster: Mapping[int, torch.Tensor],
    feature_mask_by_cluster: Mapping[int, torch.Tensor],
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize excessive cosine overlap between CLS-to-feature distributions.

    Each attention tensor has shape [B, M, A_c] and each feature mask [B, A_c].
    Rows with fewer than two available features are excluded because two slots
    cannot meaningfully specialize there.
    """
    penalties, similarities = [], []
    reference = None

    for cluster_id, attention in attention_by_cluster.items():
        reference = attention if reference is None else reference
        feature_mask = feature_mask_by_cluster[cluster_id].bool()
        if attention.ndim != 3:
            raise ValueError(
                f"cluster {cluster_id}: attention must be [B, M, A], "
                f"got {tuple(attention.shape)}."
            )
        B, M, A = attention.shape
        if M < 2:
            raise ValueError("summary auxiliary losses require at least 2 summary tokens.")
        if feature_mask.shape != (B, A):
            raise ValueError(
                f"cluster {cluster_id}: feature mask must be {(B, A)}, "
                f"got {tuple(feature_mask.shape)}."
            )

        normalized = F.normalize(attention, p=2, dim=-1, eps=1e-8)
        cosine = torch.matmul(normalized, normalized.transpose(-1, -2))
        valid_rows = feature_mask.sum(dim=-1).ge(2) & attention.sum(dim=-1).gt(1e-8).all(dim=-1)
        valid = valid_rows[:, None, None] & _upper_triangle_mask(M, attention.device)
        values = cosine.masked_select(valid)
        if values.numel() > 0:
            similarities.append(values)
            penalties.append(F.relu(values - float(margin)).square())

    if reference is None:
        raise ValueError("attention_by_cluster must not be empty.")
    if not penalties:
        zero = _zero(reference)
        return zero, zero
    return torch.cat(penalties).mean(), torch.cat(similarities).mean()


def attention_mutual_information_loss(
    attention_by_cluster: Mapping[int, torch.Tensor],
    feature_mask_by_cluster: Mapping[int, torch.Tensor],
    *,
    beta: float = 1.0,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Encourage balanced, confident feature-to-slot assignments.

    Attention is originally normalized over features independently for each slot.
    We turn it into a competitive assignment p(slot | feature) by applying a
    softmax across slots to log-attention values.

    The optimized non-negative objective is

        H(S|J)/log(M) + beta * (1 - H(S)/log(M)).

    It has the same gradients as H(S|J) - beta H(S), up to a positive scale and
    additive constant, while being easier to monitor: zero is the ideal balanced,
    deterministic assignment.
    """
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0.")
    if float(beta) < 0.0:
        raise ValueError("beta must be >= 0.")

    losses, conditional_entropies, marginal_entropies = [], [], []
    reference = None

    for cluster_id, attention in attention_by_cluster.items():
        reference = attention if reference is None else reference
        feature_mask = feature_mask_by_cluster[cluster_id].bool()
        if attention.ndim != 3:
            raise ValueError(
                f"cluster {cluster_id}: attention must be [B, M, A], "
                f"got {tuple(attention.shape)}."
            )
        B, M, A = attention.shape
        if M < 2:
            raise ValueError("summary auxiliary losses require at least 2 summary tokens.")
        if feature_mask.shape != (B, A):
            raise ValueError(
                f"cluster {cluster_id}: feature mask must be {(B, A)}, "
                f"got {tuple(feature_mask.shape)}."
            )

        valid_rows = feature_mask.sum(dim=-1).ge(2) & attention.sum(dim=-1).gt(1e-8).all(dim=-1)
        if not valid_rows.any():
            continue

        # p(slot | feature), shape [B, M, A].
        slot_logits = attention.clamp_min(1e-12).log() / float(temperature)
        slot_prob = torch.softmax(slot_logits, dim=1)
        log_slot_prob = slot_prob.clamp_min(1e-12).log()

        mask = feature_mask.to(attention.dtype)
        n_features = mask.sum(dim=-1).clamp_min(1.0)

        # H(S|J): first entropy over slots for each feature, then mean features.
        entropy_per_feature = -(slot_prob * log_slot_prob).sum(dim=1)
        h_cond = (entropy_per_feature * mask).sum(dim=-1) / n_features

        # H(S): entropy of each patient's average slot usage.
        marginal = (slot_prob * mask[:, None, :]).sum(dim=-1) / n_features[:, None]
        h_marg = -(marginal * marginal.clamp_min(1e-12).log()).sum(dim=-1)

        log_m = attention.new_tensor(float(M)).log()
        normalized_loss = h_cond / log_m + float(beta) * (1.0 - h_marg / log_m)

        losses.append(normalized_loss[valid_rows])
        conditional_entropies.append(h_cond[valid_rows])
        marginal_entropies.append(h_marg[valid_rows])

    if reference is None:
        raise ValueError("attention_by_cluster must not be empty.")
    if not losses:
        zero = _zero(reference)
        return {"loss": zero, "conditional_entropy": zero, "marginal_entropy": zero}

    return {
        "loss": torch.cat(losses).mean(),
        "conditional_entropy": torch.cat(conditional_entropies).mean(),
        "marginal_entropy": torch.cat(marginal_entropies).mean(),
    }


def compute_summary_auxiliary_loss(
    loss_type: str,
    *,
    group_tokens: torch.Tensor,
    cluster_available_mask: torch.Tensor,
    attention_by_cluster: Mapping[int, torch.Tensor] | None = None,
    feature_mask_by_cluster: Mapping[int, torch.Tensor] | None = None,
    attention_margin: float = 0.9,
    output_margin: float = 0.9,
    mi_beta: float = 1.0,
    mi_temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Dispatch one summary-token auxiliary loss and return scalar diagnostics."""
    loss_type = str(loss_type).lower()
    if loss_type not in SUMMARY_AUX_LOSS_CHOICES:
        raise ValueError(
            f"Unknown summary auxiliary loss {loss_type!r}; "
            f"choose from {SUMMARY_AUX_LOSS_CHOICES}."
        )

    zero = _zero(group_tokens)
    result = {
        "loss": zero,
        "pairwise_similarity": zero,
        "conditional_entropy": zero,
        "marginal_entropy": zero,
    }
    if loss_type == "none":
        return result

    if loss_type == "output_margin":
        loss, similarity = output_cosine_margin_loss(
            group_tokens, cluster_available_mask, margin=output_margin
        )
        result.update(loss=loss, pairwise_similarity=similarity)
        return result

    if attention_by_cluster is None or feature_mask_by_cluster is None:
        raise ValueError(f"{loss_type} requires differentiable feature attention.")

    if loss_type == "attention_margin":
        loss, similarity = attention_cosine_margin_loss(
            attention_by_cluster, feature_mask_by_cluster, margin=attention_margin
        )
        result.update(loss=loss, pairwise_similarity=similarity)
        return result

    mi = attention_mutual_information_loss(
        attention_by_cluster,
        feature_mask_by_cluster,
        beta=mi_beta,
        temperature=mi_temperature,
    )
    result.update(mi)
    return result
