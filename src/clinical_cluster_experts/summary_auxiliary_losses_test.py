from __future__ import annotations

import torch

from src.clinical_cluster_experts.summary_auxiliary_losses import (
    attention_cosine_margin_loss,
    attention_mutual_information_loss,
    output_cosine_margin_loss,
)


def test_output_margin_detects_duplicate_slots() -> None:
    base = torch.randn(4, 3, 1, 8)
    tokens = torch.cat([base, base.clone()], dim=2).requires_grad_(True)
    available = torch.ones(4, 3, dtype=torch.bool)
    loss, similarity = output_cosine_margin_loss(tokens, available, margin=0.9)
    assert similarity > 0.999
    assert loss > 0
    loss.backward()
    assert tokens.grad is not None


def test_attention_margin_detects_duplicate_attention() -> None:
    attention = torch.softmax(torch.randn(5, 1, 7), dim=-1).repeat(1, 2, 1)
    attention.requires_grad_(True)
    mask = torch.ones(5, 7, dtype=torch.bool)
    loss, similarity = attention_cosine_margin_loss({0: attention}, {0: mask}, margin=0.9)
    assert similarity > 0.999
    assert loss > 0
    loss.backward()
    assert attention.grad is not None


def test_attention_mi_prefers_balanced_confident_assignment() -> None:
    mask = torch.ones(2, 4, dtype=torch.bool)
    collapsed = torch.full((2, 2, 4), 0.25, requires_grad=True)
    specialized = torch.tensor(
        [[[0.49, 0.49, 0.01, 0.01], [0.01, 0.01, 0.49, 0.49]]],
        dtype=torch.float32,
    ).repeat(2, 1, 1).requires_grad_(True)

    collapsed_loss = attention_mutual_information_loss({0: collapsed}, {0: mask})["loss"]
    specialized_loss = attention_mutual_information_loss({0: specialized}, {0: mask})["loss"]
    assert specialized_loss < collapsed_loss


if __name__ == "__main__":
    test_output_margin_detects_duplicate_slots()
    test_attention_margin_detects_duplicate_attention()
    test_attention_mi_prefers_balanced_confident_assignment()
    print("All summary auxiliary loss tests passed.")
