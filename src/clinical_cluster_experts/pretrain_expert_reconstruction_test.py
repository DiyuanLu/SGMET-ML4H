"""Tests for clinical_cluster_experts.pretrain_expert_reconstruction.

Run with:
python -m src.clinical_cluster_experts.pretrain_expert_reconstruction_test --device mps
"""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable

import torch

from src.tokenizer.schema import FeatureType
from src.clinical_cluster_experts.model import ClusterBranch, TOKEN_BATCH_KEYS
from src.clinical_cluster_experts.pretrain_expert_reconstruction import (
    ClusterMAEReconstructionHead,
    prepare_masked_reconstruction_batch,
    reconstruction_loss,
)
from src.clinical_cluster_experts.summary_auxiliary_losses import (
    SUMMARY_AUX_LOSS_CHOICES,
    compute_summary_auxiliary_loss,
)


USE_COLOR = True


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(text: str) -> str:
    return color(text, "92")


def red(text: str) -> str:
    return color(text, "91")


def cyan(text: str) -> str:
    return color(text, "96")


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available.")
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def fake_metadata(A: int, name_dim: int = 16) -> dict:
    """Create tiny mixed numerical/categorical metadata."""
    feature_type_ids = torch.tensor(
        [int(FeatureType.NUMERICAL) if i % 3 == 0 else int(FeatureType.CATEGORICAL) for i in range(A)]
    )
    return {
        "name_embeddings": torch.randn(A, name_dim),
        "feature_type_ids": feature_type_ids,
        "categorical_cardinalities": [
            1 if int(t) == int(FeatureType.NUMERICAL) else 5
            for t in feature_type_ids
        ],
        "continuous_bin_cardinality": 11,
        "missing_reason_cardinality": 11,
    }


def fake_batch(B: int, A: int, metadata: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Create a fake local token batch with all schema features available."""
    categorical_codes = torch.zeros(B, A, dtype=torch.long, device=device)
    for j, card in enumerate(metadata["categorical_cardinalities"]):
        if card > 1:
            categorical_codes[:, j] = torch.randint(0, card, (B,), device=device)
    return {
        "numeric_values": torch.randn(B, A, device=device),
        "continuous_bin_codes": torch.randint(0, 11, (B, A), dtype=torch.long, device=device),
        "categorical_codes": categorical_codes,
        "missing_reason_codes": torch.zeros(B, A, dtype=torch.long, device=device),
        "missing_mask": torch.zeros(B, A, dtype=torch.bool, device=device),
        "observed_mask": torch.ones(B, A, dtype=torch.bool, device=device),
    }


def make_branch_and_decoder(
    A: int,
    d: int,
    metadata: dict,
    device: torch.device,
    num_summary_tokens: int = 1,
) -> tuple[ClusterBranch, ClusterMAEReconstructionHead]:
    """Construct a small cluster branch and reconstruction decoder."""
    branch = ClusterBranch(
        feature_indices=torch.arange(A),
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        num_summary_tokens=num_summary_tokens,
    ).to(device)
    decoder = ClusterMAEReconstructionHead(
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        d_model=d,
        dropout=0.0,
    ).to(device)
    assert branch.token_builder is not None
    return branch, decoder


def test_masking_keeps_one_observed_context(device: torch.device) -> None:
    """Artificial target masking should not hide the last observed value when avoidable."""
    metadata = fake_metadata(6)
    batch = fake_batch(3, 6, metadata, device)
    result = prepare_masked_reconstruction_batch(
        batch,
        feature_mask_prob_max=1.0,
        schema_dropout_prob_max=0.0,
        mask_reason_code=10,
        sample_probabilities=False,
        force_target_per_row=True,
        keep_one_observed_value_visible=True,
    )

    assert result.artificial_masked_mask.any()
    assert torch.equal(result.schema_available_mask, torch.ones_like(result.schema_available_mask))
    assert torch.equal(result.encoder_available_mask, result.schema_available_mask & ~result.artificial_masked_mask)

    # With six true observed values per row and p=1, all but one should be masked.
    assert torch.equal(result.artificial_masked_mask.sum(dim=1), torch.full((3,), 5, device=device))
    assert torch.equal((result.encoder_available_mask & result.has_value_mask).sum(dim=1), torch.ones(3, device=device))

    m = result.artificial_masked_mask
    assert torch.all(result.corrupted_batch["numeric_values"][m] == 0)
    assert torch.all(result.corrupted_batch["continuous_bin_codes"][m] == 0)
    assert torch.all(result.corrupted_batch["categorical_codes"][m] == 0)
    assert torch.all(result.corrupted_batch["missing_mask"][m])
    assert torch.all(result.corrupted_batch["missing_reason_codes"][m] == 10)


def test_schema_dropout_removes_features_without_targets(device: torch.device) -> None:
    """Schema-dropped features should be invisible and should not become targets."""
    metadata = fake_metadata(5)
    batch = fake_batch(2, 5, metadata, device)
    result = prepare_masked_reconstruction_batch(
        batch,
        feature_mask_prob_max=1.0,
        schema_dropout_prob_max=1.0,
        mask_reason_code=10,
        sample_probabilities=False,
        force_target_per_row=True,
        keep_one_observed_value_visible=True,
    )

    assert not result.schema_available_mask.any()
    assert result.schema_dropped_mask.all()
    assert not result.target_candidate_mask.any()
    assert not result.artificial_masked_mask.any()
    assert not result.encoder_available_mask.any()


def test_true_patient_missing_tokens_can_remain_visible(device: torch.device) -> None:
    """Patient-missing tokens are not reconstruction targets but can remain encoder-visible."""
    metadata = fake_metadata(4)
    batch = fake_batch(2, 4, metadata, device)
    batch["missing_mask"][:, 0] = True
    batch["missing_reason_codes"][:, 0] = 3

    result = prepare_masked_reconstruction_batch(
        batch,
        feature_mask_prob_max=0.0,
        schema_dropout_prob_max=0.0,
        mask_reason_code=10,
        sample_probabilities=False,
        force_target_per_row=False,
        keep_one_observed_value_visible=True,
    )

    assert not result.has_value_mask[:, 0].any()
    assert result.encoder_available_mask[:, 0].all()
    assert not result.target_candidate_mask[:, 0].any()
    assert not result.artificial_masked_mask[:, 0].any()


def test_decoder_queries(device: torch.device) -> None:
    """Metadata-only decoder queries should have the expected shape."""
    A = 7
    D = 32
    metadata = fake_metadata(A)
    branch, decoder = make_branch_and_decoder(A, D, metadata, device)
    del decoder

    q = branch.token_builder.metadata_tokens(
        batch_size=2,
        device=device,
    )
    assert q.shape == (2, A, D)


def test_mae_pretraining_step_backward(device: torch.device) -> None:
    """One full pretraining step should give gradients to branch and decoder."""
    B, A, d = 4, 9, 32
    metadata = fake_metadata(A)
    batch = fake_batch(B, A, metadata, device)
    branch, decoder = make_branch_and_decoder(A, d, metadata, device)

    result = prepare_masked_reconstruction_batch(
        batch,
        feature_mask_prob_max=0.5,
        schema_dropout_prob_max=0.0,
        mask_reason_code=10,
        sample_probabilities=False,
        force_target_per_row=True,
        keep_one_observed_value_visible=True,
    )
    z = branch.token_builder(result.corrupted_batch)
    h_cls = branch.expert(z, result.encoder_available_mask)
    loss, logs = reconstruction_loss(
        decoder=decoder,
        h_cls=h_cls,
        token_builder=branch.token_builder,
        original_local_batch=batch,
        mask_positions=result.artificial_masked_mask,
    )
    loss.backward()

    assert logs["n_masked"] > 0
    assert branch.expert.cls_token.grad is not None and branch.expert.cls_token.grad.abs().sum() > 0
    assert branch.token_builder.name_projection.weight.grad is not None
    assert branch.token_builder.name_projection.weight.grad.abs().sum() > 0
    assert decoder.decoder[0].weight.grad is not None and decoder.decoder[0].weight.grad.abs().sum() > 0


def test_multi_summary_auxiliary_loss_options(device: torch.device) -> None:
    """Two summary tokens should support reconstruction with every auxiliary-loss option."""
    B, A, d = 4, 9, 32
    metadata = fake_metadata(A)
    batch = fake_batch(B, A, metadata, device)

    assert set(SUMMARY_AUX_LOSS_CHOICES) == {"none", "attention_margin", "attention_mi", "output_margin"}

    for loss_type in SUMMARY_AUX_LOSS_CHOICES:
        branch, decoder = make_branch_and_decoder(A, d, metadata, device, num_summary_tokens=2)
        result = prepare_masked_reconstruction_batch(
            batch,
            feature_mask_prob_max=0.5,
            schema_dropout_prob_max=0.0,
            mask_reason_code=10,
            sample_probabilities=False,
            force_target_per_row=True,
            keep_one_observed_value_visible=True,
        )

        assert branch.token_builder is not None
        z = branch.token_builder(result.corrupted_batch)
        need_summary_attention = loss_type in {"attention_margin", "attention_mi"}
        if need_summary_attention:
            h_cls, _, feature_attention = branch.expert(
                z,
                result.encoder_available_mask,
                return_attention_overlap=True,
            )
        else:
            h_cls = branch.expert(z, result.encoder_available_mask)
            feature_attention = None

        assert h_cls.shape == (B, 2, d)
        if need_summary_attention:
            assert feature_attention is not None
            assert feature_attention.shape == (B, 2, A)

        reconstruction, logs = reconstruction_loss(
            decoder=decoder,
            h_cls=h_cls,
            token_builder=branch.token_builder,
            original_local_batch=batch,
            mask_positions=result.artificial_masked_mask,
        )

        aux = compute_summary_auxiliary_loss(
            loss_type,
            group_tokens=h_cls.unsqueeze(1),
            cluster_available_mask=result.encoder_available_mask.any(dim=1, keepdim=True),
            attention_by_cluster=None if feature_attention is None else {0: feature_attention},
            feature_mask_by_cluster=None if feature_attention is None else {0: result.encoder_available_mask},
            attention_margin=0.0,
            output_margin=-0.99,
            mi_beta=1.0,
            mi_temperature=1.0,
        )

        optimization_loss = reconstruction + 0.005 * aux["loss"]
        optimization_loss.backward()

        assert logs["n_masked"] > 0
        assert torch.isfinite(reconstruction)
        assert torch.isfinite(aux["loss"])
        assert torch.isfinite(aux["pairwise_similarity"])
        assert torch.isfinite(aux["conditional_entropy"])
        assert torch.isfinite(aux["marginal_entropy"])

        grad = branch.expert.cls_token.grad
        assert grad is not None
        assert grad.shape == (1, 2, d)
        assert grad[0, 0].abs().sum() > 0
        assert grad[0, 1].abs().sum() > 0
        assert decoder.decoder[0].weight.grad is not None
        assert decoder.decoder[0].weight.grad.abs().sum() > 0


def test_categorical_loss_options(device: torch.device) -> None:
    """All categorical loss variants should be finite."""
    B, A, d = 4, 6, 32
    metadata = fake_metadata(A)
    metadata["feature_type_ids"] = torch.full((A,), int(FeatureType.CATEGORICAL), dtype=torch.long)
    metadata["categorical_cardinalities"] = [3 for _ in range(A)]
    batch = fake_batch(B, A, metadata, device)
    branch, decoder = make_branch_and_decoder(A, d, metadata, device)

    result = prepare_masked_reconstruction_batch(
        batch,
        feature_mask_prob_max=0.5,
        schema_dropout_prob_max=0.0,
        mask_reason_code=10,
        sample_probabilities=False,
        force_target_per_row=True,
        keep_one_observed_value_visible=True,
    )
    z = branch.token_builder(result.corrupted_batch)
    h_cls = branch.expert(z, result.encoder_available_mask)

    cat_weights = [torch.ones(3, device=device) for _ in range(A)]
    for cat_loss in ["ce", "weighted_ce", "focal", "weighted_focal"]:
        loss, logs = reconstruction_loss(
            decoder=decoder,
            h_cls=h_cls,
            token_builder=branch.token_builder,
            original_local_batch=batch,
            mask_positions=result.artificial_masked_mask,
            cat_loss=cat_loss,
            cat_class_weights=cat_weights,
            focal_gamma=2.0,
        )
        assert torch.isfinite(loss)
        assert logs["n_masked_cat"] > 0


def run_test(name: str, fn: Callable[[torch.device], None], device: torch.device) -> bool:
    try:
        fn(device)
        print(green(f"PASS {name}"))
        return True
    except Exception as exc:
        print(red(f"FAIL {name}: {type(exc).__name__}: {exc}"))
        traceback.print_exc()
        return False


def main() -> None:
    global USE_COLOR
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    USE_COLOR = not args.no_color
    device = resolve_device(args.device)
    print(cyan(f"Using device: {device}"))

    tests = [
        ("masking_keeps_one_observed_context", test_masking_keeps_one_observed_context),
        ("schema_dropout_removes_features_without_targets", test_schema_dropout_removes_features_without_targets),
        ("true_patient_missing_tokens_can_remain_visible", test_true_patient_missing_tokens_can_remain_visible),
        ("decoder_queries", test_decoder_queries),
        ("mae_pretraining_step_backward", test_mae_pretraining_step_backward),
        ("multi_summary_auxiliary_loss_options", test_multi_summary_auxiliary_loss_options),
        ("categorical_loss_options", test_categorical_loss_options),
    ]

    passed = sum(run_test(name, fn, device) for name, fn in tests)
    failed = len(tests) - passed
    print((green if failed == 0 else red)(f"\nSUMMARY: {passed}/{len(tests)} tests passed, {failed} failed."))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
