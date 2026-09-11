"Run with: python -m src.clinical_cluster_experts.model_test --device mps"

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable

import torch

from src.clinical_cluster_experts.model import (
    TransformerSetEncoder,
    ClusterExpertEncoder,
    ClusterBranch,
    ExpertBank,
    PatientFusionTransformer,
    DownstreamPredictionHead,
    ClinicalClusterEncoder,
    #ClinicalClusterDownstreamModel,
)

from src.clinical_cluster_experts.token_builder import FeatureTokenBuilder
from src.tokenizer.schema import FeatureType

USE_COLOR = True


def color(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return color(text, "92")


def red(text: str) -> str:
    return color(text, "91")


def yellow(text: str) -> str:
    return color(text, "93")


def cyan(text: str) -> str:
    return color(text, "96")


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_fake_metadata(N: int, name_dim: int = 16) -> dict:
    feature_type_ids = torch.tensor(
        [0 if i % 3 == 0 else 1 for i in range(N)],
        dtype=torch.long,
    )
    categorical_cardinalities = [1 if int(t) == 0 else 5 for t in feature_type_ids]

    return {
        "name_embeddings": torch.randn(N, name_dim),
        "feature_type_ids": feature_type_ids,
        "categorical_cardinalities": categorical_cardinalities,
        "continuous_bin_cardinality": 11,
        "missing_reason_cardinality": 10,
    }


def make_fake_batch(B: int, N: int, metadata: dict, device: torch.device) -> dict[str, torch.Tensor]:
    categorical_cardinalities = metadata["categorical_cardinalities"]

    categorical_codes = torch.zeros(B, N, dtype=torch.long)
    for j, card in enumerate(categorical_cardinalities):
        if card > 1:
            categorical_codes[:, j] = torch.randint(0, card, (B,))

    return {
        "numeric_values": torch.randn(B, N, device=device),
        "continuous_bin_codes": torch.randint(0, 11, (B, N), dtype=torch.long, device=device),
        "categorical_codes": categorical_codes.to(device),
        "missing_reason_codes": torch.randint(0, 10, (B, N), dtype=torch.long, device=device),
        "missing_mask": (torch.rand(B, N, device=device) < 0.1),
        "row_idx": torch.arange(B, dtype=torch.long, device=device),
    }


# Test the shared no-positional-encoding Transformer block and its key-padding mask shape behavior.
def test_transformer_set_encoder(device: torch.device) -> None:
    B, L, d = 3, 5, 32
    encoder = TransformerSetEncoder(d_model=d, n_heads=4, n_layers=1, dropout=0.1).to(device)
    encoder.eval()

    tokens = torch.randn(B, L, d, device=device)
    key_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    key_padding_mask[:, -1:] = True

    with torch.no_grad():
        out = encoder(tokens, key_padding_mask=key_padding_mask)

    assert out.shape == (B, L, d), out.shape


# Test one cluster expert maps local feature tokens to one group token without a learned missing token.
def test_cluster_expert_encoder(device: torch.device) -> None:
    B, A_c, d = 3, 4, 32
    expert = ClusterExpertEncoder(d_model=d, n_heads=4, n_layers=1, dropout=0.1).to(device)
    expert.eval()

    assert not hasattr(expert, "missing_cluster_token"), (
        "ClusterExpertEncoder should no longer learn missing_cluster_token. "
        "Unavailable clusters are handled by cluster_available_mask in fusion."
    )

    feature_tokens = torch.randn(B, A_c, d, device=device)
    feature_available_mask = torch.ones(B, A_c, dtype=torch.bool, device=device)
    feature_available_mask[1, :] = False

    with torch.no_grad():
        group_token = expert(feature_tokens, feature_available_mask)

    assert group_token.shape == (B, d), group_token.shape


# Test one non-empty branch selects local features, tokenizes them, and reports unavailable rows.
def test_cluster_branch(device: torch.device) -> None:
    B, N, d = 3, 16, 32
    metadata = make_fake_metadata(N)
    batch = make_fake_batch(B, N, metadata, device)
    feature_indices = torch.tensor([0, 2, 5, 9], dtype=torch.long)

    branch = ClusterBranch(
        feature_indices=feature_indices,
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ).to(device)
    branch.eval()

    feature_available_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    feature_available_mask[2, feature_indices.to(device)] = False

    with torch.no_grad():
        group_token, cluster_available = branch(batch, feature_available_mask=feature_available_mask)

    assert group_token.shape == (B, d), group_token.shape
    assert cluster_available.shape == (B,), cluster_available.shape
    assert cluster_available.tolist() == [True, True, False], cluster_available
    assert torch.allclose(group_token[2], torch.zeros_like(group_token[2]))


# Test a schema-empty branch returns zero placeholders and cluster_available=False for every patient.
def test_empty_cluster_branch(device: torch.device) -> None:
    B, N, d = 3, 16, 32
    metadata = make_fake_metadata(N)
    batch = make_fake_batch(B, N, metadata, device)

    branch = ClusterBranch(
        feature_indices=torch.empty(0, dtype=torch.long),
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ).to(device)
    branch.eval()

    with torch.no_grad():
        group_token, cluster_available = branch(batch)

    assert group_token.shape == (B, d), group_token.shape
    assert cluster_available.shape == (B,), cluster_available.shape
    assert not cluster_available.any()
    assert torch.allclose(group_token, torch.zeros_like(group_token))


# Test ExpertBank routes a raw batch through branch-local token builders to K group tokens.
def test_expert_bank(device: torch.device) -> None:
    B, N, d, K = 3, 24, 32, 4
    metadata = make_fake_metadata(N)
    batch = make_fake_batch(B, N, metadata, device)
    cluster_assignments = torch.arange(N) % K

    bank = ExpertBank(
        cluster_assignments=cluster_assignments,
        num_clusters=K,
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ).to(device)
    bank.eval()

    feature_available_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    cluster_0_idx = torch.where(cluster_assignments == 0)[0].to(device)
    feature_available_mask[2, cluster_0_idx] = False

    with torch.no_grad():
        group_tokens, cluster_available_mask = bank(batch, feature_available_mask=feature_available_mask)

    assert group_tokens.shape == (B, K, d), group_tokens.shape
    assert cluster_available_mask.shape == (B, K), cluster_available_mask.shape
    assert cluster_available_mask[2, 0].item() is False
    assert torch.allclose(group_tokens[2, 0], torch.zeros_like(group_tokens[2, 0]))


# Test an empty high-index cluster is preserved and masked out instead of represented by a learned token.
def test_expert_bank_with_empty_schema_cluster(device: torch.device) -> None:
    B, N, d, K = 3, 24, 32, 4
    metadata = make_fake_metadata(N)
    batch = make_fake_batch(B, N, metadata, device)

    # Uses only clusters 0..2. Cluster 3 has no schema feature.
    cluster_assignments = torch.arange(N) % (K - 1)

    bank = ExpertBank(
        cluster_assignments=cluster_assignments,
        num_clusters=K,
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ).to(device)
    bank.eval()

    with torch.no_grad():
        group_tokens, cluster_available_mask = bank(batch)

    assert group_tokens.shape == (B, K, d), group_tokens.shape
    assert cluster_available_mask.shape == (B, K), cluster_available_mask.shape

    empty_cluster_id = K - 1
    assert bank.feature_indices[empty_cluster_id].numel() == 0
    assert not cluster_available_mask[:, empty_cluster_id].any()
    assert torch.allclose(
        group_tokens[:, empty_cluster_id, :],
        torch.zeros_like(group_tokens[:, empty_cluster_id, :]),
    )


# Test empty middle clusters remain empty and are not shifted/remapped to neighboring cluster IDs.
def test_expert_bank_with_empty_middle_schema_clusters(device: torch.device) -> None:
    B, N, d, K = 3, 30, 32, 12
    metadata = make_fake_metadata(N)
    batch = make_fake_batch(B, N, metadata, device)

    empty_clusters = {3, 7, 11}
    active_clusters = [c for c in range(K) if c not in empty_clusters]

    cluster_assignments = torch.tensor(
        [active_clusters[j % len(active_clusters)] for j in range(N)],
        dtype=torch.long,
    )

    bank = ExpertBank(
        cluster_assignments=cluster_assignments,
        num_clusters=K,
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ).to(device)
    bank.eval()

    with torch.no_grad():
        group_tokens, cluster_available_mask = bank(batch)

    assert group_tokens.shape == (B, K, d)
    assert cluster_available_mask.shape == (B, K)

    for c in empty_clusters:
        assert bank.feature_indices[c].numel() == 0
        assert not cluster_available_mask[:, c].any()
        assert torch.allclose(
            group_tokens[:, c, :],
            torch.zeros_like(group_tokens[:, c, :]),
        )

    for c in active_clusters:
        assert bank.feature_indices[c].numel() > 0
        assert cluster_available_mask[:, c].all()


# Test non-consecutive cluster IDs stay assigned to their original expert/branch indices.
def test_no_cluster_id_remapping(device: torch.device) -> None:
    N, K = 12, 12

    # Only use non-consecutive cluster IDs.
    cluster_assignments = torch.tensor(
        [0, 0, 2, 2, 4, 4, 8, 8, 10, 10, 10, 10],
        dtype=torch.long,
    )

    metadata = make_fake_metadata(N)

    bank = ExpertBank(
        cluster_assignments=cluster_assignments,
        num_clusters=K,
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=32,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ).to(device)

    expected_non_empty = {0, 2, 4, 8, 10}
    expected_empty = set(range(K)) - expected_non_empty

    for c in expected_non_empty:
        assert bank.feature_indices[c].numel() > 0, f"cluster {c} should be non-empty"

    for c in expected_empty:
        assert bank.feature_indices[c].numel() == 0, f"cluster {c} should be empty"


# Test invalid num_clusters is rejected when cluster_assignments contain IDs outside the valid range.
def test_invalid_num_clusters_raises(device: torch.device) -> None:
    N = 12
    metadata = make_fake_metadata(N)

    cluster_assignments = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10])

    try:
        _ = ExpertBank(
            cluster_assignments=cluster_assignments,
            num_clusters=8,
            name_embeddings=metadata["name_embeddings"],
            feature_type_ids=metadata["feature_type_ids"],
            categorical_cardinalities=metadata["categorical_cardinalities"],
            continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
            missing_reason_cardinality=metadata["missing_reason_cardinality"],
            d_model=32,
            n_heads=4,
            n_layers=1,
            dropout=0.1,
        ).to(device)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for invalid num_clusters")


# Test partial schema dropout keeps a cluster available, while full schema dropout masks it from fusion.
def test_cluster_available_mask_under_partial_schema_dropout(device: torch.device) -> None:
    B, N, d, K = 3, 24, 32, 4
    metadata = make_fake_metadata(N)
    batch = make_fake_batch(B, N, metadata, device)
    cluster_assignments = torch.arange(N) % K

    bank = ExpertBank(
        cluster_assignments=cluster_assignments,
        num_clusters=K,
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ).to(device)
    bank.eval()

    feature_available_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    cluster_2_idx = torch.where(cluster_assignments == 2)[0].to(device)

    # Patient 0: all cluster 2 features available.
    # Patient 1: only some cluster 2 features dropped.
    feature_available_mask[1, cluster_2_idx[:-1]] = False

    # Patient 2: all cluster 2 features dropped.
    feature_available_mask[2, cluster_2_idx] = False

    with torch.no_grad():
        _, cluster_available_mask = bank(
            batch=batch,
            feature_available_mask=feature_available_mask,
        )

    assert cluster_available_mask[0, 2].item() is True
    assert cluster_available_mask[1, 2].item() is True
    assert cluster_available_mask[2, 2].item() is False


# Test fusion with cluster identity embeddings and cluster_available_mask.
def test_patient_fusion_transformer(device: torch.device) -> None:
    B, K, d = 3, 4, 32
    fusion = PatientFusionTransformer(
        num_clusters=K,
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
        use_cluster_embedding=True,
    ).to(device)
    fusion.eval()

    group_tokens = torch.randn(B, K, d, device=device)
    cluster_available_mask = torch.ones(B, K, dtype=torch.bool, device=device)
    cluster_available_mask[1, 2] = False
    cluster_available_mask[2, :] = False

    with torch.no_grad():
        patient_embedding, fused_group_tokens = fusion(
            group_tokens=group_tokens,
            cluster_available_mask=cluster_available_mask,
        )

    assert patient_embedding.shape == (B, d), patient_embedding.shape
    assert fused_group_tokens.shape == (B, K, d), fused_group_tokens.shape
    assert fusion.cluster_embedding is not None


# Test fusion ablation without learned cluster identity embeddings.
def test_patient_fusion_without_cluster_embedding(device: torch.device) -> None:
    B, K, d = 3, 4, 32
    fusion = PatientFusionTransformer(
        num_clusters=K,
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
        use_cluster_embedding=False,
    ).to(device)
    fusion.eval()

    assert getattr(fusion, "cluster_embedding", None) is None

    group_tokens = torch.randn(B, K, d, device=device)
    cluster_available_mask = torch.ones(B, K, dtype=torch.bool, device=device)
    cluster_available_mask[1, 2] = False

    with torch.no_grad():
        patient_embedding, fused_group_tokens = fusion(
            group_tokens=group_tokens,
            cluster_available_mask=cluster_available_mask,
        )

    assert patient_embedding.shape == (B, d), patient_embedding.shape
    assert fused_group_tokens.shape == (B, K, d), fused_group_tokens.shape


# Test the current binary prediction head used by supervised training.
def test_downstream_prediction_head(device: torch.device) -> None:
    B, d, T = 3, 32, 5
    head = DownstreamPredictionHead(
        d_model=d,
        n_binary_targets=T,
        hidden_dim=32,
        dropout=0.1,
    ).to(device)
    head.eval()

    patient_embedding = torch.randn(B, d, device=device)
    with torch.no_grad():
        out = head(patient_embedding)
        pred = head.predictions(out)

    assert out["binary_logits"].shape == (B, T), out["binary_logits"].shape
    assert pred["binary_probs"].shape == (B, T), pred["binary_probs"].shape
    assert torch.all(pred["binary_probs"] >= 0.0)
    assert torch.all(pred["binary_probs"] <= 1.0)


# # Test the full encoder and full downstream model from raw batch to logits/probabilities.
# def test_full_encoder_and_model(device: torch.device) -> None:
#     B, N, d, K = 3, 24, 32, 4
#     metadata = make_fake_metadata(N)
#     batch = make_fake_batch(B, N, metadata, device)
#     cluster_assignments = torch.arange(N) % K

#     feature_available_mask = torch.ones(B, N, dtype=torch.bool, device=device)
#     cluster_1_idx = torch.where(cluster_assignments == 1)[0].to(device)
#     feature_available_mask[2, cluster_1_idx] = False

#     encoder = ClinicalClusterEncoder(
#         cluster_assignments=cluster_assignments,
#         num_clusters=K,
#         name_embeddings=metadata["name_embeddings"],
#         feature_type_ids=metadata["feature_type_ids"],
#         categorical_cardinalities=metadata["categorical_cardinalities"],
#         continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
#         missing_reason_cardinality=metadata["missing_reason_cardinality"],
#         d_model=d,
#         expert_n_heads=4,
#         expert_n_layers=1,
#         fusion_n_heads=4,
#         fusion_n_layers=1,
#         dropout=0.1,
#         use_cluster_embedding=True,
#     ).to(device)
#     encoder.eval()

#     with torch.no_grad():
#         encoder_out = encoder(batch=batch, feature_available_mask=feature_available_mask)

#     assert encoder_out["group_tokens"].shape == (B, K, d)
#     assert encoder_out["cluster_available_mask"].shape == (B, K)
#     assert encoder_out["fused_group_tokens"].shape == (B, K, d)
#     assert encoder_out["patient_embedding"].shape == (B, d)
#     assert encoder_out["cluster_available_mask"][2, 1].item() is False

#     model = ClinicalClusterDownstreamModel(
#         cluster_assignments=cluster_assignments,
#         num_clusters=K,
#         name_embeddings=metadata["name_embeddings"],
#         feature_type_ids=metadata["feature_type_ids"],
#         categorical_cardinalities=metadata["categorical_cardinalities"],
#         continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
#         missing_reason_cardinality=metadata["missing_reason_cardinality"],
#         d_model=d,
#         expert_n_heads=4,
#         expert_n_layers=1,
#         fusion_n_heads=4,
#         fusion_n_layers=1,
#         dropout=0.1,
#         use_cluster_embedding=True,
#     ).to(device)
#     model.eval()

#     with torch.no_grad():
#         out = model(batch=batch, feature_available_mask=feature_available_mask)

#     assert out["group_tokens"].shape == (B, K, d)
#     assert out["cluster_available_mask"].shape == (B, K)
#     assert out["patient_embedding"].shape == (B, d)
#     assert out["tyg"].shape == (B,)
#     assert out["ckm_logits"].shape == (B, 3)
#     assert out["ckm_probs"].shape == (B, 3)
#     assert out["ckm_pred"].shape == (B,)
#     assert out["binary_logits"].shape == (B, 3)
#     assert out["binary_probs"].shape == (B, 3)


# # Test model state_dict roundtrip restores all branch-local builders, experts, fusion, and head weights.
# def test_checkpoint_save_load_roundtrip(device: torch.device) -> None:
#     B, N, d, K = 2, 16, 32, 4
#     metadata = make_fake_metadata(N)
#     batch = make_fake_batch(B, N, metadata, device)
#     cluster_assignments = torch.arange(N) % K

#     model1 = ClinicalClusterDownstreamModel(
#         cluster_assignments=cluster_assignments,
#         num_clusters=K,
#         name_embeddings=metadata["name_embeddings"],
#         feature_type_ids=metadata["feature_type_ids"],
#         categorical_cardinalities=metadata["categorical_cardinalities"],
#         continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
#         missing_reason_cardinality=metadata["missing_reason_cardinality"],
#         d_model=d,
#         expert_n_heads=4,
#         expert_n_layers=1,
#         fusion_n_heads=4,
#         fusion_n_layers=1,
#         dropout=0.0,
#         use_cluster_embedding=True,
#     ).to(device)
#     model1.eval()

#     with torch.no_grad():
#         out1_dict = model1(batch=batch)
#         out1 = torch.cat([
#             out1_dict["tyg"].unsqueeze(1),
#             out1_dict["ckm_logits"],
#             out1_dict["binary_logits"],
#         ], dim=1)

#     state = {k: v.detach().cpu().clone() for k, v in model1.state_dict().items()}

#     model2 = ClinicalClusterDownstreamModel(
#         cluster_assignments=cluster_assignments,
#         num_clusters=K,
#         name_embeddings=metadata["name_embeddings"],
#         feature_type_ids=metadata["feature_type_ids"],
#         categorical_cardinalities=metadata["categorical_cardinalities"],
#         continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
#         missing_reason_cardinality=metadata["missing_reason_cardinality"],
#         d_model=d,
#         expert_n_heads=4,
#         expert_n_layers=1,
#         fusion_n_heads=4,
#         fusion_n_layers=1,
#         dropout=0.0,
#         use_cluster_embedding=True,
#     ).to(device)

#     model2.load_state_dict(state)
#     model2.eval()

#     with torch.no_grad():
#         out2_dict = model2(batch=batch)
#         out2 = torch.cat([
#             out2_dict["tyg"].unsqueeze(1),
#             out2_dict["ckm_logits"],
#             out2_dict["binary_logits"],
#         ], dim=1)

#     assert torch.allclose(out1, out2, atol=1e-5), (out1, out2)


# # Test supervised losses can backpropagate into branch-local FeatureTokenBuilder_c and Expert_c parameters.
# def test_gradients_through_branch_local_token_builders(device: torch.device) -> None:
#     B, N, d, K = 3, 16, 32, 4
#     metadata = make_fake_metadata(N)
#     batch = make_fake_batch(B, N, metadata, device)
#     cluster_assignments = torch.arange(N) % K

#     model = ClinicalClusterDownstreamModel(
#         cluster_assignments=cluster_assignments,
#         num_clusters=K,
#         name_embeddings=metadata["name_embeddings"],
#         feature_type_ids=metadata["feature_type_ids"],
#         categorical_cardinalities=metadata["categorical_cardinalities"],
#         continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
#         missing_reason_cardinality=metadata["missing_reason_cardinality"],
#         d_model=d,
#         expert_n_heads=4,
#         expert_n_layers=1,
#         fusion_n_heads=4,
#         fusion_n_layers=1,
#         dropout=0.0,
#         use_cluster_embedding=True,
#     ).to(device)

#     model.train()

#     out = model(batch=batch)
#     loss = (
#         out["tyg"].pow(2).mean()
#         + out["ckm_logits"].pow(2).mean()
#         + out["binary_logits"].pow(2).mean()
#     )
#     loss.backward()

#     # Check one branch-local token-builder parameter.
#     grad = model.encoder.expert_bank.branches[0].token_builder.name_projection.weight.grad
#     assert grad is not None
#     assert torch.isfinite(grad).all()
#     assert grad.abs().sum().item() > 0

#     # Check expert CLS also receives gradient.
#     cls_grad = model.encoder.expert_bank.branches[0].expert.cls_token.grad
#     assert cls_grad is not None
#     assert torch.isfinite(cls_grad).all()
#     assert cls_grad.abs().sum().item() > 0

def test_feature_context_codes_change_tokens(device):
    B, N, d = 2, 3, 16

    # feature 0 has 2 contexts, feature 1 has 1, feature 2 has 2
    name_embeddings = torch.randn(5, 8)
    feature_context_offsets = torch.tensor([0, 2, 3])

    feature_type_ids = torch.tensor([
        int(FeatureType.NUMERICAL),
        int(FeatureType.CATEGORICAL),
        int(FeatureType.NUMERICAL),
    ])

    builder = FeatureTokenBuilder(
        name_embeddings=name_embeddings,
        feature_context_offsets=feature_context_offsets,
        feature_type_ids=feature_type_ids,
        categorical_cardinalities=[1, 4, 1],
        continuous_bin_cardinality=11,
        missing_reason_cardinality=10,
        token_dim=d,
        dropout=0.0,
    ).to(device)

    batch = {
        "numeric_values": torch.randn(B, N, device=device),
        "continuous_bin_codes": torch.zeros(B, N, dtype=torch.long, device=device),
        "categorical_codes": torch.zeros(B, N, dtype=torch.long, device=device),
        "missing_reason_codes": torch.zeros(B, N, dtype=torch.long, device=device),
        "missing_mask": torch.zeros(B, N, dtype=torch.bool, device=device),
        "feature_context_codes": torch.tensor([[0, 0, 0], [1, 0, 1]], device=device),
    }

    z = builder(batch)
    assert z.shape == (B, N, d)

def test_multiple_summary_tokens_and_attention_diagnostics(device: torch.device) -> None:
    """M>1 should produce separate group tokens and valid attention diagnostics."""
    B, N, d, K, M = 4, 21, 32, 3, 2
    metadata = make_fake_metadata(N)
    batch = make_fake_batch(B, N, metadata, device)
    cluster_assignments = torch.arange(N) % K

    bank = ExpertBank(
        cluster_assignments=cluster_assignments,
        num_clusters=K,
        name_embeddings=metadata["name_embeddings"],
        feature_type_ids=metadata["feature_type_ids"],
        categorical_cardinalities=metadata["categorical_cardinalities"],
        continuous_bin_cardinality=metadata["continuous_bin_cardinality"],
        missing_reason_cardinality=metadata["missing_reason_cardinality"],
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        num_summary_tokens=M,
    ).to(device)
    bank.eval()

    # A single normal initialization call samples each slot independently.
    cls = bank.branches[0].expert.cls_token.detach()
    assert cls.shape == (1, M, d)
    assert not torch.equal(cls[:, 0], cls[:, 1])

    feature_available = torch.ones(B, N, dtype=torch.bool, device=device)
    with torch.no_grad():
        group_tokens, cluster_available, attention_overlap = bank(
            batch,
            feature_available_mask=feature_available,
            return_attention_overlap=True,
        )

    assert group_tokens.shape == (B, K, M, d), group_tokens.shape
    assert cluster_available.shape == (B, K), cluster_available.shape
    assert attention_overlap.shape == (B, K, M, M), attention_overlap.shape
    assert torch.isfinite(attention_overlap).all()
    assert torch.allclose(
        attention_overlap[..., 0, 0],
        torch.ones_like(attention_overlap[..., 0, 0]),
        atol=1e-5,
    )

    fusion = PatientFusionTransformer(
        num_clusters=K,
        d_model=d,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        num_summary_tokens=M,
    ).to(device)
    fusion.eval()
    with torch.no_grad():
        patient_embedding, fused = fusion(group_tokens, cluster_available)
    assert patient_embedding.shape == (B, d)
    assert fused.shape == (B, K, M, d)


def run_test(name: str, fn: Callable[[torch.device], None], device: torch.device) -> bool:
    print(cyan(f"RUN  {name}"))
    try:
        fn(device)
    except Exception as exc:
        print(red(f"FAIL {name}: {type(exc).__name__}: {exc}"))
        print(red(traceback.format_exc()))
        return False

    print(green(f"PASS {name}"))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    args = parser.parse_args()

    global USE_COLOR
    USE_COLOR = not args.no_color

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    tests: list[tuple[str, Callable[[torch.device], None]]] = [
        ("TransformerSetEncoder", test_transformer_set_encoder),
        ("ClusterExpertEncoder", test_cluster_expert_encoder),
        ("ClusterBranch", test_cluster_branch),
        ("EmptyClusterBranch", test_empty_cluster_branch),
        ("ExpertBank", test_expert_bank),
        ("ExpertBankEmptyHighIndexCluster", test_expert_bank_with_empty_schema_cluster),
        ("ExpertBankEmptyMiddleClusters", test_expert_bank_with_empty_middle_schema_clusters),
        ("NoClusterIDRemapping", test_no_cluster_id_remapping),
        ("InvalidNumClustersRaises", test_invalid_num_clusters_raises),
        ("ClusterAvailableMaskUnderSchemaDropout", test_cluster_available_mask_under_partial_schema_dropout),
        ("PatientFusionTransformer", test_patient_fusion_transformer),
        ("PatientFusionWithoutClusterEmbedding", test_patient_fusion_without_cluster_embedding),
        ("DownstreamPredictionHead", test_downstream_prediction_head),
        # ("FullEncoderAndModel", test_full_encoder_and_model),
        # ("CheckpointSaveLoadRoundtrip", test_checkpoint_save_load_roundtrip),
        # ("GradientsThroughBranchLocalTokenBuilders", test_gradients_through_branch_local_token_builders),
        ("MultipleSummaryTokensAndDiagnostics", test_multiple_summary_tokens_and_attention_diagnostics),
    ]

    passed = 0
    failed = 0

    print(cyan(f"\nRunning {len(tests)} model tests...\n"))

    for name, fn in tests:
        ok = run_test(name, fn, device)
        if ok:
            passed += 1
        else:
            failed += 1
        print()

    total = passed + failed
    if failed == 0:
        print(green(f"SUMMARY: {passed}/{total} tests passed, {failed} failed."))
        print(green("All model.py component tests passed."))
    else:
        print(red(f"SUMMARY: {passed}/{total} tests passed, {failed} failed."))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
