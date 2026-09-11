from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.name_value_transformer.checkpoint import validate_flat_checkpoint
from src.name_value_transformer.data import (
    build_flat_model_inputs,
    load_leiden_feature_view,
    select_batch_features,
)
from src.name_value_transformer.model import (
    BinaryPredictionHead,
    FeatureTokenTransformer,
    parameter_counts,
)


def tiny_batch(batch_size: int = 3, n_features: int = 4) -> dict[str, torch.Tensor]:
    return {
        "numeric_values": torch.randn(batch_size, n_features),
        "continuous_bin_codes": torch.randint(0, 4, (batch_size, n_features)),
        "categorical_codes": torch.tensor([[0, 1, 0, 2]] * batch_size),
        "feature_context_codes": torch.zeros(batch_size, n_features, dtype=torch.long),
        "missing_reason_codes": torch.zeros(batch_size, n_features, dtype=torch.long),
        "missing_mask": torch.zeros(batch_size, n_features, dtype=torch.bool),
        "observed_mask": torch.ones(batch_size, n_features, dtype=torch.bool),
    }


def tiny_model(*, categorical_weights: bool = False, dropout: float = 0.0) -> FeatureTokenTransformer:
    weights = None
    if categorical_weights:
        weights = {1: torch.randn(3, 12), 3: torch.randn(4, 12)}
    return FeatureTokenTransformer(
        name_embeddings=torch.randn(4, 8),
        feature_type_ids=torch.tensor([0, 1, 0, 1]),
        categorical_cardinalities=[1, 3, 1, 4],
        continuous_bin_cardinality=4,
        missing_reason_cardinality=5,
        d_model=16,
        n_heads=4,
        n_layers=2,
        dim_feedforward=64,
        dropout=dropout,
        categorical_embedding_weights=weights,
    )


def test_flat_transformer_outputs_patient_and_all_feature_tokens() -> None:
    model = tiny_model()
    output = model(tiny_batch())

    assert output.patient_embedding.shape == (3, 16)
    assert output.token_embeddings.shape == (3, 4, 16)
    assert output.feature_available_mask.shape == (3, 4)
    assert torch.isfinite(output.patient_embedding).all()


def test_flat_transformer_has_no_cluster_expert_or_fusion_modules() -> None:
    model = tiny_model()
    module_names = {name for name, _ in model.named_modules()}

    assert not any("cluster" in name for name in module_names)
    assert not any("expert" in name for name in module_names)
    assert not any("fusion" in name for name in module_names)
    assert hasattr(model, "token_builder")
    assert hasattr(model, "encoder")


def test_feature_availability_mask_is_accepted_with_all_features_unavailable() -> None:
    model = tiny_model()
    batch = tiny_batch()
    available = torch.zeros(3, 4, dtype=torch.bool)

    output = model(batch, available)

    assert torch.equal(output.feature_available_mask, available)
    assert torch.isfinite(output.patient_embedding).all()


def test_biolord_category_tables_are_frozen_but_projection_is_trainable() -> None:
    model = tiny_model(categorical_weights=True)

    assert all(not embedding.weight.requires_grad for embedding in model.token_builder.category_embeddings)
    assert all(parameter.requires_grad for parameter in model.token_builder.category_projection.parameters())


def test_supervised_loss_backpropagates_through_the_complete_flat_encoder() -> None:
    model = tiny_model(categorical_weights=True)
    head = BinaryPredictionHead(d_model=16, n_binary_targets=11, dropout=0.0)
    output = model(tiny_batch())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head(output.patient_embedding)["binary_logits"],
        torch.randint(0, 2, (3, 11)).float(),
    )

    loss.backward()

    assert model.cls_token.grad is not None
    assert model.token_builder.name_projection.weight.grad is not None
    for layer in model.encoder.encoder.layers:
        assert layer.self_attn.in_proj_weight.grad is not None


def test_leiden_view_is_ordered_by_tokenizer_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "leiden.csv"
    pd.DataFrame(
        {"feature_name": ["c", "a", "b", "d"], "cluster_id": [7, 0, 2, 7]}
    ).to_csv(csv_path, index=False)

    view = load_leiden_feature_view(csv_path, ["a", "b", "c", "d"])

    assert view.feature_names == ("a", "b")
    assert view.global_indices.tolist() == [0, 1]
    assert view.n_global_features == 4


def test_batch_selection_happens_before_the_model(tmp_path: Path) -> None:
    batch = tiny_batch(batch_size=2, n_features=4)
    frame = pd.DataFrame(
        {"feature_name": ["a", "b", "c", "d"], "cluster_id": [0, 7, 1, 7]}
    )
    path = tmp_path / "leiden.csv"
    frame.to_csv(path, index=False)
    view = load_leiden_feature_view(path, ["a", "b", "c", "d"])
    selected = select_batch_features(batch, view)

    assert selected["numeric_values"].shape == (2, 2)
    assert torch.equal(selected["numeric_values"], batch["numeric_values"][:, [0, 2]])


def test_legacy_checkpoint_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy checkpoints must be retrained"):
        validate_flat_checkpoint({"model_state_dict": {}}, "flat_supervised")


def test_v4_parameter_parity_and_feature_view() -> None:
    token_dir = Path("data/processed/name_value_tokens/nhanes_2011_2023_v4")
    cluster_csv = token_dir / "feature_clusters_biolord_v4_k8_leiden.csv"
    if not (token_dir / "tokenizer_metadata.pt").exists() or not cluster_csv.exists():
        pytest.skip("v4 BioLORD/Leiden artifacts are not available")
    from src.tokenizer.dataset import load_tokenizer_metadata

    metadata = load_tokenizer_metadata(token_dir)
    inputs = build_flat_model_inputs(token_dir, cluster_csv, metadata)
    encoder = FeatureTokenTransformer(
        **inputs.model_kwargs(missing_reason_cardinality=inputs.missing_reason_cardinality + 1)
    )
    head = BinaryPredictionHead(d_model=160, n_binary_targets=11)
    counts = parameter_counts(encoder, head)

    assert inputs.feature_view.n_features == 161
    assert counts["active_total"] == 2_051_947
    assert counts["active_trainable"] == 1_593_451
    assert counts["active_frozen"] == 458_496
    assert abs(counts["active_total"] - 2_053_707) / 2_053_707 < 0.001
