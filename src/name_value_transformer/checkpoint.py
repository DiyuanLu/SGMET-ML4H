from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.tokenizer.dataset import load_tokenizer_metadata

from .data import FlatModelInputs, build_flat_model_inputs, validate_checkpoint_feature_view
from .model import (
    CHECKPOINT_VERSION,
    BinaryPredictionHead,
    FeatureTokenTransformer,
)


def architecture_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("Checkpoint is missing the flat-transformer architecture metadata.")
    return {
        "d_model": int(architecture["d_model"]),
        "n_heads": int(architecture["n_heads"]),
        "n_layers": int(architecture["n_layers"]),
        "dim_feedforward": int(architecture["dim_feedforward"]),
        "dropout": float(architecture["dropout"]),
    }


def validate_flat_checkpoint(checkpoint: Mapping[str, Any], expected_type: str | None = None) -> None:
    if int(checkpoint.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported name_value_transformer checkpoint. Only the flat v2 architecture is supported; "
            "legacy checkpoints must be retrained."
        )
    checkpoint_type = str(checkpoint.get("checkpoint_type", ""))
    if expected_type is not None and checkpoint_type != expected_type:
        raise ValueError(f"Expected checkpoint_type={expected_type!r}, got {checkpoint_type!r}.")


def load_flat_encoder_and_head(
    checkpoint_path: str | Path,
    *,
    token_dir: str | Path,
    cluster_csv: str | Path,
    device: torch.device,
) -> tuple[
    FeatureTokenTransformer,
    BinaryPredictionHead,
    dict[str, Any],
    FlatModelInputs,
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a dictionary checkpoint.")
    validate_flat_checkpoint(checkpoint, "flat_supervised")
    metadata = load_tokenizer_metadata(token_dir)
    ignored = tuple(int(value) for value in checkpoint.get("ignored_cluster_ids", (7,)))
    inputs = build_flat_model_inputs(
        token_dir,
        cluster_csv,
        metadata,
        ignored_cluster_ids=ignored,
    )
    validate_checkpoint_feature_view(checkpoint, inputs.feature_view)
    architecture = architecture_from_checkpoint(checkpoint)
    missing_cardinality = int(checkpoint["model_missing_reason_cardinality"])
    encoder = FeatureTokenTransformer(
        **inputs.model_kwargs(missing_reason_cardinality=missing_cardinality),
        **architecture,
    ).to(device)
    target_columns = [str(value) for value in checkpoint["target_columns"]]
    head = BinaryPredictionHead(
        d_model=architecture["d_model"],
        n_binary_targets=len(target_columns),
        dropout=architecture["dropout"],
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    encoder.eval()
    head.eval()
    return encoder, head, checkpoint, inputs


def checkpoint_architecture(
    encoder: FeatureTokenTransformer,
) -> dict[str, int | float]:
    return {
        "d_model": encoder.d_model,
        "n_heads": encoder.n_heads,
        "n_layers": encoder.n_layers,
        "dim_feedforward": encoder.dim_feedforward,
        "dropout": encoder.dropout,
    }
