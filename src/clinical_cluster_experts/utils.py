from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from src.clinical_cluster_experts.model import (
    ClinicalClusterEncoder,
)

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
    if name != "auto":
        raise ValueError(f"Unknown device: {name}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_cluster_assignments(cluster_csv: str | Path, feature_names: list[str]) -> torch.Tensor:
    """Load feature-to-cluster assignments while preserving numeric cluster IDs."""
    cluster_csv = Path(cluster_csv)
    df = pd.read_csv(cluster_csv)

    dups = df["feature_name"][df["feature_name"].duplicated()].unique().tolist()
    if dups:
        raise ValueError(f"Duplicate feature_name entries in cluster CSV: {dups[:10]}")

    required_columns = {"feature_name", "cluster_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Cluster CSV must contain columns {required_columns}. Missing: {missing_columns}")

    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise")
    if not (df["cluster_id"] == df["cluster_id"].astype(int)).all():
        raise ValueError("All cluster_id values must be integers.")
    df["cluster_id"] = df["cluster_id"].astype(int)

    mapping = dict(zip(df["feature_name"], df["cluster_id"]))
    missing_features = [name for name in feature_names if name not in mapping]
    if missing_features:
        raise ValueError(
            "Cluster CSV does not contain all tokenizer features. "
            f"First missing features: {missing_features[:10]}"
        )

    cluster_ids = [int(mapping[name]) for name in feature_names]
    if min(cluster_ids) < 0:
        raise ValueError("cluster_id values must be non-negative.")

    return torch.tensor(cluster_ids, dtype=torch.long)



def flatten_category_texts(texts: object) -> list[str]:
    if isinstance(texts, list):
        return [str(text) for text in texts]

    if isinstance(texts, dict):
        flattened: list[str] = []
        for context_code in sorted(texts, key=lambda value: int(value)):
            values = texts[context_code]
            if isinstance(values, list):
                flattened.extend(str(text) for text in values)
        return flattened

    return []


def load_feature_name_embeddings(
    token_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Load feature/context BGE embeddings and optional feature_context_offsets.

    Expected file:
        feature_bge_embeddings.pt

    Supported payloads:
        Tensor
        or dict with:
            embeddings
            feature_names optional
            feature_texts optional
            feature_context_offsets optional
    """
    path = Path(token_dir) / "feature_bge_embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature name embeddings: {path}.")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        embeddings = payload.get("embeddings")
        feature_context_offsets = payload.get("feature_context_offsets")

        if metadata is not None:
            feature_names = list(metadata.get("feature_names", []))
            feature_texts = list(metadata.get("feature_texts", []))

            saved_feature_names = payload.get("feature_names")
            if saved_feature_names is not None and list(saved_feature_names) != feature_names:
                raise ValueError("Feature embedding feature order does not match tokenizer metadata.")

            saved_feature_texts = payload.get("feature_texts")
            if saved_feature_texts is not None and list(saved_feature_texts) != feature_texts:
                raise ValueError("Feature embedding feature_texts do not match tokenizer metadata.")
    else:
        embeddings = payload
        feature_context_offsets = None

    if not isinstance(embeddings, torch.Tensor):
        raise ValueError(f"feature_bge_embeddings.pt must contain a tensor or dict['embeddings']: {path}")

    if metadata is not None:
        feature_names = list(metadata.get("feature_names", []))
        feature_texts = list(metadata.get("feature_texts", []))

        expected_rows = len(feature_texts) if feature_texts else len(feature_names)
        if expected_rows > 0 and int(embeddings.shape[0]) != expected_rows:
            raise ValueError(
                f"Feature embedding row count mismatch: expected {expected_rows}, "
                f"got {int(embeddings.shape[0])}."
            )
        
    if feature_context_offsets is None and metadata is not None and "feature_context_offsets" in metadata:
        feature_context_offsets = metadata["feature_context_offsets"]

    if feature_context_offsets is not None:
        feature_context_offsets = torch.as_tensor(feature_context_offsets).long()

        if metadata is not None:
            n_features = len(metadata.get("feature_names", []))
            if feature_context_offsets.ndim != 1 or int(feature_context_offsets.shape[0]) != n_features:
                raise ValueError(
                    f"feature_context_offsets must have shape [{n_features}], "
                    f"got {tuple(feature_context_offsets.shape)}."
                )
            if int(feature_context_offsets.min().item()) < 0:
                raise ValueError("feature_context_offsets must be non-negative.")
            if int(feature_context_offsets.max().item()) >= int(embeddings.shape[0]):
                raise ValueError("feature_context_offsets points outside feature embedding rows.")

    return embeddings.float(), feature_context_offsets


def load_category_embedding_weights(
    token_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    required: bool = False,
) -> dict[int, torch.Tensor] | None:
    """
    Load category text embeddings keyed by global feature index.

    Expected file:
        category_bge_embeddings.pt

    Expected payload:
        dict with:
            embeddings: dict[int, Tensor]
            feature_names optional
            category_value_texts optional
    """
    path = Path(token_dir) / "category_bge_embeddings.pt"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing category embedding weights: {path}.")
        return None
    
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError(f"Expected category_bge_embeddings.pt to contain an 'embeddings' dict: {path}")

    raw_embeddings = payload["embeddings"]
    if not isinstance(raw_embeddings, dict):
        raise ValueError(f"'embeddings' must be a dict in {path}")

    if metadata is None:
        return {int(feature_idx): tensor.float() for feature_idx, tensor in raw_embeddings.items()}

    feature_names = list(metadata.get("feature_names", []))
    categorical_cardinalities = [int(x) for x in metadata.get("categorical_cardinalities", [])]
    category_value_texts = dict(metadata.get("category_value_texts", {}))

    if len(categorical_cardinalities) != len(feature_names):
        raise ValueError("metadata['categorical_cardinalities'] must have one entry per feature.")

    saved_feature_names = payload.get("feature_names")
    if saved_feature_names is not None and list(saved_feature_names) != feature_names:
        raise ValueError("Category embedding feature order does not match tokenizer metadata.")

    saved_category_value_texts = payload.get("category_value_texts")
    if saved_category_value_texts is not None:
        for feature_name, texts in category_value_texts.items():
            expected = flatten_category_texts(texts)
            actual = flatten_category_texts(saved_category_value_texts.get(feature_name, []))
            if actual != expected:
                raise ValueError(f"Category embedding text order does not match for feature {feature_name}.")

    weights: dict[int, torch.Tensor] = {}
    for raw_feature_idx, tensor in raw_embeddings.items():
        feature_idx = int(raw_feature_idx)

        # Match teammate behavior: ignore embeddings for features outside current metadata.
        if feature_idx >= len(feature_names):
            continue

        expected_rows = categorical_cardinalities[feature_idx]
        if int(tensor.shape[0]) != expected_rows:
            raise ValueError(
                f"Category embedding row count mismatch for {feature_names[feature_idx]}: "
                f"expected {expected_rows}, got {int(tensor.shape[0])}."
            )

        weights[feature_idx] = tensor.float()

    return weights


def load_embeddings(
    token_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    category_required: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[int, torch.Tensor] | None]:
    name_embeddings, feature_context_offsets = load_feature_name_embeddings(token_dir, metadata)
    category_weights = load_category_embedding_weights(token_dir, metadata, required=category_required)
    return name_embeddings, feature_context_offsets, category_weights




def load_pretrained_branches(
    encoder: ClinicalClusterEncoder,
    branch_checkpoints: dict[int, Path],
    *,
    strict: bool,
) -> None:
    for cluster_id, path in branch_checkpoints.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "branch_state_dict" not in ckpt:
            raise ValueError(f"Expected pretraining checkpoint with branch_state_dict: {path}")
        if int(ckpt.get("cluster_id", cluster_id)) != int(cluster_id):
            raise ValueError(f"Checkpoint cluster_id={ckpt.get('cluster_id')} does not match requested {cluster_id}: {path}")

        branch = encoder.expert_bank.branches[int(cluster_id)]
        if "feature_indices" in ckpt:
            ckpt_idx = ckpt["feature_indices"].cpu().long()
            branch_idx = branch.feature_indices.cpu().long()
            if not torch.equal(ckpt_idx, branch_idx):
                raise ValueError(
                    f"Feature indices mismatch for cluster {cluster_id}. Check cluster CSV/tokenizer consistency."
                )
        branch.load_state_dict(ckpt["branch_state_dict"], strict=strict)
        print(f"[INFO] Loaded pretrained branch {cluster_id} from {path}")