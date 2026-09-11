"""Run with:
python -m src.clinical_cluster_experts.token_builder_test \
  --token-dir data/processed/tokenized_nhanes_v4 \
  --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.tokenizer.dataset import TokenizedTabularDataset
from src.tokenizer.schema import FeatureType
from src.clinical_cluster_experts.token_builder import FeatureTokenBuilder


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if name == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_feature_embeddings(token_dir: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    feature_payload = torch.load(
        token_dir / "feature_bge_embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(feature_payload, dict):
        name_embeddings = feature_payload["embeddings"]
        feature_context_offsets = feature_payload.get("feature_context_offsets")
    else:
        name_embeddings = feature_payload
        feature_context_offsets = None

    if feature_context_offsets is not None:
        feature_context_offsets = torch.as_tensor(feature_context_offsets).long()

    return name_embeddings, feature_context_offsets


def load_category_embeddings(token_dir: Path) -> dict[int, torch.Tensor] | None:
    cat_path = token_dir / "category_bge_embeddings.pt"
    if not cat_path.exists():
        return None

    category_payload = torch.load(
        cat_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(category_payload, dict) or "embeddings" not in category_payload:
        raise ValueError("category_bge_embeddings.pt must be a dict containing key 'embeddings'.")

    return {int(k): v for k, v in category_payload["embeddings"].items()}


def real_data_smoke_test(token_dir: Path, device: torch.device, batch_size: int, token_dim: int) -> None:
    print("\n[TEST 1] Real tokenized data smoke test")

    dataset = TokenizedTabularDataset.from_dir(token_dir, split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    metadata = dataset.metadata

    name_embeddings, feature_context_offsets = load_feature_embeddings(token_dir)

    if feature_context_offsets is None and "feature_context_offsets" in metadata:
        feature_context_offsets = torch.as_tensor(metadata["feature_context_offsets"]).long()

    categorical_embedding_weights = load_category_embeddings(token_dir)

    print(f"token_dir: {token_dir}")
    print(f"batch keys: {list(batch.keys())}")
    print(f"metadata keys: {list(metadata.keys())}")
    print(f"name_embeddings shape: {tuple(name_embeddings.shape)}")
    print(
        "feature_context_offsets:",
        None if feature_context_offsets is None else tuple(feature_context_offsets.shape),
    )
    print("feature_context_codes in batch:", "feature_context_codes" in batch)

    builder = FeatureTokenBuilder(
        name_embeddings=name_embeddings,
        feature_context_offsets=feature_context_offsets,
        feature_type_ids=torch.as_tensor(metadata["feature_type_ids"]).long(),
        categorical_cardinalities=[int(x) for x in metadata["categorical_cardinalities"]],
        continuous_bin_cardinality=int(metadata["continuous_bin_cardinality"]),
        missing_reason_cardinality=int(metadata["missing_reason_cardinality"]),
        token_dim=token_dim,
        dropout=0.0,
        categorical_embedding_weights=categorical_embedding_weights,
    ).to(device)

    if categorical_embedding_weights is not None:
        print("\n[CHECK] categorical lookup table / projection requires_grad")
        for name, p in builder.named_parameters():
            if "category_embeddings" in name or "category_projection" in name:
                print(name, p.requires_grad, tuple(p.shape))

        frozen_lookup_ok = all(
            not emb.weight.requires_grad
            for emb in builder.category_embeddings
        )
        assert frozen_lookup_ok, "Expected all category_embeddings lookup tables to be frozen."

        projection_params = list(builder.category_projection.parameters())
        if projection_params:
            projection_trainable_ok = all(p.requires_grad for p in projection_params)
            assert projection_trainable_ok, "Expected category_projection parameters to be trainable."

    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

    z_all = builder(batch)

    expected_shape = (batch_size, len(metadata["feature_names"]), token_dim)
    print("z_all shape:", tuple(z_all.shape))
    assert z_all.shape == expected_shape, f"Expected {expected_shape}, got {tuple(z_all.shape)}"

    print("PASS real_data_smoke_test")


def synthetic_context_test(device: torch.device) -> None:
    print("\n[TEST 2] Synthetic context-aware token test")

    B = 2
    N = 3
    name_dim = 8
    token_dim = 16

    # Context embedding table:
    # feature 0 has two contexts: rows 0, 1
    # feature 1 has one context:  row 2
    # feature 2 has two contexts: rows 3, 4
    name_embeddings = torch.randn(5, name_dim)
    feature_context_offsets = torch.tensor([0, 2, 3], dtype=torch.long)

    feature_type_ids = torch.tensor(
        [
            int(FeatureType.NUMERICAL),
            int(FeatureType.CATEGORICAL),
            int(FeatureType.NUMERICAL),
        ],
        dtype=torch.long,
    )

    builder = FeatureTokenBuilder(
        name_embeddings=name_embeddings,
        feature_context_offsets=feature_context_offsets,
        feature_type_ids=feature_type_ids,
        categorical_cardinalities=[1, 4, 1],
        continuous_bin_cardinality=11,
        missing_reason_cardinality=10,
        token_dim=token_dim,
        dropout=0.0,
    ).to(device)
    builder.eval()

    # Same values for both patients, but different context codes for features 0 and 2.
    batch = {
        "numeric_values": torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
            ],
            device=device,
        ),
        "continuous_bin_codes": torch.tensor(
            [
                [5, 0, 6],
                [5, 0, 6],
            ],
            dtype=torch.long,
            device=device,
        ),
        "categorical_codes": torch.tensor(
            [
                [0, 2, 0],
                [0, 2, 0],
            ],
            dtype=torch.long,
            device=device,
        ),
        "missing_reason_codes": torch.zeros(B, N, dtype=torch.long, device=device),
        "missing_mask": torch.zeros(B, N, dtype=torch.bool, device=device),
        "feature_context_codes": torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 1],
            ],
            dtype=torch.long,
            device=device,
        ),
    }

    z = builder(batch)
    print("synthetic z shape:", tuple(z.shape))
    assert z.shape == (B, N, token_dim)

    # Feature 0 and feature 2 should differ across patients because only context code changed.
    assert not torch.allclose(z[0, 0], z[1, 0]), "Feature 0 token should change with context code."
    assert not torch.allclose(z[0, 2], z[1, 2]), "Feature 2 token should change with context code."

    # Feature 1 has same context code and same categorical value, so it should be identical in eval mode.
    assert torch.allclose(z[0, 1], z[1, 1]), "Feature 1 token should be identical."

    q = builder.metadata_tokens(
        batch_size=B,
        device=device,
        feature_context_codes=batch["feature_context_codes"],
    )
    print("metadata q shape:", tuple(q.shape))
    assert q.shape == (B, N, token_dim)

    print("PASS synthetic_context_test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", type=str, default="data/processed/tokenized_nhanes_v2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--skip-real-data", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if not args.skip_real_data:
        real_data_smoke_test(
            token_dir=Path(args.token_dir),
            device=device,
            batch_size=args.batch_size,
            token_dim=args.token_dim,
        )

    synthetic_context_test(device=device)

    print("\nAll token_builder tests passed.")


if __name__ == "__main__":
    main()