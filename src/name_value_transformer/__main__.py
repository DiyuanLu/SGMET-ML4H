from __future__ import annotations

import torch

from .model import BinaryPredictionHead, FeatureTokenTransformer, parameter_counts


def main() -> None:
    batch_size, n_features, d_model = 3, 8, 160
    feature_types = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    model = FeatureTokenTransformer(
        name_embeddings=torch.randn(n_features, 32),
        feature_type_ids=feature_types,
        categorical_cardinalities=[1, 4, 1, 3, 1, 5, 1, 2],
        continuous_bin_cardinality=11,
        missing_reason_cardinality=11,
        d_model=d_model,
        n_heads=4,
        n_layers=4,
        dim_feedforward=640,
    )
    head = BinaryPredictionHead(d_model=d_model, n_binary_targets=11)
    batch = {
        "numeric_values": torch.randn(batch_size, n_features),
        "continuous_bin_codes": torch.randint(0, 11, (batch_size, n_features)),
        "categorical_codes": torch.zeros(batch_size, n_features, dtype=torch.long),
        "feature_context_codes": torch.zeros(batch_size, n_features, dtype=torch.long),
        "missing_reason_codes": torch.zeros(batch_size, n_features, dtype=torch.long),
        "missing_mask": torch.zeros(batch_size, n_features, dtype=torch.bool),
        "observed_mask": torch.ones(batch_size, n_features, dtype=torch.bool),
    }
    encoded = model(batch)
    logits = head(encoded.patient_embedding)["binary_logits"]
    print("Flat name-value Transformer smoke test OK")
    print(f"patient_embedding={tuple(encoded.patient_embedding.shape)}")
    print(f"feature_embeddings={tuple(encoded.token_embeddings.shape)}")
    print(f"binary_logits={tuple(logits.shape)}")
    print(f"parameters={parameter_counts(model, head)}")


if __name__ == "__main__":
    main()
