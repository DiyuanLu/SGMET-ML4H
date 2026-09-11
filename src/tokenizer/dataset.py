from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from .preprocessing import TransformedBatch, transformed_batch_from_dict


TOKEN_FIELDS = (
    "numeric_values",
    "continuous_bin_codes",
    "categorical_codes",
    "feature_context_codes",
    "missing_reason_codes",
    "missing_mask",
    "observed_mask",
)


class TokenizedTabularDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper for token directories produced by tokenize_nhanes.py."""

    def __init__(
        self,
        batch: TransformedBatch,
        metadata: Mapping[str, Any],
        split: str | None = None,
    ) -> None:
        self.batch = batch
        self.metadata = dict(metadata)
        self.split = split
        self.feature_names = list(self.metadata.get("feature_names", []))
        n_rows, n_features = self.batch.numeric_values.shape
        if self.feature_names and len(self.feature_names) != n_features:
            raise ValueError(
                f"metadata feature_names has {len(self.feature_names)} entries, "
                f"but token tensors have {n_features} features."
            )
        for field in TOKEN_FIELDS:
            tensor = getattr(self.batch, field)
            if tensor.shape[:2] != (n_rows, n_features):
                raise ValueError(f"{field} must have shape [{n_rows}, {n_features}], got {tuple(tensor.shape)}.")

    @classmethod
    def from_dir(cls, token_dir: Path | str, split: str) -> "TokenizedTabularDataset":
        token_dir = Path(token_dir)
        metadata = load_tokenizer_metadata(token_dir)
        batch = load_tokenized_batch(token_dir, split)
        return cls(batch=batch, metadata=metadata, split=split)

    def __len__(self) -> int:
        return int(self.batch.numeric_values.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "numeric_values": self.batch.numeric_values[index],
            "continuous_bin_codes": self.batch.continuous_bin_codes[index],
            "categorical_codes": self.batch.categorical_codes[index],
            "feature_context_codes": self.batch.feature_context_codes[index],
            "missing_reason_codes": self.batch.missing_reason_codes[index],
            "missing_mask": self.batch.missing_mask[index],
            "observed_mask": self.batch.observed_mask[index],
            "row_idx": torch.tensor(index, dtype=torch.long),
        }

    @property
    def n_features(self) -> int:
        return int(self.batch.numeric_values.shape[1])


def load_tokenizer_metadata(token_dir: Path | str) -> dict[str, Any]:
    metadata_path = Path(token_dir) / "tokenizer_metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing tokenizer metadata: {metadata_path}")
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    if not isinstance(metadata, dict):
        raise ValueError(f"Tokenizer metadata must contain a dict: {metadata_path}")
    return metadata


def load_tokenized_batch(token_dir: Path | str, split: str) -> TransformedBatch:
    path = Path(token_dir) / f"{split}_tokens.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing token tensor file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, TransformedBatch):
        return payload
    if isinstance(payload, dict):
        return transformed_batch_from_dict(payload)
    raise ValueError(f"Token file must contain a tensor dict or TransformedBatch: {path}")


def batch_from_dataset_tensors(payload: Mapping[str, torch.Tensor], device: torch.device | str) -> TransformedBatch:
    return TransformedBatch(
        numeric_values=payload["numeric_values"].to(device),
        continuous_bin_codes=payload["continuous_bin_codes"].to(device),
        categorical_codes=payload["categorical_codes"].to(device),
        feature_context_codes=payload["feature_context_codes"].to(device),
        missing_reason_codes=payload["missing_reason_codes"].to(device),
        missing_mask=payload["missing_mask"].to(device),
        observed_mask=payload["observed_mask"].to(device),
    )
