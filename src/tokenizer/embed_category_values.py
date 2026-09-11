from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .embed_feature_names import DEFAULT_BACKEND, encode_texts, default_model_name, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode per-feature category value texts with a frozen text model.")
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=None,
        help="Directory produced by tokenize_nhanes.py. Defaults input/output paths inside this directory.",
    )
    parser.add_argument(
        "--category-value-texts",
        type=Path,
        default=None,
        help="category_value_texts.json produced by train.py or tokenize_nhanes.py.",
    )
    parser.add_argument(
        "--tokenizer-metadata",
        type=Path,
        default=None,
        help="tokenizer_metadata.pt produced by tokenize_nhanes.py, containing feature_names.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Output .pt file with category embeddings keyed by feature index.",
    )
    parser.add_argument("--model-name", type=str, default=default_model_name())
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "transformers"],
        default=DEFAULT_BACKEND,
        help="Text embedding backend. BioLORD uses sentence-transformers.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="How to pool token embeddings for the transformers backend. Ignored by sentence-transformers.",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_value_texts_path, tokenizer_metadata_path, out_file = resolve_paths(args)
    category_texts = json.loads(category_value_texts_path.read_text(encoding="utf-8"))
    metadata = torch.load(tokenizer_metadata_path, map_location="cpu", weights_only=True)
    feature_names = list(metadata["feature_names"])

    device = resolve_device(args.device)

    flat_texts: list[str] = []
    feature_ranges: dict[int, tuple[int, int]] = {}
    for feature_idx, feature_name in enumerate(feature_names):
        texts = flatten_category_texts(category_texts.get(feature_name, []))
        if not texts:
            continue
        start = len(flat_texts)
        flat_texts.extend(texts)
        feature_ranges[feature_idx] = (start, len(flat_texts))
    if not flat_texts:
        raise ValueError(f"No category texts found in {category_value_texts_path}.")

    flat_embeddings = encode_texts(
        flat_texts,
        backend=args.backend,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        progress_desc="encoding category texts",
    )
    embeddings_by_feature = {
        feature_idx: flat_embeddings[start:end].contiguous()
        for feature_idx, (start, end) in feature_ranges.items()
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": args.backend,
            "model_name": args.model_name,
            "feature_names": feature_names,
            "category_value_texts": {feature_name: category_texts.get(feature_name, []) for feature_name in feature_names},
            "embeddings": embeddings_by_feature,
        },
        out_file,
    )
    print(f"[OK] saved category embeddings for {len(embeddings_by_feature)} features to {out_file}")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    category_value_texts = args.category_value_texts
    tokenizer_metadata = args.tokenizer_metadata
    out_file = args.out_file
    if args.token_dir is not None:
        category_value_texts = category_value_texts or args.token_dir / "category_value_texts.json"
        tokenizer_metadata = tokenizer_metadata or args.token_dir / "tokenizer_metadata.pt"
        out_file = out_file or args.token_dir / "category_bge_embeddings.pt"
    if category_value_texts is None:
        raise ValueError("Provide --category-value-texts or --token-dir.")
    if tokenizer_metadata is None:
        raise ValueError("Provide --tokenizer-metadata or --token-dir.")
    if out_file is None:
        raise ValueError("Provide --out-file or --token-dir.")
    return category_value_texts, tokenizer_metadata, out_file


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


if __name__ == "__main__":
    main()
