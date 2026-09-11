from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm


DEFAULT_MODEL = "FremyCompany/BioLORD-2023"
DEFAULT_BACKEND = "sentence-transformers"


def default_model_name() -> str:
    return DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode feature descriptions with a frozen text embedding model.")
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=None,
        help="Directory produced by tokenize_nhanes.py. Defaults input/output paths inside this directory.",
    )
    parser.add_argument(
        "--feature-texts",
        type=Path,
        default=None,
        help="Feature text JSON produced by tokenize_nhanes.py. Legacy blank-line .txt files are also supported.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Output .pt path. Stores embeddings plus optional alignment metadata.",
    )
    parser.add_argument(
        "--tokenizer-metadata",
        type=Path,
        default=None,
        help="Optional tokenizer_metadata.pt with feature_names and feature_texts for alignment checks.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=default_model_name(),
        help="Text embedding model id or local model path.",
    )
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "transformers"],
        default=DEFAULT_BACKEND,
        help="Text embedding backend. BioLORD uses sentence-transformers.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="How to pool token embeddings for the transformers backend. Ignored by sentence-transformers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Defaults to cuda if available, else cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_texts_path, tokenizer_metadata_path, out_file = resolve_paths(args)
    feature_names: list[str] | None = None
    metadata_feature_texts: list[str] = []
    if tokenizer_metadata_path is not None:
        metadata = torch.load(tokenizer_metadata_path, map_location="cpu", weights_only=True)
        feature_names = list(metadata["feature_names"])
        metadata_feature_texts = list(metadata.get("feature_context_texts", metadata.get("feature_texts", [])))
    texts = read_feature_texts(feature_texts_path, feature_names=feature_names)
    if not texts:
        raise ValueError(f"No feature descriptions found in {feature_texts_path}.")

    device = resolve_device(args.device)
    embedding_tensor = encode_texts(
        texts,
        backend=args.backend,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        progress_desc="encoding features",
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload: torch.Tensor | dict[str, object] = embedding_tensor
    if feature_names is not None:
        if len(texts) != embedding_tensor.shape[0]:
            raise ValueError(
                f"Tokenizer metadata has {len(texts)} feature contexts, but encoded "
                f"{embedding_tensor.shape[0]} feature texts."
            )
        if metadata_feature_texts and metadata_feature_texts != texts:
            raise ValueError("Feature text file does not match tokenizer_metadata.pt feature_texts.")
        payload = {
            "backend": args.backend,
            "model_name": args.model_name,
            "feature_names": feature_names,
            "feature_texts": texts,
            "feature_context_texts": texts,
            "embeddings": embedding_tensor,
        }
    torch.save(payload, out_file)
    print(f"[OK] saved {tuple(embedding_tensor.shape)} embeddings to {out_file}")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    feature_texts = args.feature_texts
    tokenizer_metadata = args.tokenizer_metadata
    out_file = args.out_file
    if args.token_dir is not None:
        feature_texts = feature_texts or args.token_dir / "feature_semantics.json"
        tokenizer_metadata = tokenizer_metadata or args.token_dir / "tokenizer_metadata.pt"
        out_file = out_file or args.token_dir / "feature_bge_embeddings.pt"
    if feature_texts is None:
        raise ValueError("Provide --feature-texts or --token-dir.")
    if out_file is None:
        raise ValueError("Provide --out-file or --token-dir.")
    return feature_texts, tokenizer_metadata, out_file


def read_feature_texts(path: Path, feature_names: list[str] | None = None) -> list[str]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError(f"Feature text JSON must contain an object keyed by feature name: {path}")
        if feature_names is not None:
            missing = [feature_name for feature_name in feature_names if feature_name not in payload]
            extra = sorted(set(payload) - set(feature_names))
            if missing:
                raise ValueError(f"Feature text JSON is missing feature keys: {missing[:10]}")
            if extra:
                raise ValueError(f"Feature text JSON contains unknown feature keys: {extra[:10]}")
            ordered_names = feature_names
        else:
            ordered_names = list(payload)
        texts = []
        for feature_name in ordered_names:
            spec = payload[feature_name]
            if not isinstance(spec, dict):
                raise ValueError(f"Feature text entry for {feature_name} must be an object.")
            contexts = spec.get("contexts")
            if isinstance(contexts, list):
                ordered_contexts = sorted(contexts, key=lambda item: int(item["context_code"]))
                for context in ordered_contexts:
                    feature_semantic = context.get("feature_semantic")
                    if not isinstance(feature_semantic, str) or not feature_semantic.strip():
                        raise ValueError(f"Feature context entry for {feature_name} is missing feature_semantic.")
                    texts.append(feature_semantic.strip())
                continue
            feature_semantic = spec.get("feature_semantic")
            if not isinstance(feature_semantic, str) or not feature_semantic.strip():
                raise ValueError(f"Feature text entry for {feature_name} is missing feature_semantic.")
            texts.append(feature_semantic.strip())
        return texts
    return [block.strip() for block in content.split("\n\n") if block.strip()]


def load_transformers_model(model_name: str):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Feature-name embedding requires transformers. Install/update the environment with:\n"
            "  conda env update -n di-lab -f environment.yml --prune"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name)
    except ValueError as exc:
        if "torch.load" in str(exc) and "torch to at least v2.6" in str(exc):
            raise RuntimeError(
                "Transformers refused to load this model's .bin weights with torch<2.6 because of "
                "CVE-2025-32434. Update the environment, then rerun:\n"
                "  conda env update -n di-lab -f environment.yml --prune\n"
                "  conda activate di-lab\n"
                "  python -c \"import torch; print(torch.__version__)\"\n"
                "The printed torch version must be >=2.6. Alternatively, choose a Hugging Face "
                "model that provides safetensors weights."
            ) from exc
        raise
    return tokenizer, model


def resolve_device(device: str | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def encode_texts(
    texts: list[str],
    *,
    backend: str = DEFAULT_BACKEND,
    model_name: str = DEFAULT_MODEL,
    device: torch.device | str | None = None,
    batch_size: int = 16,
    max_length: int = 256,
    pooling: str = "cls",
    progress_desc: str = "encoding texts",
) -> torch.Tensor:
    if not texts:
        raise ValueError("No texts provided for embedding.")
    resolved_device = resolve_device(str(device) if device is not None else None)
    if backend == "sentence-transformers":
        return encode_sentence_transformer_texts(
            texts,
            model_name=model_name,
            device=resolved_device,
            batch_size=batch_size,
            max_length=max_length,
            progress_desc=progress_desc,
        )
    if backend == "transformers":
        return encode_transformers_texts(
            texts,
            model_name=model_name,
            device=resolved_device,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
            progress_desc=progress_desc,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")


def encode_sentence_transformer_texts(
    texts: list[str],
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    progress_desc: str,
) -> torch.Tensor:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "SentenceTransformer embedding requires sentence-transformers. "
            "Install/update the environment with:\n"
            "  conda env update -n di-lab -f environment.yml --prune"
        ) from exc

    model = SentenceTransformer(model_name)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length
    if hasattr(model, "to"):
        model.to(str(device))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=True,
        device=str(device),
    )
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)
    return embeddings.detach().cpu().float().contiguous()


def encode_transformers_texts(
    texts: list[str],
    *,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
    progress_desc: str,
) -> torch.Tensor:
    tokenizer, model = load_transformers_model(model_name)
    model.to(device)
    model.eval()

    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=progress_desc):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            pooled = pool_output(output.last_hidden_state, encoded["attention_mask"], pooling)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).float().contiguous()


def pool_output(hidden_states: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden_states[:, 0, :]
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


if __name__ == "__main__":
    main()
