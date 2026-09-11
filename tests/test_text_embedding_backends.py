import json
import sys
import types

import torch

from src.tokenizer import embed_category_values, embed_feature_names


class FakeSentenceTransformer:
    instances = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.max_seq_length = None
        self.device = None
        FakeSentenceTransformer.instances.append(self)

    def to(self, device: str) -> None:
        self.device = device

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_tensor: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        device: str,
    ) -> torch.Tensor:
        assert convert_to_tensor is True
        assert normalize_embeddings is False
        assert show_progress_bar is True
        assert device == self.device
        values = torch.arange(len(texts) * 3, dtype=torch.float32).reshape(len(texts), 3)
        return values + float(batch_size)


def install_fake_sentence_transformers(monkeypatch) -> None:
    FakeSentenceTransformer.instances.clear()
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_feature_embedding_main_writes_sentence_transformer_payload(tmp_path, monkeypatch) -> None:
    install_fake_sentence_transformers(monkeypatch)
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    (token_dir / "feature_semantics.json").write_text(
        json.dumps(
            {
                "first": {"contexts": [{"context_code": 0, "feature_semantic": "First feature."}]},
                "second": {"contexts": [{"context_code": 0, "feature_semantic": "Second feature."}]},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "feature_names": ["second", "first"],
            "feature_context_texts": ["Second feature.", "First feature."],
        },
        token_dir / "tokenizer_metadata.pt",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embed_feature_names",
            "--token-dir",
            str(token_dir),
            "--model-name",
            "FremyCompany/BioLORD-2023",
            "--batch-size",
            "7",
            "--max-length",
            "128",
            "--device",
            "cpu",
        ],
    )

    embed_feature_names.main()

    payload = torch.load(token_dir / "feature_bge_embeddings.pt", map_location="cpu", weights_only=True)
    assert payload["backend"] == "sentence-transformers"
    assert payload["model_name"] == "FremyCompany/BioLORD-2023"
    assert payload["feature_names"] == ["second", "first"]
    assert payload["feature_context_texts"] == ["Second feature.", "First feature."]
    assert payload["embeddings"].shape == (2, 3)
    assert FakeSentenceTransformer.instances[0].max_seq_length == 128


def test_category_embedding_main_writes_sentence_transformer_payload(tmp_path, monkeypatch) -> None:
    install_fake_sentence_transformers(monkeypatch)
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    torch.save({"feature_names": ["question", "numeric"]}, token_dir / "tokenizer_metadata.pt")
    (token_dir / "category_value_texts.json").write_text(
        json.dumps(
            {
                "question": {
                    "0": ["Question: First?\nAnswer: Yes.", "Question: First?\nAnswer: No."],
                    "1": ["Question: Second?\nAnswer: Later."],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embed_category_values",
            "--token-dir",
            str(token_dir),
            "--model-name",
            "FremyCompany/BioLORD-2023",
            "--batch-size",
            "5",
            "--device",
            "cpu",
        ],
    )

    embed_category_values.main()

    payload = torch.load(token_dir / "category_bge_embeddings.pt", map_location="cpu", weights_only=True)
    assert payload["backend"] == "sentence-transformers"
    assert payload["model_name"] == "FremyCompany/BioLORD-2023"
    assert payload["feature_names"] == ["question", "numeric"]
    assert set(payload["embeddings"]) == {0}
    assert payload["embeddings"][0].shape == (3, 3)
