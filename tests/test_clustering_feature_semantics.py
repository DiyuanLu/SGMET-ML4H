import json

import pandas as pd
import torch

from src.clustering.build_feature_semantics import (
    build_feature_semantics_frame,
    select_latest_context,
)
from src.clustering.cluster_evaluation import (
    cluster_serialized_embeddings,
    write_feature_cluster_assignments,
)


def test_select_latest_context_prefers_latest_known_cycle() -> None:
    entry = {
        "contexts": [
            {"context_code": 0, "feature_semantic": "older text"},
            {"context_code": 1, "feature_semantic": "newer text"},
        ]
    }

    cycle, context_code, text = select_latest_context(
        feature_name="ALQ101",
        semantic_entry=entry,
        cycle_map={"2011-2012": 0, "2021-2023": 1, "2017-2018": 0},
    )

    assert cycle == "2021-2023"
    assert context_code == 1
    assert text == "newer text"


def test_select_latest_context_handles_missing_cycle_map() -> None:
    entry = {
        "contexts": [
            {"context_code": 0, "feature_semantic": "older text"},
            {"context_code": 2, "feature_semantic": "fallback latest text"},
        ]
    }

    assert select_latest_context(
        feature_name="UNKNOWN",
        semantic_entry=entry,
        cycle_map=None,
    ) == ("", 2, "fallback latest text")


def test_build_feature_semantics_frame_preserves_tokenizer_feature_order(tmp_path) -> None:
    token_dir = tmp_path / "tokens"
    token_dir.mkdir()
    (token_dir / "feature_semantics.json").write_text(
        json.dumps(
            {
                "B": {
                    "contexts": [
                        {"context_code": 0, "feature_semantic": "B old"},
                        {"context_code": 1, "feature_semantic": "B new"},
                    ]
                },
                "A": {
                    "contexts": [
                        {"context_code": 0, "feature_semantic": "A only"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "feature_names": ["A", "B"],
            "feature_context_cycle_maps": {
                "A": {"2013-2014": 0},
                "B": {"2011-2012": 0, "2021-2023": 1},
            },
        },
        token_dir / "tokenizer_metadata.pt",
    )

    frame = build_feature_semantics_frame(token_dir)

    assert frame["feature_name"].tolist() == ["A", "B"]
    assert frame["feature_idx"].tolist() == [0, 1]
    assert frame["selected_cycle"].tolist() == ["2013-2014", "2021-2023"]
    assert frame["serialized_feature_text"].tolist() == ["A only", "B new"]


def test_feature_cluster_export_requires_non_negative_clusters(tmp_path) -> None:
    df = pd.DataFrame({"feature_name": ["A", "B", "C"]})
    output = tmp_path / "feature_clusters.csv"

    write_feature_cluster_assignments(df, labels=pd.Series([2, 1, 0]).to_numpy(), output_path=str(output))

    exported = pd.read_csv(output)
    assert exported.to_dict("list") == {
        "feature_name": ["A", "B", "C"],
        "cluster_id": [2, 1, 0],
    }

    try:
        write_feature_cluster_assignments(df, labels=pd.Series([2, -1, 0]).to_numpy(), output_path=str(output))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative cluster labels to fail.")


def test_agglomerative_cosine_can_create_exact_requested_cluster_count() -> None:
    eye = torch.eye(12, dtype=torch.float32)
    embeddings = eye.repeat_interleave(2, dim=0).numpy()

    labels = cluster_serialized_embeddings(
        embeddings,
        num_clusters=12,
        method="agglomerative-cosine",
        random_state=42,
    )

    assert sorted(set(labels.tolist())) == list(range(12))


def test_kmeans_serialized_clustering_remains_available() -> None:
    centers = torch.arange(12, dtype=torch.float32).view(12, 1).repeat_interleave(2, dim=0).numpy()
    embeddings = torch.cat(
        [
            torch.from_numpy(centers),
            torch.zeros((24, 1)),
        ],
        dim=1,
    ).numpy()

    labels = cluster_serialized_embeddings(
        embeddings,
        num_clusters=12,
        method="kmeans",
        random_state=42,
    )

    assert sorted(set(labels.tolist())) == list(range(12))
