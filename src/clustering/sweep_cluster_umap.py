"""Sweep fixed-K feature clustering and write UMAP visualizations per K."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering.cluster_evaluation import (
    SERIALIZED_CLUSTER_METHODS,
    cluster_serialized_embeddings,
    print_cluster_report,
    write_feature_cluster_assignments,
)


DEFAULT_K_LIST = (8, 10, 12, 16)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def k_artifact_paths(root: Path, k: int) -> dict[str, Path]:
    processed = root / "data" / "processed"
    results = root / "data" / "results"
    return {
        "clustered_csv": processed / f"nhanes_feature_semantics_with_clusters_k{k}.csv",
        "feature_clusters_csv": processed / f"feature_clusters_k{k}.csv",
        "labeled_csv": processed / f"nhanes_feature_semantics_labeled_clusters_k{k}.csv",
        "umap_html": results / f"nhanes_feature_semantics_umap_k{k}.html",
        "umap_png": results / f"nhanes_feature_semantics_umap_k{k}.png",
    }


def require_existing(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}. "
            "Build feature semantics and embeddings before running the sweep."
        )
    return path


def run_sweep_for_k(
    *,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    num_clusters: int,
    cluster_method: str,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
    n_callouts: int,
    paths: dict[str, Path],
    sleep_seconds: float,
) -> None:
    from src.clustering.label_clusters import label_dataframe_clusters
    from src.clustering.umap_visual import visualize_clusters

    print(f"\n===== Clustering with K={num_clusters} ({cluster_method}) =====")
    labels = cluster_serialized_embeddings(
        embeddings,
        num_clusters=num_clusters,
        method=cluster_method,
        random_state=random_state,
    )
    clustered = df.copy()
    clustered["cluster_label_serialized"] = labels

    paths["clustered_csv"].parent.mkdir(parents=True, exist_ok=True)
    paths["feature_clusters_csv"].parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(paths["clustered_csv"], index=False)
    write_feature_cluster_assignments(clustered, labels, str(paths["feature_clusters_csv"]))
    print_cluster_report(
        f"serialized/{cluster_method}/k{num_clusters}",
        labels,
        embeddings,
        len(clustered),
        has_noise=False,
    )
    print(f"Wrote clustered metadata to {paths['clustered_csv']}")
    print(f"Wrote feature cluster assignments to {paths['feature_clusters_csv']}")

    labeled, mappings = label_dataframe_clusters(
        clustered,
        sleep_seconds=sleep_seconds,
    )
    paths["labeled_csv"].parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(paths["labeled_csv"], index=False)
    print(f"Wrote labeled clusters to {paths['labeled_csv']}")
    print("--- Categories ---")
    for cluster_id, name in sorted(mappings.get("cluster_label_serialized", {}).items()):
        print(f"  Cluster {cluster_id}: {name}")

    hover_name = "feature_name" if "feature_name" in labeled.columns else None
    paths["umap_html"].parent.mkdir(parents=True, exist_ok=True)
    visualize_clusters(
        labeled,
        embeddings,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        output_html=str(paths["umap_html"]),
        output_png=str(paths["umap_png"]),
        label_column="Human_Readable_Category_Serialized",
        title=f"Serialized Feature Semantics K={num_clusters} (UMAP 2D Projection)",
        hover_name=hover_name,
        n_callouts=n_callouts,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep fixed-K serialized feature clustering and write UMAP HTML/PNG per K."
    )
    parser.add_argument(
        "--num-clusters-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_LIST),
        help=f"Cluster counts to sweep (default: {' '.join(map(str, DEFAULT_K_LIST))}).",
    )
    parser.add_argument(
        "--cluster-method",
        choices=SERIALIZED_CLUSTER_METHODS,
        default="agglomerative-cosine",
        help="Fixed-k clustering method for serialized feature semantics.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--n-callouts", type=int, default=3)
    parser.add_argument(
        "--label-sleep-seconds",
        type=float,
        default=1.0,
        help="Pause between Gemini labeling calls.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="CSV aligned to embeddings (defaults to nhanes_feature_semantics_with_embeddings.csv).",
    )
    parser.add_argument(
        "--embeddings-npy",
        default=None,
        help="Serialized feature embeddings .npy aligned to the CSV.",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    input_csv = Path(
        args.input_csv
        or root / "data" / "processed" / "nhanes_feature_semantics_with_embeddings.csv"
    )
    embeddings_path = Path(
        args.embeddings_npy
        or root / "data" / "processed" / "nhanes_feature_semantics_embeddings.npy"
    )
    require_existing(input_csv, "feature semantics CSV")
    require_existing(embeddings_path, "feature embeddings")

    df = pd.read_csv(input_csv)
    embeddings = np.load(embeddings_path)
    if len(embeddings) != len(df):
        raise ValueError(
            f"Embeddings length ({len(embeddings)}) does not match CSV rows ({len(df)})."
        )

    for k in args.num_clusters_list:
        if k < 2:
            raise ValueError(f"num_clusters must be >= 2; got {k}.")
        run_sweep_for_k(
            df=df,
            embeddings=embeddings,
            num_clusters=k,
            cluster_method=args.cluster_method,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_callouts=args.n_callouts,
            paths=k_artifact_paths(root, k),
            sleep_seconds=args.label_sleep_seconds,
        )

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
