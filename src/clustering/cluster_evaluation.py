import argparse
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


SERIALIZED_CLUSTER_METHODS = ("agglomerative-cosine", "kmeans")


def cluster_serialized_embeddings(
    embeddings: np.ndarray,
    *,
    num_clusters: int,
    method: str,
    random_state: int,
) -> np.ndarray:
    if len(embeddings) < num_clusters:
        raise ValueError(
            f"Need at least {num_clusters} rows to create fixed serialized clusters; "
            f"got {len(embeddings)}."
        )
    if method == "agglomerative-cosine":
        clusterer = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="cosine",
            linkage="average",
        )
    elif method == "kmeans":
        clusterer = KMeans(n_clusters=num_clusters, random_state=random_state, n_init="auto")
    else:
        raise ValueError(f"Unsupported serialized cluster method: {method}")
    return np.asarray(clusterer.fit_predict(embeddings), dtype=int)


def write_feature_cluster_assignments(
    df: pd.DataFrame,
    labels: np.ndarray,
    output_path: str,
) -> None:
    if "feature_name" not in df.columns:
        raise ValueError("Expected column 'feature_name' to export feature cluster assignments.")
    labels = np.asarray(labels, dtype=int)
    if (labels < 0).any():
        raise ValueError("Serialized feature cluster assignments must be non-negative.")
    out = pd.DataFrame(
        {
            "feature_name": df["feature_name"].astype(str),
            "cluster_id": labels,
        }
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)


def print_cluster_report(
    label_name: str,
    labels: np.ndarray,
    embeddings: np.ndarray,
    n_rows: int,
    *,
    has_noise: bool,
) -> None:
    if has_noise:
        valid_mask = labels != -1
        score_embeddings = embeddings[valid_mask]
        score_labels = labels[valid_mask]
        score_suffix = ", Excluding Noise"
    else:
        score_embeddings = embeddings
        score_labels = labels
        score_suffix = ""

    if np.unique(score_labels).size > 1:
        sil_score = silhouette_score(score_embeddings, score_labels)
        print(f"Silhouette Score ({label_name}{score_suffix}): {sil_score:.4f}")
    else:
        print(f"Not enough distinct clusters formed to calculate a Silhouette Score ({label_name}).")

    print(f"Total Features ({label_name}): {n_rows}")
    if has_noise:
        noise_count = int((labels == -1).sum())
        print(
            f"Valid Clusters Created ({label_name}): "
            f"{len(set(labels)) - (1 if -1 in labels else 0)}"
        )
        print(
            f"Noise Features Quarantined ({label_name}): "
            f"{noise_count} ({noise_count / n_rows * 100:.1f}%)"
        )
    else:
        print(f"Clusters Created ({label_name}): {len(set(labels))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster NHANES feature text embeddings.")
    parser.add_argument(
        "--mode",
        choices=["serialized", "legacy"],
        default="serialized",
        help="Cluster serialized feature semantics by default, or legacy normalized/raw embeddings.",
    )
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="Minimum cluster size for HDBSCAN.")
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=12,
        help="Number of fixed clusters for serialized feature semantics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for KMeans serialized clustering.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=SERIALIZED_CLUSTER_METHODS,
        default="agglomerative-cosine",
        help="Fixed-k clustering method for serialized feature semantics.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="CSV aligned to the embeddings with an embedding_index column.",
    )
    parser.add_argument(
        "--normalized-embeddings",
        default=None,
        help=".npy embeddings file for normalized embeddings.",
    )
    parser.add_argument(
        "--raw-embeddings",
        default=None,
        help=".npy embeddings file for raw embeddings.",
    )
    parser.add_argument(
        "--serialized-embeddings",
        default=None,
        help=".npy embeddings file for serialized feature semantics.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV to write cluster labels.",
    )
    parser.add_argument(
        "--feature-clusters-csv",
        default=None,
        help="Output CSV with feature_name,cluster_id for clinical cluster experts.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    serialized_embeddings_path = args.serialized_embeddings or os.path.join(
        repo_root, "data", "processed", "nhanes_feature_semantics_embeddings.npy"
    )
    normalized_embeddings_path = args.normalized_embeddings or os.path.join(
        repo_root, "data", "processed", "nhanes_normalized_embeddings.npy"
    )
    raw_embeddings_path = args.raw_embeddings or os.path.join(
        repo_root, "data", "processed", "nhanes_embeddings.npy"
    )
    input_csv = args.input_csv or os.path.join(
        repo_root,
        "data",
        "processed",
        "nhanes_feature_semantics_with_embeddings.csv"
        if args.mode == "serialized"
        else "nhanes_normalized_features_with_embeddings.csv",
    )
    output_csv = args.output_csv or os.path.join(
        repo_root,
        "data",
        "processed",
        "nhanes_feature_semantics_with_clusters.csv"
        if args.mode == "serialized"
        else "nhanes_normalized_features_with_clusters.csv",
    )
    feature_clusters_csv = args.feature_clusters_csv or os.path.join(
        repo_root, "data", "processed", "feature_clusters.csv"
    )

    df = pd.read_csv(input_csv)
    if "embedding_index" not in df.columns:
        raise ValueError("Expected column 'embedding_index' in input CSV.")

    if args.mode == "serialized":
        serialized_embeddings = np.load(serialized_embeddings_path)
        if len(serialized_embeddings) != len(df):
            raise ValueError("Serialized embeddings array length does not match the input CSV length.")
        serialized_labels = cluster_serialized_embeddings(
            serialized_embeddings,
            num_clusters=args.num_clusters,
            method=args.cluster_method,
            random_state=args.random_state,
        )
        df["cluster_label_serialized"] = serialized_labels
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        write_feature_cluster_assignments(df, serialized_labels, feature_clusters_csv)
        print_cluster_report(
            f"serialized/{args.cluster_method}",
            serialized_labels,
            serialized_embeddings,
            len(df),
            has_noise=False,
        )
        print(f"Wrote clustered metadata to {output_csv}")
        print(f"Wrote feature cluster assignments to {feature_clusters_csv}")
    else:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=2,
            cluster_selection_epsilon=0.45,
            metric='euclidean'
        )
        normalized_embeddings = np.load(normalized_embeddings_path)
        if len(normalized_embeddings) != len(df):
            raise ValueError("Normalized embeddings array length does not match the input CSV length.")

        raw_embeddings = np.load(raw_embeddings_path)
        if len(raw_embeddings) != len(df):
            raise ValueError("Raw embeddings array length does not match the input CSV length.")

        normalized_labels = clusterer.fit_predict(normalized_embeddings)
        raw_labels = clusterer.fit_predict(raw_embeddings)

        df["cluster_label_normalized"] = normalized_labels
        df["cluster_label_raw"] = raw_labels
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

        for label_name, labels, embeddings in (
            ("normalized", normalized_labels, normalized_embeddings),
            ("raw", raw_labels, raw_embeddings),
        ):
            print_cluster_report(label_name, labels, embeddings, len(df), has_noise=True)

        print(f"Wrote clustered metadata to {output_csv}")
