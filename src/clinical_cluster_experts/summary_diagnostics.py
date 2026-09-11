from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch


class SummaryTokenDiagnosticsAccumulator:
    """Streaming dataset-level diagnostics for expert feature-group (summary) tokens.

    The accumulator stores only sufficient statistics on CPU, not patient-level embeddings. 
    It therefore works for complete validation/test sets without increasing GPU memory usage.

    Reported statistics:
        - mean L2 norm for each cluster/summary slot;
        - mean per-dimension variance across patients for each slot;
        - mean pairwise cosine similarity between summary slots;
        - mean pairwise overlap of final-layer CLS-to-feature attention.

    Only patients for which the cluster is available contribute. 
    Attention overlap additionally excludes rows where the expert had fewer than two available feature tokens; the model marks those rows as NaN.
    """

    def __init__(self, *, num_clusters: int, num_summary_tokens: int, d_model: int):
        self.num_clusters = int(num_clusters)
        self.num_summary_tokens = int(num_summary_tokens)
        self.d_model = int(d_model)
        if self.num_clusters < 1:
            raise ValueError("num_clusters must be >= 1.")
        if self.num_summary_tokens < 1:
            raise ValueError("num_summary_tokens must be >= 1.")
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1.")

        K, M, D = self.num_clusters, self.num_summary_tokens, self.d_model
        self.token_count = torch.zeros(K, M, dtype=torch.float64)
        self.token_sum = torch.zeros(K, M, D, dtype=torch.float64)
        self.token_sq_sum = torch.zeros(K, M, D, dtype=torch.float64)
        self.norm_sum = torch.zeros(K, M, dtype=torch.float64)

        self.cosine_count = torch.zeros(K, M, M, dtype=torch.float64)
        self.cosine_sum = torch.zeros(K, M, M, dtype=torch.float64)
        self.attention_count = torch.zeros(K, M, M, dtype=torch.float64)
        self.attention_sum = torch.zeros(K, M, M, dtype=torch.float64)

    def update(
        self,
        *,
        group_tokens: torch.Tensor,
        cluster_available_mask: torch.Tensor,
        attention_overlap: torch.Tensor | None = None,
    ) -> None:
        """Add one batch of model outputs.

        Args:
            group_tokens: ``[B, K, d]`` for one summary token or
                ``[B, K, M, d]`` for multiple summary tokens.
            cluster_available_mask: bool tensor ``[B, K]``.
            attention_overlap: optional ``[B, K, M, M]`` matrix returned by
                ``ClinicalClusterEncoder(..., return_summary_attention=True)``.
        """
        if group_tokens.ndim == 3:
            group_tokens = group_tokens.unsqueeze(2)
        if group_tokens.ndim != 4:
            raise ValueError(
                "group_tokens must have shape [B, K, d] or [B, K, M, d], "
                f"got {tuple(group_tokens.shape)}."
            )

        B, K, M, D = group_tokens.shape
        if (K, M, D) != (
            self.num_clusters,
            self.num_summary_tokens,
            self.d_model,
        ):
            raise ValueError(
                f"Unexpected group token shape {tuple(group_tokens.shape)}; "
                f"expected [B, {self.num_clusters}, "
                f"{self.num_summary_tokens}, {self.d_model}]."
            )
        if cluster_available_mask.shape != (B, K):
            raise ValueError(
                f"cluster_available_mask must have shape {(B, K)}, "
                f"got {tuple(cluster_available_mask.shape)}."
            )
        if attention_overlap is not None and attention_overlap.shape != (B, K, M, M):
            raise ValueError(
                f"attention_overlap must have shape {(B, K, M, M)}, "
                f"got {tuple(attention_overlap.shape)}."
            )

        tokens = group_tokens.detach().cpu().double()
        available = cluster_available_mask.detach().to(device="cpu", dtype=torch.bool)
        attention = (
            attention_overlap.detach().detach().cpu().double()
            if attention_overlap is not None
            else None
        )

        for cluster_id in range(K):
            valid = available[:, cluster_id]
            if not valid.any():
                continue
            x = tokens[valid, cluster_id]  # [n, M, D]
            n = float(x.shape[0])
            self.token_count[cluster_id] += n
            self.token_sum[cluster_id] += x.sum(dim=0)
            self.token_sq_sum[cluster_id] += x.square().sum(dim=0)
            self.norm_sum[cluster_id] += torch.linalg.vector_norm(
                x, ord=2, dim=-1
            ).sum(dim=0)

            normalized = x / torch.linalg.vector_norm(
                x, ord=2, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            cosine = torch.matmul(normalized, normalized.transpose(1, 2))
            self.cosine_sum[cluster_id] += cosine.sum(dim=0)
            self.cosine_count[cluster_id] += n

            if attention is not None:
                a = attention[valid, cluster_id]  # [n, M, M]
                finite = torch.isfinite(a)
                self.attention_sum[cluster_id] += torch.where(
                    finite, a, torch.zeros_like(a)
                ).sum(dim=0)
                self.attention_count[cluster_id] += finite.sum(dim=0)

    def compute(self) -> dict[str, list[dict[str, Any]]]:
        """Return tidy token-level and pair-level diagnostic rows."""
        token_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []

        for cluster_id in range(self.num_clusters):
            for slot in range(self.num_summary_tokens):
                count = float(self.token_count[cluster_id, slot].item())
                if count > 0:
                    mean = self.token_sum[cluster_id, slot] / count
                    variance = (
                        self.token_sq_sum[cluster_id, slot] / count
                        - mean.square()
                    ).clamp_min(0.0)
                    norm_mean = float(
                        self.norm_sum[cluster_id, slot].item() / count
                    )
                    variance_mean = float(variance.mean().item())
                else:
                    norm_mean = float("nan")
                    variance_mean = float("nan")

                token_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "summary_slot": slot + 1,
                        "n_patients": int(count),
                        "norm_mean": norm_mean,
                        "variance_mean_across_dimensions": variance_mean,
                    }
                )

            for slot_a in range(self.num_summary_tokens):
                for slot_b in range(slot_a + 1, self.num_summary_tokens):
                    cosine_n = float(
                        self.cosine_count[cluster_id, slot_a, slot_b].item()
                    )
                    attention_n = float(
                        self.attention_count[cluster_id, slot_a, slot_b].item()
                    )
                    pair_rows.append(
                        {
                            "cluster_id": cluster_id,
                            "summary_slot_a": slot_a + 1,
                            "summary_slot_b": slot_b + 1,
                            "n_patients_cosine": int(cosine_n),
                            "cosine_similarity_mean": (
                                float(
                                    self.cosine_sum[
                                        cluster_id, slot_a, slot_b
                                    ].item()
                                    / cosine_n
                                )
                                if cosine_n > 0
                                else float("nan")
                            ),
                            "n_patients_attention": int(attention_n),
                            "attention_overlap_mean": (
                                float(
                                    self.attention_sum[
                                        cluster_id, slot_a, slot_b
                                    ].item()
                                    / attention_n
                                )
                                if attention_n > 0
                                else float("nan")
                            ),
                        }
                    )

        return {"token_rows": token_rows, "pair_rows": pair_rows}


def append_summary_diagnostics_csv(
    diagnostics: dict[str, list[dict[str, Any]]],
    *,
    output_dir: str | Path,
    epoch: int | None = None,
    split: str | None = None,
    prefix: str = "summary_token",
) -> tuple[Path, Path]:
    """Append tidy diagnostics to two CSV files and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / f"{prefix}_token_stats.csv"
    pair_path = output_dir / f"{prefix}_pair_stats.csv"

    def _write(path: Path, rows: list[dict[str, Any]]) -> None:
        enriched = []
        for row in rows:
            item: dict[str, Any] = {}
            if epoch is not None:
                item["epoch"] = int(epoch)
            if split is not None:
                item["split"] = str(split)
            item.update(row)
            enriched.append(item)
        if not enriched:
            return
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(enriched)

    _write(token_path, diagnostics.get("token_rows", []))
    _write(pair_path, diagnostics.get("pair_rows", []))
    return token_path, pair_path


def log_summary_diagnostics_tensorboard(
    writer,
    diagnostics: dict[str, list[dict[str, Any]]],
    *,
    step: int,
    split: str = "val",
) -> None:
    """Write the tidy diagnostics to TensorBoard when a writer is available."""
    if writer is None:
        return
    for row in diagnostics.get("token_rows", []):
        cluster_id = int(row["cluster_id"])
        slot = int(row["summary_slot"])
        base = f"summary_tokens/{split}/cluster_{cluster_id}/slot_{slot}"
        writer.add_scalar(f"{base}/norm_mean", row["norm_mean"], step)
        writer.add_scalar(
            f"{base}/variance_mean",
            row["variance_mean_across_dimensions"],
            step,
        )
    for row in diagnostics.get("pair_rows", []):
        cluster_id = int(row["cluster_id"])
        slot_a = int(row["summary_slot_a"])
        slot_b = int(row["summary_slot_b"])
        base = (
            f"summary_tokens/{split}/cluster_{cluster_id}/"
            f"slots_{slot_a}_{slot_b}"
        )
        writer.add_scalar(
            f"{base}/cosine_similarity_mean",
            row["cosine_similarity_mean"],
            step,
        )
        writer.add_scalar(
            f"{base}/attention_overlap_mean",
            row["attention_overlap_mean"],
            step,
        )
