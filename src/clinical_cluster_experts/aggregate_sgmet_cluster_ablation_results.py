#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

CONDITION_RE = re.compile(r"without_cluster_(\d+)$")


def sample_std(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.std(ddof=1)) if len(s) > 1 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    macro_parts, task_parts = [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or (d.name != "full" and not CONDITION_RE.fullmatch(d.name)):
            continue
        removed = None if d.name == "full" else int(CONDITION_RE.fullmatch(d.name).group(1))
        m = pd.read_csv(d / "macro_metrics_per_fold.csv")
        t = pd.read_csv(d / "per_target_metrics_per_fold.csv")
        m["condition"], m["removed_cluster"] = d.name, removed
        t["condition"], t["removed_cluster"] = d.name, removed
        macro_parts.append(m)
        task_parts.append(t)

    macro = pd.concat(macro_parts, ignore_index=True)
    task = pd.concat(task_parts, ignore_index=True)
    macro.to_csv(out / "macro_metrics_all_conditions_per_fold.csv", index=False)
    task.to_csv(out / "per_target_metrics_all_conditions_per_fold.csv", index=False)

    full_m = macro[macro.condition.eq("full")][["fold", "macro_auroc", "macro_auprc"]].rename(
        columns={"macro_auroc": "full_macro_auroc", "macro_auprc": "full_macro_auprc"}
    )
    rm = macro[~macro.condition.eq("full")].merge(full_m, on="fold", validate="many_to_one")
    rm["delta_macro_auroc"] = rm.full_macro_auroc - rm.macro_auroc
    rm["delta_macro_auprc"] = rm.full_macro_auprc - rm.macro_auprc
    rm.to_csv(out / "macro_paired_deltas_per_fold.csv", index=False)

    full_t = task[task.condition.eq("full")][["fold", "task", "auroc", "auprc"]].rename(
        columns={"auroc": "full_auroc", "auprc": "full_auprc"}
    )
    rt = task[~task.condition.eq("full")].merge(full_t, on=["fold", "task"], validate="many_to_one")
    rt["delta_auroc"] = rt.full_auroc - rt.auroc
    rt["delta_auprc"] = rt.full_auprc - rt.auprc
    rt.to_csv(out / "per_target_paired_deltas_per_fold.csv", index=False)

    ms = rm.groupby(["condition", "removed_cluster"], as_index=False).agg(
        macro_auroc_mean=("macro_auroc", "mean"),
        macro_auroc_std=("macro_auroc", sample_std),
        macro_auprc_mean=("macro_auprc", "mean"),
        macro_auprc_std=("macro_auprc", sample_std),
        delta_macro_auroc_mean=("delta_macro_auroc", "mean"),
        delta_macro_auroc_std=("delta_macro_auroc", sample_std),
        delta_macro_auprc_mean=("delta_macro_auprc", "mean"),
        delta_macro_auprc_std=("delta_macro_auprc", sample_std),
        n_folds=("fold", "nunique"),
    ).sort_values("removed_cluster")
    ms.to_csv(out / "macro_loco_summary.csv", index=False)

    ts = rt.groupby(["task", "condition", "removed_cluster"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", sample_std),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", sample_std),
        delta_auroc_mean=("delta_auroc", "mean"),
        delta_auroc_std=("delta_auroc", sample_std),
        delta_auprc_mean=("delta_auprc", "mean"),
        delta_auprc_std=("delta_auprc", sample_std),
        n_folds=("fold", "nunique"),
    ).sort_values(["task", "removed_cluster"])
    ts.to_csv(out / "per_target_loco_summary.csv", index=False)

    robustness = pd.DataFrame([
        {
            "metric": "macro_auroc",
            "full_mean": full_m.full_macro_auroc.mean(),
            "mean_leave_one_out_score": ms.macro_auroc_mean.mean(),
            "mean_paired_drop": ms.delta_macro_auroc_mean.mean(),
            "max_paired_drop": ms.delta_macro_auroc_mean.max(),
            "worst_leave_one_out_score": ms.macro_auroc_mean.min(),
            "most_critical_cluster": int(ms.loc[ms.delta_macro_auroc_mean.idxmax(), "removed_cluster"]),
        },
        {
            "metric": "macro_auprc",
            "full_mean": full_m.full_macro_auprc.mean(),
            "mean_leave_one_out_score": ms.macro_auprc_mean.mean(),
            "mean_paired_drop": ms.delta_macro_auprc_mean.mean(),
            "max_paired_drop": ms.delta_macro_auprc_mean.max(),
            "worst_leave_one_out_score": ms.macro_auprc_mean.min(),
            "most_critical_cluster": int(ms.loc[ms.delta_macro_auprc_mean.idxmax(), "removed_cluster"]),
        },
    ])
    robustness.to_csv(out / "robustness_summary.csv", index=False)

    print("\n[MACRO LOCO SUMMARY]")
    print(ms.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n[ROBUSTNESS SUMMARY]")
    print(robustness.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
