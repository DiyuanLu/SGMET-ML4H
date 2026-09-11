#!/usr/bin/env python3
"""
196 - Build cluster maps over the ACTIVE feature set, with the reconstruction-validity gate applied.

Writes NEW files only (suffix `_act<N>`); nothing in data/cluster_maps from the earlier 172-feature
study is modified or removed, because the frozen results reference those maps and the feature-ablation
arm needs the old convention.

WHAT IS DIFFERENT FROM 170
--------------------------
1. Excluded features are removed BEFORE embedding, so they never influence the kNN graph or Leiden.
   (In 170 every feature was clustered, so dead features occupied cluster slots and pulled centroids.)

2. A validity gate is applied after partitioning. The original reason for dropping cluster 7 in the
   earlier setup was that it has no context/target for masked value reconstruction pretraining. That generalises,
   but cluster SIZE is not a sufficient test: the pretrainer only emits a target when a cluster has >=2
   features WITH VALUES for that patient (the `keep_one_observed_value_visible` guard at
   pretrain_expert_reconstruction.py:297). Measured on the old maps, a cluster could hold 11 features
   and still reach 0.0% of patients.

       valid iff  size >= 2  AND  at least one training patient has >=2 valued features

   Invalid clusters are MERGED into their nearest valid cluster by embedding cosine, iterating to a
   fixed point. Merging rather than ignoring keeps every feature (only absolute constant features are
   removed) and keeps feature content identical at every grid point, which is what makes a
   granularity comparison valid at all.

3. Per-cluster viability is REPORTED, not gated. An earlier draft gated at ">=20% of patients"; that was
   withdrawn as arbitrary after a sensitivity check showed the achievable cluster count moves from 21 to
   12 purely as a function of that threshold, and moves as much again with the kNN parameter. So the
   only gate is "can this expert receive a gradient at all", and viability ships as metadata.

4. Excluded + team-ignored features are appended as a JUNK cluster with id = achieved K, because
   load_cluster_assignments() requires the CSV to list every tokenizer feature. Runs then pass
   --num-clusters K+1 --ignore-clusters K; inactive clusters get zero placeholders and
   cluster_available_mask=False (model.py: forward_active_clusters), so fusion ignores them.

  python scripts/196_build_active_maps.py --targets 7 10 15 20 25 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[7, 10, 15, 20, 25, 30])
    ap.add_argument("--knn", type=int, default=17, help="17 = the original V4 setting, inherited not chosen")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--token-dir", type=Path, required=True)
    ap.add_argument("--feature-embeddings", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=HERE / "cluster_maps")
    ap.add_argument(
        "--feature-set",
        type=Path,
        default=HERE / "config/active_149.json",
        help="Feature-set JSON. It may contain active_features directly or "
             "base_feature_set plus exclude_additional.",
    )
    ap.add_argument(
        "--resolution-override",
        action="append",
        default=[],
        metavar="TARGET=RESOLUTION",
        help="Use an explicit Leiden resolution for a target count when the non-monotonic "
             "community-count curve makes binary search skip a narrow valid interval.",
    )
    ap.add_argument("--no-merge", action="store_true",
                    help="Emit the RAW Leiden partition at the exact target K, skipping the validity "
                         "merge. Singleton clusters survive. Legal for end-to-end-from-scratch only, "
                         "where experts are trained on the labels and a 1-feature expert still receives "
                         "gradients; such maps are NOT pretrainable.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolution_overrides = {}
    for item in args.resolution_override:
        target_text, resolution_text = item.split("=", 1)
        resolution_overrides[int(target_text)] = float(resolution_text)

    import igraph as ig
    import leidenalg

    spec_path = args.feature_set.resolve()
    spec = json.loads(spec_path.read_text())
    if "active_features" in spec:
        active = list(spec["active_features"])
    else:
        base = json.loads((spec_path.parent / spec["base_feature_set"]).read_text())
        excluded = set(spec.get("exclude_additional", []))
        active = [name for name in base["active_features"] if name not in excluded]
    if len(active) != int(spec["n_active"]):
        raise ValueError(
            f"{spec_path.name}: derived {len(active)} active features, expected {spec['n_active']}"
        )
    md = torch.load(args.token_dir / "tokenizer_metadata.pt", map_location="cpu", weights_only=False)
    names: list[str] = list(md["feature_names"])
    unknown = sorted(set(active) - set(names))
    if unknown:
        raise ValueError(f"{spec_path.name}: features absent from tokenizer metadata: {unknown}")
    inactive = [n for n in names if n not in set(active)]
    print(f"[features] {len(names)} total -> {len(active)} active, {len(inactive)} inactive (junk cluster)")

    # candidate mask: a feature is a possible reconstruction target for a patient when it is
    # schema-available AND actually has a value
    tk = torch.load(args.token_dir / "train_tokens.pt", map_location="cpu", weights_only=False)
    cand_all = (tk["observed_mask"].bool() & ~tk["missing_mask"].bool()).numpy()
    idx_of = {n: i for i, n in enumerate(names)}
    A = np.array([idx_of[n] for n in active])
    C = cand_all[:, A]
    n_pat = C.shape[0]

    blob = torch.load(args.feature_embeddings, map_location="cpu", weights_only=False)
    X = np.asarray(blob["embeddings"], dtype="float64")
    assert X.shape[0] == len(names), f"embedding rows {X.shape[0]} != {len(names)} features"
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xa = X[A]
    n = len(Xa)

    S = Xa @ Xa.T
    np.fill_diagonal(S, -np.inf)
    edges, wts = {}, {}
    for i in range(n):
        for j in np.argsort(-S[i])[: args.knn]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b and (a, b) not in edges:
                edges[(a, b)] = True
                wts[(a, b)] = float(max(S[i, j], 0.0))
    g = ig.Graph(n=n, edges=list(edges.keys()), directed=False)
    g.es["weight"] = [wts[e] for e in edges]
    print(f"[graph] cosine kNN k={args.knn}: {n} nodes, {g.ecount()} edges")

    def viable_rows(cols: np.ndarray) -> int:
        return int((C[:, cols].sum(1) >= 2).sum())

    def leiden(res: float) -> np.ndarray:
        return np.asarray(
            leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"],
                resolution_parameter=float(res), seed=args.seed, n_iterations=-1,
            ).membership
        )

    def solve(target: int):
        lo, hi, best = 0.05, 60.0, None
        for _ in range(40):
            mid = (lo + hi) / 2
            lab = leiden(mid)
            k = len(set(lab.tolist()))
            if best is None or abs(k - target) < abs(best[1] - target):
                best = (lab, k, mid)
            if k == target:
                return lab, k, mid
            if k < target:
                lo = mid
            else:
                hi = mid
        return best

    def apply_gate(lab: np.ndarray):
        """Merge invalid clusters into the nearest valid one, iterating to a fixed point."""
        lab = lab.copy()
        merges = []
        for _ in range(500):
            vc = pd.Series(lab).value_counts()
            bad, good = [], []
            for c in vc.index:
                cols = np.where(lab == c)[0]
                (bad if (len(cols) < 2 or viable_rows(cols) == 0) else good).append(c)
            if not bad or not good:
                break
            cent = {}
            for c in good:
                v = Xa[lab == c].mean(0)
                cent[c] = v / np.linalg.norm(v)
            c = bad[0]
            v = Xa[lab == c].mean(0)
            v = v / np.linalg.norm(v)
            host = max(good, key=lambda t: float(v @ cent[t]))
            merges.append((int(c), int(host), round(float(v @ cent[host]), 3), int((lab == c).sum())))
            lab[lab == c] = host
        order = pd.Series(lab).value_counts().index.tolist()
        remap = {o: i for i, o in enumerate(order)}
        return np.array([remap[x] for x in lab]), merges

    rows = []
    for target in args.targets:
        if target in resolution_overrides:
            override_resolution = resolution_overrides[target]
            override_labels = leiden(override_resolution)
            override_k = len(set(override_labels.tolist()))
            if override_k != target:
                raise ValueError(
                    f"resolution override {target}={override_resolution} produced K={override_k}"
                )
            got = (override_labels, override_k, override_resolution)
        else:
            got = solve(target)
        if got is None:
            print(f"[k={target}] not reachable, skipped")
            continue
        raw, raw_k, res = got
        raw_sing = int((pd.Series(raw).value_counts() == 1).sum())
        if args.no_merge:
            order = pd.Series(raw).value_counts().index.tolist()
            remap = {o: i for i, o in enumerate(order)}
            lab = np.array([remap[x] for x in raw]); merges = []
        else:
            lab, merges = apply_gate(raw)
        K = len(set(lab.tolist()))

        sizes, viab = [], []
        for c in range(K):
            cols = np.where(lab == c)[0]
            sizes.append(len(cols))
            viab.append(viable_rows(cols) / n_pat)
        if not args.no_merge:
            assert min(sizes) >= 2, "gate failed: a cluster smaller than 2 survived"
            assert min(viab) > 0, "gate failed: a zero-viability cluster survived"

        # write CSV: active features with their cluster, then every inactive feature in junk cluster K
        suffix = "raw" if args.no_merge else ""
        out = args.output_dir / f"feature_clusters_act{len(active)}_t{target}_k{K}{suffix}.csv"
        df = pd.DataFrame(
            {"feature_name": active + inactive,
             "cluster_id": [int(x) for x in lab] + [K] * len(inactive)}
        )
        # keep the tokenizer's feature order so the file is easy to diff against other maps
        df = df.set_index("feature_name").loc[names].reset_index()
        df.to_csv(out, index=False)

        rows.append(dict(target=target, achieved_k=K, resolution=round(res, 4), knn=args.knn,
                         seed=args.seed, raw_k=raw_k, raw_singletons=raw_sing, n_merges=len(merges),
                         min_size=min(sizes), median_size=int(np.median(sizes)), max_size=max(sizes),
                         min_viability=round(min(viab), 4), median_viability=round(float(np.median(viab)), 4),
                         n_below_50pct=int(sum(1 for v in viab if v < 0.5)),
                         junk_cluster_id=K, num_clusters_arg=K + 1, file=out.name))
        print(f"[t={target:>2} -> K={K:>2}] res={res:.3f}  raw_singletons={raw_sing}  merges={len(merges)}  "
              f"sizes {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}  "
              f"viability min {min(viab)*100:.1f}% med {np.median(viab)*100:.1f}%  "
              f"<50%: {sum(1 for v in viab if v < 0.5)}/{K}")
        if merges:
            print(f"                merged: {merges[:4]}{' ...' if len(merges) > 4 else ''}")

    summ = pd.DataFrame(rows)
    sp = args.output_dir / (
        f"active_map_summary_act{len(active)}"
        + ("_raw" if args.no_merge else "")
        + ".csv"
    )
    summ.to_csv(sp, index=False)
    print(f"\n{summ.to_string(index=False)}")
    print(f"\nwrote {len(rows)} maps + {sp.name}")
    dup = summ[summ.duplicated('achieved_k', keep=False)]
    if len(dup):
        print("\nNOTE: these targets collapsed to the same achieved K. Achieved K does not identify a")
        print("partition, so both maps are kept and treated as distinct partitions at that K:")
        print(dup[['target', 'achieved_k', 'resolution', 'file']].to_string(index=False))


if __name__ == "__main__":
    main()
