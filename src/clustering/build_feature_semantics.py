from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


CYCLE_ORDER = [
    "2011-2012",
    "2013-2014",
    "2015-2016",
    "2017-2018",
    "2019-2020",
    "2021-2023",
]


def context_by_code(entry: dict[str, Any]) -> dict[int, str]:
    contexts = entry.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("feature_semantics entry must contain a non-empty contexts list.")
    by_code: dict[int, str] = {}
    for context in contexts:
        code = int(context["context_code"])
        text = str(context["feature_semantic"]).strip()
        if not text:
            raise ValueError(f"Empty feature_semantic for context {code}.")
        by_code[code] = text
    return by_code


def select_latest_context(
    *,
    feature_name: str,
    semantic_entry: dict[str, Any],
    cycle_map: dict[str, int] | None,
) -> tuple[str, int, str]:
    contexts = context_by_code(semantic_entry)
    if cycle_map:
        known_cycles = [cycle for cycle in CYCLE_ORDER if cycle in cycle_map]
        if known_cycles:
            selected_cycle = known_cycles[-1]
            selected_context_code = int(cycle_map[selected_cycle])
        else:
            selected_cycle, raw_code = sorted(cycle_map.items())[-1]
            selected_context_code = int(raw_code)
        if selected_context_code not in contexts:
            raise ValueError(
                f"Cycle map for {feature_name} selects missing context_code={selected_context_code}."
            )
        return selected_cycle, selected_context_code, contexts[selected_context_code]

    selected_context_code = max(contexts)
    return "", selected_context_code, contexts[selected_context_code]


def build_feature_semantics_frame(token_dir: Path) -> pd.DataFrame:
    feature_semantics_path = token_dir / "feature_semantics.json"
    metadata_path = token_dir / "tokenizer_metadata.pt"
    if not feature_semantics_path.exists():
        raise FileNotFoundError(f"Missing feature semantics JSON: {feature_semantics_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing tokenizer metadata: {metadata_path}")

    feature_semantics = json.loads(feature_semantics_path.read_text(encoding="utf-8"))
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    feature_names = list(metadata["feature_names"])
    cycle_maps = metadata.get("feature_context_cycle_maps", {})

    rows = []
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_name not in feature_semantics:
            raise ValueError(f"Feature {feature_name} is missing from {feature_semantics_path}.")
        entry = feature_semantics[feature_name]
        selected_cycle, selected_context_code, text = select_latest_context(
            feature_name=feature_name,
            semantic_entry=entry,
            cycle_map=cycle_maps.get(feature_name),
        )
        rows.append(
            {
                "feature_idx": feature_idx,
                "feature_name": feature_name,
                "selected_cycle": selected_cycle,
                "selected_context_code": selected_context_code,
                "n_contexts": len(context_by_code(entry)),
                "serialized_feature_text": text,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature-level clustering input from tokenizer feature_semantics.json."
    )
    parser.add_argument(
        "--token-dir",
        type=Path,
        default=Path("data/processed/name_value_tokens/nhanes_2011_2023_v2"),
        help="Token directory produced by src.tokenizer.tokenize_nhanes.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/nhanes_feature_semantics_for_clustering.csv"),
        help="Output CSV with one latest-cycle serialized text row per feature.",
    )
    args = parser.parse_args()

    df = build_feature_semantics_frame(args.token_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote feature semantics clustering input to {args.output_csv} rows={len(df)}")


if __name__ == "__main__":
    main()
