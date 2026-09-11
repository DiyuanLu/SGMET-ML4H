""" Example usage:
python -m src.baseline_models.prepare_xgboost_dataset \
  --df-path data/processed/nhanes_2011_2023.parquet \
  --token-dir data/processed/tokenized_nhanes_v4 \
  --x-out-path data/processed/xgboost_X_v4.parquet \
  --y-out-path data/processed/xgboost_Y_v4.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SPLITS = ["train", "val", "test"]

def load_pt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create XGBoost-ready X and Y parquet files from NHANES data and tokenized targets.")
    parser.add_argument("--df-path", type=Path, required=True, help="Path to processed NHANES dataframe, e.g. data/processed/nhanes_2011_2023.parquet")
    parser.add_argument("--token-dir", type=Path, required=True, help="Path to tokenized_nhanes_v4 folder.")
    parser.add_argument("--x-out-path", type=Path, required=True, help="Output path for X parquet.")
    parser.add_argument("--y-out-path", type=Path, required=True, help="Output path for Y parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.df_path)
    meta = load_pt(args.token_dir / "tokenizer_metadata.pt")

    feature_names = list(meta["feature_names"])
    target_columns = list(meta["target_columns"])

    print("Number of X features:", len(feature_names))
    print("Number of Y targets:", len(target_columns))
    print("Targets:", target_columns)
    assert "split" in df.columns, "df must contain a split column."

    missing_features = [c for c in feature_names if c not in df.columns]
    assert not missing_features, f"Missing feature columns in df: {missing_features}"

    x_parts = []
    y_parts = []

    for split in SPLITS:
        split_df = df[df["split"] == split].copy()
        targets = load_pt(args.token_dir / f"{split}_targets.pt")
        tokens = load_pt(args.token_dir / f"{split}_tokens.pt")
        assert len(split_df) == len(next(iter(targets.values()))), (f"{split}: df rows and target rows do not match.")

        # Sanity check: row/order alignment using missing pattern.
        raw_missing = split_df[feature_names].isna().to_numpy()
        token_missing = tokens["missing_mask"].cpu().numpy()
        agreement = np.mean(raw_missing == token_missing)

        print(f"{split}: rows={len(split_df)}, missing-mask agreement={agreement:.6f}")

        X_split = split_df[feature_names].copy()
        X_split["split"] = split

        Y_split = pd.DataFrame(index=split_df.index)
        for target in target_columns:
            Y_split[target] = pd.Series(targets[target], index=split_df.index, dtype="Int64")
        Y_split["split"] = split

        x_parts.append(X_split)
        y_parts.append(Y_split)

    X = pd.concat(x_parts, axis=0).reset_index(drop=True)
    Y = pd.concat(y_parts, axis=0).reset_index(drop=True)

    # Final column order
    X = X[feature_names + ["split"]]
    Y = Y[target_columns + ["split"]]

    assert len(X) == len(Y), "X and Y row counts do not match."
    assert X["split"].equals(Y["split"]), "X and Y split columns do not align."

    args.x_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.y_out_path.parent.mkdir(parents=True, exist_ok=True)

    X.to_parquet(args.x_out_path, index=False)
    Y.to_parquet(args.y_out_path, index=False)

    print("\nSaved:")
    print("X:", args.x_out_path, X.shape)
    print("Y:", args.y_out_path, Y.shape)
    print("\nX split counts:")
    print(X["split"].value_counts())
    print("\nY target missing counts:")
    print(Y[target_columns].isna().sum())


if __name__ == "__main__":
    main()