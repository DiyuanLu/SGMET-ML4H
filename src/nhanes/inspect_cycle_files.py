from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xpt":
        return pd.read_sas(path, format="xport", encoding="utf-8")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        missing_pct = (missing / n * 100.0) if n else 0.0
        nunique = int(s.nunique(dropna=True))
        row: dict[str, object] = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": missing,
            "missing_pct": round(missing_pct, 4),
            "n_unique_non_null": nunique,
        }

        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() > 0:
            row.update(
                {
                    "numeric_mean": float(numeric.mean()),
                    "numeric_std": float(numeric.std()),
                    "numeric_min": float(numeric.min()),
                    "numeric_p25": float(numeric.quantile(0.25)),
                    "numeric_median": float(numeric.median()),
                    "numeric_p75": float(numeric.quantile(0.75)),
                    "numeric_max": float(numeric.max()),
                }
            )
        else:
            top_values = s.value_counts(dropna=True).head(5).to_dict()
            row["top_values"] = str(top_values)

        rows.append(row)
    return pd.DataFrame(rows)


def write_dataset_summary(df: pd.DataFrame, out_txt: Path, source_file: Path) -> None:
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cell_pct = (missing_cells / total_cells * 100.0) if total_cells else 0.0
    lines = [
        f"source_file: {source_file}",
        f"n_rows: {df.shape[0]}",
        f"n_columns: {df.shape[1]}",
        f"total_cells: {total_cells}",
        f"missing_cells: {missing_cells}",
        f"missing_cell_pct: {missing_cell_pct:.4f}",
        "columns:",
    ]
    lines.extend([f"- {c}" for c in df.columns])
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create head(50) previews and per-file exploratory summaries for NHANES cycle files."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=[Path("data/raw/2011-2012"), Path("data/raw_imaging/2011-2012")],
        help="Directories to scan for files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/explore/inspection_2011-2012"),
        help="Output directory for previews and summaries.",
    )
    parser.add_argument(
        "--head-n",
        type=int,
        default=50,
        help="Number of rows to export per file for quick inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    head_dir = args.out_dir / "head"
    profile_dir = args.out_dir / "profiles"
    summary_dir = args.out_dir / "summaries"
    head_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    scanned = 0
    processed = 0
    for in_dir in args.input_dirs:
        if not in_dir.exists():
            print(f"[WARN] Input dir does not exist, skipping: {in_dir}")
            continue
        for path in sorted(in_dir.iterdir()):
            if not path.is_file():
                continue
            scanned += 1
            stem = path.stem
            try:
                df = read_table(path)
            except Exception as e:
                print(f"[WARN] Could not read {path.name}: {e}")
                continue

            processed += 1
            head_out = head_dir / f"{stem}_head{args.head_n}.csv"
            profile_out = profile_dir / f"{stem}_column_profile.csv"
            summary_out = summary_dir / f"{stem}_dataset_summary.txt"

            df.head(args.head_n).to_csv(head_out, index=False)
            build_column_profile(df).to_csv(profile_out, index=False)
            write_dataset_summary(df, summary_out, path)

            print(f"[OK] {path.name}: shape={df.shape}")
            print(f"     head -> {head_out}")
            print(f"     profile -> {profile_out}")
            print(f"     summary -> {summary_out}")

    print(f"[DONE] scanned_files={scanned}, processed_files={processed}")


if __name__ == "__main__":
    main()

