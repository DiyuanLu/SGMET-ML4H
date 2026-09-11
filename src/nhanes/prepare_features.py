from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.nhanes.constants import CYCLE_INFO, DEFAULT_MODULES, module_file_prefix


SPLIT_SEED = 42

MISSING_REASON_CODES = {
    "not_missing": 0,
    "skipped": 1,
    "item_missing": 2,
    "not_eligible": 3,
    "variable_not_in_cycle": 4,
    "unknown_missing": 5,
    "refused": 6,
    "dont_know": 7,
}

IDENTIFIER_COLUMNS = {"SEQN", "cycle", "SEQN_cycle", "split"}


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport", encoding="utf-8")


def ensure_seqn_str(df: pd.DataFrame) -> pd.DataFrame:
    if "SEQN" not in df.columns:
        raise ValueError("Expected SEQN column for participant-level join.")
    df["SEQN"] = df["SEQN"].astype("Int64").astype(str)
    return df


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, module_name: str) -> pd.DataFrame:
    overlap = [c for c in right.columns if c in left.columns and c != "SEQN"]
    if overlap:
        rename_map = {c: f"{module_name}_{c}" for c in overlap}
        right = right.rename(columns=rename_map)
    return left.merge(right, on="SEQN", how="left")


def module_file_stem(module: str, cycle: str) -> str:
    suffix = CYCLE_INFO[cycle]["suffix"]
    actual_module = module_file_prefix(module, cycle)
    return f"{actual_module}_{suffix}"


def load_cycle_merged(raw_root: Path, cycle: str, modules: list[str]) -> pd.DataFrame:
    cycle_dir = raw_root / cycle
    loaded: list[tuple[str, pd.DataFrame]] = []
    for module in modules:
        stem = module_file_stem(module, cycle)
        xpt_path = cycle_dir / f"{stem}.XPT"
        if not xpt_path.exists():
            print(f"[WARN] Missing file, skipping: {xpt_path}")
            continue
        try:
            df = ensure_seqn_str(read_xpt(xpt_path))
        except Exception as e:
            print(f"[WARN] Could not parse XPT, skipping: {xpt_path} ({e})")
            continue
        if module == "DEMO":
            loaded.insert(0, (module, df))
        else:
            loaded.append((module, df))
        print(f"[OK] Loaded {cycle} {stem}: {df.shape}")

    if not loaded:
        raise RuntimeError(f"No modules loaded for cycle {cycle}. Check downloads.")

    merged = loaded[0][1]
    for module_name, df in loaded[1:]:
        merged = safe_merge(merged, df, module_name)
    merged["cycle"] = cycle
    merged["SEQN_cycle"] = merged["cycle"] + "_" + merged["SEQN"]
    return merged


def read_metadata(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Codebook metadata not found, missing reason columns will use limited fallbacks: {path}")
        return None
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except ImportError as exc:
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            print(f"[WARN] Could not read parquet metadata ({exc}); falling back to {csv_path}")
            return pd.read_csv(csv_path)
        raise


def normalize_code(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value).rstrip("0").rstrip(".")
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return str(int(float(text)))
    return text


def parse_skip_rules(value: object) -> list[dict[str, str]]:
    if pd.isna(value) or value in {"", "[]"}:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_value_labels(value: object) -> dict[str, str]:
    if pd.isna(value) or value == "":
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def nonresponse_code_sets(value_labels: dict[str, str]) -> tuple[set[str], set[str]]:
    refused: set[str] = set()
    dont_know: set[str] = set()
    for code, label in value_labels.items():
        normalized_label = str(label).strip().lower().replace("’", "'")
        normalized_code = normalize_code(code)
        if normalized_code is None:
            continue
        if "refused" in normalized_label:
            refused.add(normalized_code)
        elif "don't know" in normalized_label or "dont know" in normalized_label:
            dont_know.add(normalized_code)
    return refused, dont_know


def code_match_mask(series: pd.Series, codes: set[str]) -> pd.Series:
    if not codes:
        return pd.Series(False, index=series.index)
    normalized = series.map(normalize_code)
    return normalized.isin(codes).fillna(False)


def observed_mask(series: pd.Series) -> pd.Series:
    observed = series.notna()
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        observed &= series.astype("string").str.strip().ne("").fillna(False)
    return observed


def target_eligibility_mask(target: object, df: pd.DataFrame) -> pd.Series | None:
    if pd.isna(target):
        return None
    text = str(target).lower()
    mask = pd.Series(True, index=df.index)

    if "females only" in text:
        if "RIAGENDR" not in df.columns:
            return None
        mask &= df["RIAGENDR"].eq(2)
    elif "males only" in text:
        if "RIAGENDR" not in df.columns:
            return None
        mask &= df["RIAGENDR"].eq(1)

    age_match = re.search(r"(\d+)\s+(years?|months?)\s*-\s*(\d+)\s+(years?|months?)", text)
    if age_match:
        age_months = age_in_months(df)
        if age_months is None:
            return None
        min_age_months = age_bound_to_months(age_match.group(1), age_match.group(2))
        max_age_months = age_bound_to_months(age_match.group(3), age_match.group(4))
        mask &= age_months.between(min_age_months, max_age_months, inclusive="both")

    return mask


def age_bound_to_months(value: str, unit: str) -> int:
    multiplier = 12 if unit.startswith("year") else 1
    return int(value) * multiplier


def age_in_months(df: pd.DataFrame) -> pd.Series | None:
    if "RIDAGEYR" not in df.columns:
        return None
    age_months = pd.to_numeric(df["RIDAGEYR"], errors="coerce") * 12
    if "RIDAGEMN" in df.columns:
        exact_months = pd.to_numeric(df["RIDAGEMN"], errors="coerce")
        age_months = exact_months.where(exact_months.notna(), age_months)
    return age_months


def metadata_for_cycle(metadata: pd.DataFrame | None, cycle: str) -> pd.DataFrame:
    if metadata is None or "cycle" not in metadata.columns:
        return pd.DataFrame()
    return metadata[metadata["cycle"].astype(str).eq(cycle)].copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in IDENTIFIER_COLUMNS and not col.endswith("__missing_reason")]


def add_missing_reason_columns(
    merged: pd.DataFrame,
    *,
    cycle_columns: dict[str, set[str]],
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    feature_cols = feature_columns(merged)
    reason_data = {
        f"{col}__missing_reason": np.where(
            observed_mask(merged[col]),
            MISSING_REASON_CODES["not_missing"],
            MISSING_REASON_CODES["unknown_missing"],
        ).astype(np.int8)
        for col in feature_cols
    }

    for cycle, row_idx in merged.groupby("cycle", sort=False).groups.items():
        cycle = str(cycle)
        idx = pd.Index(row_idx)
        cycle_df = merged.loc[idx]
        existing_columns = cycle_columns.get(cycle, set())
        cycle_metadata = metadata_for_cycle(metadata, cycle)
        metadata_by_variable = (
            cycle_metadata.drop_duplicates("variable_name").set_index("variable_name")
            if not cycle_metadata.empty and "variable_name" in cycle_metadata.columns
            else pd.DataFrame()
        )

        for col in feature_cols:
            reason_col = f"{col}__missing_reason"
            if (
                col in existing_columns
                and not metadata_by_variable.empty
                and col in metadata_by_variable.index
                and "value_labels_json" in metadata_by_variable.columns
            ):
                value_labels = parse_value_labels(metadata_by_variable.at[col, "value_labels_json"])
                refused_codes, dont_know_codes = nonresponse_code_sets(value_labels)
                refused = code_match_mask(cycle_df[col], refused_codes)
                dont_know = code_match_mask(cycle_df[col], dont_know_codes)
                reason_data[reason_col][idx[refused].to_numpy()] = MISSING_REASON_CODES["refused"]
                reason_data[reason_col][idx[dont_know].to_numpy()] = MISSING_REASON_CODES["dont_know"]

            missing = ~observed_mask(cycle_df[col])
            if not missing.any():
                continue

            if col not in existing_columns:
                reason_data[reason_col][idx[missing].to_numpy()] = MISSING_REASON_CODES["variable_not_in_cycle"]
                continue

            if not metadata_by_variable.empty and col in metadata_by_variable.index:
                target_mask = target_eligibility_mask(metadata_by_variable.at[col, "target"], cycle_df)
                if target_mask is not None:
                    not_eligible = missing & ~target_mask
                    reason_data[reason_col][idx[not_eligible].to_numpy()] = MISSING_REASON_CODES["not_eligible"]

            reachable_missing = missing & (reason_data[reason_col][idx.to_numpy()] == MISSING_REASON_CODES["unknown_missing"])
            if reachable_missing.any():
                reason_data[reason_col][idx[reachable_missing].to_numpy()] = MISSING_REASON_CODES["item_missing"]

        apply_skip_reasons(cycle_df, cycle_metadata, existing_columns, reason_data, idx)

    reason_df = pd.DataFrame(reason_data, index=merged.index)
    return pd.concat([merged, reason_df], axis=1)


def apply_skip_reasons(
    cycle_df: pd.DataFrame,
    cycle_metadata: pd.DataFrame,
    existing_columns: set[str],
    reason_data: dict[str, np.ndarray],
    idx: pd.Index,
) -> None:
    required = {"variable_name", "codebook_order", "skip_rules_json"}
    if cycle_metadata.empty or not required.issubset(cycle_metadata.columns):
        return

    ordered = cycle_metadata.dropna(subset=["codebook_order"]).copy()
    if ordered.empty:
        return
    ordered["codebook_order"] = pd.to_numeric(ordered["codebook_order"], errors="coerce")
    ordered = ordered.dropna(subset=["codebook_order"]).sort_values("codebook_order")
    order_by_variable = ordered.drop_duplicates("variable_name").set_index("variable_name")["codebook_order"].to_dict()

    for _, row in ordered.iterrows():
        source = str(row["variable_name"])
        if source not in existing_columns or source not in cycle_df.columns:
            continue
        source_order = order_by_variable.get(source)
        if source_order is None:
            continue
        normalized_source = cycle_df[source].map(normalize_code)
        for rule in parse_skip_rules(row.get("skip_rules_json")):
            skip_to = rule.get("skip_to")
            source_value = normalize_code(rule.get("source_value"))
            destination_order = order_by_variable.get(skip_to)
            if source_value is None or destination_order is None or destination_order <= source_order:
                continue
            skipped_variables = [
                variable
                for variable, order in order_by_variable.items()
                if source_order < order < destination_order and variable in existing_columns
            ]
            if not skipped_variables:
                continue
            triggered = normalized_source.eq(source_value)
            if not triggered.any():
                continue
            for variable in skipped_variables:
                reason_col = f"{variable}__missing_reason"
                if reason_col not in reason_data or variable not in cycle_df.columns:
                    continue
                current_reasons = reason_data[reason_col][idx.to_numpy()]
                reachable_reason = pd.Series(
                    np.isin(
                        current_reasons,
                        [
                            MISSING_REASON_CODES["item_missing"],
                            MISSING_REASON_CODES["unknown_missing"],
                        ],
                    ),
                    index=cycle_df.index,
                )
                skipped_missing = triggered & ~observed_mask(cycle_df[variable]) & reachable_reason
                reason_data[reason_col][idx[skipped_missing].to_numpy()] = MISSING_REASON_CODES["skipped"]


def write_missing_reason_code_map(out_file: Path) -> None:
    out_path = out_file.with_name(f"{out_file.stem}_missing_reason_codes.json")
    out_path.write_text(json.dumps(MISSING_REASON_CODES, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] Saved missing reason code map: {out_path}")


def apply_min_age_filter(df: pd.DataFrame, min_age: int) -> pd.DataFrame:
    if min_age <= 0:
        return df
    if "RIDAGEYR" not in df.columns:
        raise ValueError("RIDAGEYR not found but --min-age > 0; pass --min-age 0 to keep all ages.")
    n_before = len(df)
    filtered = df[pd.to_numeric(df["RIDAGEYR"], errors="coerce") >= min_age].copy().reset_index(drop=True)
    print(f"[age filter] RIDAGEYR >= {min_age}: {n_before} -> {len(filtered)} rows")
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stacked multi-cycle NHANES training table.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-file", type=Path, default=Path("data/processed/nhanes_2011_2023.parquet"))
    parser.add_argument(
        "--cycles",
        nargs="+",
        default=sorted(CYCLE_INFO.keys()),
        choices=sorted(CYCLE_INFO.keys()),
        help="NHANES cycles to stack together.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=DEFAULT_MODULES,
        help="Module prefixes, e.g. DEMO BMX BPX DIQ SMQ ALQ.",
    )
    parser.add_argument(
        "--codebook-metadata",
        type=Path,
        default=Path("data/external/codebooks/nhanes_metadata.parquet"),
        help="Enriched codebook metadata used to infer missingness reasons.",
    )
    parser.add_argument(
        "--no-missing-reason-columns",
        action="store_true",
        help="Do not append per-feature missing reason columns.",
    )
    parser.add_argument(
        "--min-age",
        type=int,
        default=20,
        help="Keep only rows with RIDAGEYR >= this value (default 20). Pass 0 to keep all ages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cycle_tables = []
    cycle_columns: dict[str, set[str]] = {}
    for cycle in args.cycles:
        try:
            cycle_df = load_cycle_merged(args.raw_root, cycle, args.modules)
        except RuntimeError as e:
            print(f"[WARN] Skipping cycle {cycle}: {e}")
            continue
        cycle_columns[cycle] = set(cycle_df.columns)
        cycle_tables.append(cycle_df)
        print(f"[OK] Built cycle table {cycle}: {cycle_df.shape}")

    if not cycle_tables:
        raise RuntimeError("No cycle tables built. Check downloaded raw files.")

    merged = pd.concat(cycle_tables, axis=0, ignore_index=True, sort=False)
    merged = apply_min_age_filter(merged, args.min_age)

    unique_ids = merged["SEQN_cycle"].dropna().astype(str).unique().to_numpy(copy=True)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(unique_ids)
    n_total = len(unique_ids)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])

    def assign_split(seqn_cycle: str) -> str:
        if seqn_cycle in train_ids:
            return "train"
        if seqn_cycle in val_ids:
            return "val"
        return "test"

    merged["split"] = merged["SEQN_cycle"].map(assign_split)

    if not args.no_missing_reason_columns:
        metadata = read_metadata(args.codebook_metadata)
        merged = add_missing_reason_columns(
            merged,
            cycle_columns=cycle_columns,
            metadata=metadata,
        )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out_file, index=False)
    if not args.no_missing_reason_columns:
        write_missing_reason_code_map(args.out_file)
    n_samples, n_total_features = merged.shape
    print(f"[OK] Saved stacked dataset: {args.out_file} shape={merged.shape}")
    print(f"[INFO] n_samples={n_samples}, n_total_features={n_total_features}")


if __name__ == "__main__":
    main()
