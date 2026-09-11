from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INVALID_CODES = {7, 9, 77, 99, 777, 999}
MEDICAL_QUESTIONNAIRE_PREFIXES = (
    "BPQ",
    "BPD",
    "DIQ",
    "DID",
    "MCQ",
    "MCD",
    "KIQ",
    "KID",
)
ALWAYS_DROP_FEATURE_COLUMNS = {
    "BMDSADCM",
    "BMDSTATS",
    "BPAARM",
    "BPAOARM",
    "AIALANGA",
    "FIAINTRP",
    "FIALANG",
    "FIAPROXY",
    "MIAINTRP",
    "MIALANG",
    "MIAPROXY",
    "RIDEXMON",
    "RIDSTATR",
    "SDDSRVYR",
    "SDMVPSU",
    "SDMVSTRA",
    "SIAINTRP",
    "SIALANG",
    "SIAPROXY",
    "WTINT2YR",
    "WTINTPRP",
    "WTMEC2YR",
    "WTMECPRP",
    "WTPH2YR",
    "WTSAF2YR",
    "SMAQUEX2",
    "SMD100BR",
    "SMDUPCA",
    "TRIGLY_WTSAFPRP",
    "TRIGLY_WTSAF2YR",
}


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_binary(series: pd.Series, yes_code: int = 1, no_code: int = 2) -> pd.Series:
    x = to_numeric(series)
    x = x.where(~x.isin(INVALID_CODES))
    out = pd.Series(pd.NA, index=x.index, dtype="Int64")
    out.loc[x == yes_code] = 1
    out.loc[x == no_code] = 0
    return out


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def missing_binary(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="Int64")


def clean_binary_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return missing_binary(df.index)
    return clean_binary(df[column])


def any_yes_all_no(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = pd.concat([clean_binary_column(df, column) for column in columns], axis=1)
    out = missing_binary(df.index)
    out.loc[(values == 1).any(axis=1)] = 1
    out.loc[(values == 0).all(axis=1)] = 0
    return out


def cycle_mask(df: pd.DataFrame, cycles: set[str]) -> pd.Series:
    if "cycle" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["cycle"].astype(str).isin(cycles)


def is_medical_questionnaire_column(column: str) -> bool:
    base_column = column.removesuffix("__missing_reason")
    return base_column.startswith(MEDICAL_QUESTIONNAIRE_PREFIXES)


def drop_medical_questionnaire_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [column for column in df.columns if is_medical_questionnaire_column(column)]
    if not drop_columns:
        return df
    return df.drop(columns=drop_columns)


def drop_always_excluded_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [
        column
        for column in df.columns
        if column.removesuffix("__missing_reason") in ALWAYS_DROP_FEATURE_COLUMNS
    ]
    if not drop_columns:
        return df
    return df.drop(columns=drop_columns)


def harmonized_bp_mean(df: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "systolic":
        auscultatory_prefix = "BPXSY"
        oscillometric_prefix = "BPXOSY"
        oscillometric_offset = 1.5
    elif kind == "diastolic":
        auscultatory_prefix = "BPXDI"
        oscillometric_prefix = "BPXODI"
        oscillometric_offset = -1.3
    else:
        raise ValueError(f"Unsupported blood pressure kind: {kind}")

    readings = []
    for reading_number in range(1, 5):
        auscultatory_col = f"{auscultatory_prefix}{reading_number}"
        oscillometric_col = f"{oscillometric_prefix}{reading_number}"
        auscultatory = (
            to_numeric(df[auscultatory_col])
            if auscultatory_col in df.columns
            else pd.Series(np.nan, index=df.index, dtype="Float64")
        )
        oscillometric = (
            to_numeric(df[oscillometric_col]) + oscillometric_offset
            if oscillometric_col in df.columns
            else pd.Series(np.nan, index=df.index, dtype="Float64")
        )
        reading = auscultatory.where(auscultatory.notna(), oscillometric)
        if reading.notna().any():
            readings.append(reading)

    if not readings:
        return pd.Series(np.nan, index=df.index, dtype="Float64")
    return pd.concat(readings, axis=1).mean(axis=1, skipna=True).astype("Float64")


def calculate_ckm_stage(df: pd.DataFrame) -> pd.Series:
    """
    Calculates CKM Stage (0, 1, or 2).
    Excludes Stage 3 & 4 (returns pd.NA for those rows).
    """
    ckm_stage = pd.Series(pd.NA, index=df.index, dtype="Int64")

    def check_yes(col: str) -> pd.Series:
        return df.get(col, pd.Series(0, index=df.index)) == 1

    def numeric_or_zero(col: str) -> pd.Series:
        return df.get(col, pd.Series(0, index=df.index))

    # Stage 4 exclusion: clinical CVD
    has_cvd = check_yes("MCQ160B") | check_yes("MCQ160C") | check_yes("MCQ160E") | check_yes("MCQ160F")

    # Major abnormalities (Stage 2)
    has_diabetes = (numeric_or_zero("LBXGLU") >= 126) | check_yes("DIQ010")
    systolic_bp = harmonized_bp_mean(df, "systolic")
    diastolic_bp = harmonized_bp_mean(df, "diastolic")
    high_sbp = (systolic_bp >= 130).fillna(False)
    high_dbp = (diastolic_bp >= 80).fillna(False)
    mild_sbp = (systolic_bp >= 120).fillna(False)
    has_hypertension = (
        high_sbp
        | high_dbp
        | check_yes("BPQ020")
    )
    has_high_trig = numeric_or_zero("LBXTR") >= 150
    has_ckd = check_yes("KIQ022")

    major_abnormality_count = (
        has_diabetes.astype(int)
        + has_hypertension.astype(int)
        + has_high_trig.astype(int)
        + has_ckd.astype(int)
    )

    # Mild abnormalities (Stage 1)
    has_excess_weight = numeric_or_zero("BMXBMI") >= 25
    has_mild_bp = mild_sbp & ~has_hypertension
    has_mild_metabolic = (
        (numeric_or_zero("LBXTR") >= 100) | (numeric_or_zero("LBXGLU") >= 100)
    ) & ~has_high_trig & ~has_diabetes

    has_mild_issues = has_excess_weight | has_mild_bp | has_mild_metabolic | (major_abnormality_count == 1)

    # Assignment logic (top-down priority)
    mask_stage_4 = has_cvd
    mask_stage_2 = (major_abnormality_count >= 2) & ~mask_stage_4
    ckm_stage.loc[mask_stage_2] = 2

    mask_stage_1 = has_mild_issues & ~mask_stage_2 & ~mask_stage_4
    ckm_stage.loc[mask_stage_1] = 1

    mask_stage_0 = ~mask_stage_4 & ~mask_stage_2 & ~mask_stage_1
    ckm_stage.loc[mask_stage_0] = 0

    ckm_stage.loc[mask_stage_4] = pd.NA
    return ckm_stage


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    key_cols = [c for c in ["SEQN", "SEQN_cycle", "cycle", "RIDAGEYR", "RIAGENDR", "RIDRETH3"] if c in df.columns]
    for col in key_cols:
        out[col] = df[col]

    # Diabetes: doctor-told diabetes (DIQ010; 1 yes, 2 no)
    if "DIQ010" in df.columns:
        out["label_diabetes_self_report"] = clean_binary(df["DIQ010"])
    else:
        out["label_diabetes_self_report"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # Hypertension: use BPQ020 if present, fallback to BPQ150A
    bp_diag_col = first_existing_column(df, ["BPQ020", "BPQ150A"])
    if bp_diag_col is not None:
        out["label_hypertension_self_report"] = clean_binary(df[bp_diag_col])
    else:
        out["label_hypertension_self_report"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    # TyG-WHtR index: ln((Triglycerides * Fasting Glucose) / 2) * (Waist / Height)
    required = ["LBXTR", "LBXGLU", "BMXWAIST", "BMXHT"]
    if all(c in df.columns for c in required):
        trig = to_numeric(df["LBXTR"]).where(lambda s: ~s.isin(INVALID_CODES))
        glu = to_numeric(df["LBXGLU"]).where(lambda s: ~s.isin(INVALID_CODES))
        waist = to_numeric(df["BMXWAIST"]).where(lambda s: ~s.isin(INVALID_CODES))
        height = to_numeric(df["BMXHT"]).where(lambda s: ~s.isin(INVALID_CODES))

        whtr = waist / height
        whtr[height <= 0] = pd.NA

        tyg = np.log((trig * glu) / 2)
        out["label_tyg_whtr"] = (tyg * whtr).astype("Float64")
    else:
        out["label_tyg_whtr"] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    out["label_ckm_stage"] = calculate_ckm_stage(df)

    # Coronary heart disease, self-report (MCQ160C: "ever told you had coronary heart disease").
    chd_col = first_existing_column(df, ["MCQ160C"])
    if chd_col is not None:
        out["label_chd"] = clean_binary(df[chd_col])
    else:
        out["label_chd"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    out["label_arthritis"] = clean_binary_column(df, "MCQ160A")
    out["label_asthma"] = clean_binary_column(df, "MCQ010")
    out["label_thyroid_condition"] = clean_binary_column(df, "MCQ160M")
    out["label_any_cancer"] = clean_binary_column(df, "MCQ220")
    out["label_gout"] = clean_binary_column(df, "MCQ160N")
    out["label_liver_condition"] = clean_binary_column(df, "MCQ160L")
    out["label_stroke"] = clean_binary_column(df, "MCQ160F")
    out["label_myocardial_infarction"] = clean_binary_column(df, "MCQ160E")

    combined_lung_disease = any_yes_all_no(df, ["MCQ160G", "MCQ160K", "MCQ160O"])

    out["label_copd"] = missing_binary(df.index)
    mask = cycle_mask(df, {"2013-2014", "2015-2016"})
    out.loc[mask, "label_copd"] = clean_binary_column(df, "MCQ160O").loc[mask]
    mask = cycle_mask(df, {"2017-2018"})
    out.loc[mask, "label_copd"] = combined_lung_disease.loc[mask]
    mask = cycle_mask(df, {"2019-2020", "2021-2023"})
    out.loc[mask, "label_copd"] = clean_binary_column(df, "MCQ160P").loc[mask]

    out["label_emphysema"] = missing_binary(df.index)
    mask = cycle_mask(df, {"2011-2012", "2013-2014", "2015-2016"})
    out.loc[mask, "label_emphysema"] = clean_binary_column(df, "MCQ160G").loc[mask]
    mask = cycle_mask(df, {"2017-2018"})
    out.loc[mask, "label_emphysema"] = combined_lung_disease.loc[mask]
    mask = cycle_mask(df, {"2019-2020", "2021-2023"})
    out.loc[mask, "label_emphysema"] = clean_binary_column(df, "MCQ160P").loc[mask]

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean downstream task labels from merged NHANES table.")
    parser.add_argument(
        "--in-file",
        type=Path,
        default=Path("data/processed/nhanes_2011_2023.parquet"),
        help="Input NHANES parquet from prepare_features.py.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/processed/nhanes_2011_2023.parquet"),
        help="Output canonical modeling parquet with downstream labels appended.",
    )
    parser.add_argument(
        "--task-registry",
        type=Path,
        default=Path("src/downstream_tasks/task_registry.json"),
        help="Task registry whose target_column entries are appended to the canonical modeling parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.in_file)
    labels = build_labels(df)

    target_columns = load_registry_target_columns(args.task_registry)
    missing_targets = [column for column in target_columns if column not in labels.columns]
    if missing_targets:
        raise ValueError(f"Registry target columns are not produced by build_labels(): {missing_targets}")
    labels_only = labels[target_columns]
    df = df.drop(columns=[column for column in target_columns if column in df.columns])
    n_before_drop = len(df.columns)
    df = drop_medical_questionnaire_features(df)
    df = drop_always_excluded_features(df)
    n_dropped = n_before_drop - len(df.columns)
    merged = pd.concat([df, labels_only], axis=1)

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out_file, index=False)

    print(f"[OK] Saved merged file: {args.out_file} shape={merged.shape}")
    if n_dropped:
        print(f"[INFO] dropped {n_dropped} medical questionnaire feature columns before writing output")

    for col in target_columns:
        if col in labels.columns:
            vc = labels[col].value_counts(dropna=False).to_dict()
            print(f"[INFO] {col} distribution: {vc}")


def load_registry_target_columns(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task registry must contain a JSON object: {path}")
    target_columns = []
    seen = set()
    for task_name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Task {task_name!r} must contain a JSON object in {path}")
        target_column = entry.get("target_column")
        if not target_column:
            raise ValueError(f"Task {task_name!r} is missing target_column in {path}")
        target_column = str(target_column)
        if target_column not in seen:
            target_columns.append(target_column)
            seen.add(target_column)
    return target_columns


if __name__ == "__main__":
    main()
