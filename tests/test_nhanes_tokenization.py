from pathlib import Path

import pandas as pd
import pytest
import torch
from bs4 import BeautifulSoup

from src.tokenizer.metadata import apply_value_labels, build_feature_description, load_feature_metadata
from src.tokenizer.dataset import TokenizedTabularDataset, batch_from_dataset_tensors
from src.tokenizer.embed_feature_names import read_feature_texts
from src.nhanes.build_downstream_labels import (
    build_labels,
    calculate_ckm_stage,
    clean_binary,
    drop_always_excluded_features,
    drop_medical_questionnaire_features,
    harmonized_bp_mean,
)
from src.nhanes.download_codebooks import parse_codebook_page
from src.nhanes.download_nhanes import iter_targets
from src.nhanes.prepare_features import apply_min_age_filter, module_file_stem
from src.tokenizer.columns import (
    infer_categorical_and_binary_columns,
    infer_columns_from_value_labels,
    load_registry_drop_columns,
    load_registry_target_columns,
    load_value_label_specs,
    model_feature_columns,
    sanitize_range_feature_values,
)
from src.tokenizer.preprocessing import NameValuePreprocessor, transformed_batch_to_dict
from src.tokenizer.schema import FeatureType
from src.tokenizer.tokenization_artifacts import build_tokenization_check, build_tokenization_plan


def test_kiq_aliases_resolve_to_urology_file_prefix() -> None:
    assert module_file_stem("KIQ", "2011-2012") == "KIQ_U_G"
    assert module_file_stem("KIQ_U", "2011-2012") == "KIQ_U_G"

    targets = list(iter_targets(["2011-2012"], ["KIQ", "KIQ_U"]))

    assert [target[1] for target in targets] == ["KIQ_U", "KIQ_U"]
    assert targets[0][2].endswith("/KIQ_U_G.xpt")
    assert targets[1][2].endswith("/KIQ_U_G.xpt")


def test_min_age_filter_keeps_adults_by_default() -> None:
    frame = pd.DataFrame({"RIDAGEYR": [19, 20, 44, None], "SEQN": [1, 2, 3, 4]})

    filtered = apply_min_age_filter(frame, 20)

    assert filtered["SEQN"].tolist() == [2, 3]


def test_min_age_filter_can_keep_all_ages() -> None:
    frame = pd.DataFrame({"RIDAGEYR": [12, 20], "SEQN": [1, 2]})

    filtered = apply_min_age_filter(frame, 0)

    assert filtered is frame
    assert filtered["SEQN"].tolist() == [1, 2]


def test_min_age_filter_requires_age_column_when_enabled() -> None:
    frame = pd.DataFrame({"SEQN": [1, 2]})

    with pytest.raises(ValueError, match="RIDAGEYR not found"):
        apply_min_age_filter(frame, 20)


def test_harmonized_bp_mean_uses_all_available_auscultatory_readings() -> None:
    frame = pd.DataFrame(
        {
            "BPXSY1": [120.0, None],
            "BPXSY2": [122.0, None],
            "BPXSY3": [124.0, 140.0],
            "BPXSY4": [None, 144.0],
            "BPXDI1": [70.0, None],
            "BPXDI2": [72.0, None],
            "BPXDI3": [74.0, 90.0],
            "BPXDI4": [None, 92.0],
        }
    )

    assert harmonized_bp_mean(frame, "systolic").tolist() == [122.0, 142.0]
    assert harmonized_bp_mean(frame, "diastolic").tolist() == [72.0, 91.0]


def test_harmonized_bp_mean_calibrates_oscillometric_readings() -> None:
    frame = pd.DataFrame(
        {
            "BPXOSY1": [120.0],
            "BPXOSY2": [122.0],
            "BPXOSY3": [124.0],
            "BPXODI1": [80.0],
            "BPXODI2": [82.0],
            "BPXODI3": [84.0],
        }
    )

    assert harmonized_bp_mean(frame, "systolic").tolist() == pytest.approx([123.5])
    assert harmonized_bp_mean(frame, "diastolic").tolist() == pytest.approx([80.7])


def test_harmonized_bp_mean_returns_missing_when_no_readings_exist() -> None:
    frame = pd.DataFrame({"SEQN": [1, 2]})

    assert harmonized_bp_mean(frame, "systolic").isna().tolist() == [True, True]
    assert harmonized_bp_mean(frame, "diastolic").isna().tolist() == [True, True]


def test_medical_questionnaire_features_are_dropped_after_label_generation() -> None:
    frame = pd.DataFrame(
        {
            "SEQN": ["1", "2"],
            "SEQN_cycle": ["2011-2012_1", "2011-2012_2"],
            "cycle": ["2011-2012", "2011-2012"],
            "split": ["train", "test"],
            "RIDAGEYR": [45, 62],
            "RIAGENDR": [1, 2],
            "BMXBMI": [24.0, 31.0],
            "BPXSY1": [118.0, 135.0],
            "BPXSY2": [120.0, 136.0],
            "BPXDI1": [76.0, 82.0],
            "BPXDI2": [78.0, 84.0],
            "LBXGLU": [95.0, 140.0],
            "LBXTR": [90.0, 170.0],
            "BMXWAIST": [80.0, 110.0],
            "BMXHT": [170.0, 165.0],
            "DIQ010": [2, 1],
            "DIQ010__missing_reason": [0, 0],
            "DID040": [pd.NA, 50],
            "BPQ020": [2, 1],
            "BPD035": [pd.NA, 45],
            "MCQ010": [2, 1],
            "MCQ160A": [2, 1],
            "MCQ160C": [2, 1],
            "MCQ160E": [2, 1],
            "MCQ160F": [2, 1],
            "MCQ160G": [2, 1],
            "MCQ160L": [2, 1],
            "MCQ160M": [2, 1],
            "MCQ160N": [2, 1],
            "MCQ220": [2, 1],
            "MCD180A": [pd.NA, 55],
            "KIQ022": [2, 1],
            "KID022": [2, 1],
            "SMQ020": [1, 2],
            "SMQ020__missing_reason": [0, 0],
            "ALQ101": [1, 2],
            "SDMVPSU": [1, 2],
            "SDMVPSU__missing_reason": [0, 0],
            "WTMEC2YR": [100.0, 200.0],
        }
    )

    labels = build_labels(frame)
    output_features = drop_always_excluded_features(drop_medical_questionnaire_features(frame))
    output = pd.concat(
        [output_features, labels[["label_diabetes_self_report", "label_hypertension_self_report", "label_chd"]]],
        axis=1,
    )

    assert labels["label_diabetes_self_report"].tolist() == [0, 1]
    assert labels["label_hypertension_self_report"].tolist() == [0, 1]
    assert labels["label_chd"].tolist() == [0, 1]
    assert labels["label_arthritis"].tolist() == [0, 1]
    assert labels["label_asthma"].tolist() == [0, 1]
    assert labels["label_thyroid_condition"].tolist() == [0, 1]
    assert labels["label_any_cancer"].tolist() == [0, 1]
    assert labels["label_gout"].tolist() == [0, 1]
    assert labels["label_liver_condition"].tolist() == [0, 1]
    assert labels["label_stroke"].tolist() == [0, 1]
    assert labels["label_myocardial_infarction"].tolist() == [0, 1]
    assert labels["label_emphysema"].tolist() == [0, 1]
    assert "DIQ010" not in output.columns
    assert "DIQ010__missing_reason" not in output.columns
    assert "DID040" not in output.columns
    assert "BPQ020" not in output.columns
    assert "BPD035" not in output.columns
    assert "MCQ160C" not in output.columns
    assert "MCD180A" not in output.columns
    assert "KIQ022" not in output.columns
    assert "KID022" not in output.columns
    assert "SMQ020" in output.columns
    assert "SMQ020__missing_reason" in output.columns
    assert "ALQ101" in output.columns
    assert "SDMVPSU" not in output.columns
    assert "SDMVPSU__missing_reason" not in output.columns
    assert "WTMEC2YR" not in output.columns
    assert "BMXBMI" in output.columns
    assert "BPXSY1" in output.columns
    assert "LBXGLU" in output.columns
    assert "label_diabetes_self_report" in output.columns


def test_clean_binary_maps_nonresponse_to_missing() -> None:
    result = clean_binary(pd.Series([1, 2, 7, 9, 77, 99, 777, 999, None]))

    assert result.tolist()[:2] == [1, 0]
    assert result.tolist()[2:] == [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]


def test_lung_labels_use_cycle_specific_harmonization() -> None:
    frame = pd.DataFrame(
        {
            "cycle": [
                "2011-2012",
                "2013-2014",
                "2015-2016",
                "2017-2018",
                "2017-2018",
                "2019-2020",
                "2021-2023",
            ],
            "MCQ160G": [1, 2, 2, 2, 1, pd.NA, pd.NA],
            "MCQ160K": [2, 2, 2, 1, 2, pd.NA, pd.NA],
            "MCQ160O": [pd.NA, 1, 2, 2, 2, pd.NA, pd.NA],
            "MCQ160P": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 1, 2],
        }
    )

    labels = build_labels(frame)

    assert labels["label_copd"].tolist() == [pd.NA, 1, 0, 1, 1, 1, 0]
    assert labels["label_emphysema"].tolist() == [1, 0, 0, 1, 1, 1, 0]


def test_gout_stays_missing_when_source_variable_is_absent() -> None:
    labels = build_labels(pd.DataFrame({"cycle": ["2019-2020", "2021-2023"]}))

    assert labels["label_gout"].isna().tolist() == [True, True]


def test_ckm_stage_uses_harmonized_bp_mean_without_persisted_columns() -> None:
    frame = pd.DataFrame(
        {
            "BPXSY1": [None, 118.0],
            "BPXSY2": [132.0, 121.0],
            "BPXSY3": [134.0, 124.0],
            "BPXDI1": [None, 70.0],
            "BPXDI2": [82.0, 71.0],
            "BPXDI3": [84.0, 72.0],
            "BMXBMI": [22.0, 22.0],
            "LBXTR": [80.0, 80.0],
            "LBXGLU": [80.0, 80.0],
            "DIQ010": [2.0, 2.0],
            "BPQ020": [2.0, 2.0],
            "KIQ022": [2.0, 2.0],
            "MCQ160B": [2.0, 2.0],
            "MCQ160C": [2.0, 2.0],
            "MCQ160E": [2.0, 2.0],
            "MCQ160F": [2.0, 2.0],
        }
    )

    assert calculate_ckm_stage(frame).tolist() == [1, 1]


def test_feature_description_excludes_redundant_code_fields() -> None:
    group = pd.DataFrame(
        [
            {
                "variable_name": "RIDAGEYR",
                "sas_label": "Age in years at screening",
                "english_text": "Age in years of the participant at the time of screening.",
                "variable_description": "Age in years of the participant at the time of screening.",
                "target": "Both males and females 0 YEARS - 150 YEARS",
                "unit": "",
                "hard_edits": "",
            }
        ]
    )

    text = build_feature_description(group)

    assert "RIDAGEYR" not in text
    assert "SAS label" not in text
    assert text.count("Age in years of the participant") == 1
    assert "Target population" in text


def test_feature_description_uses_sas_label_and_english_text_before_variable_description() -> None:
    group = pd.DataFrame(
        [
            {
                "variable_name": "TESTVAR",
                "sas_label": "Short SAS label",
                "english_text": "Detailed codebook question text. English Instructions: ENTER VALUE",
                "variable_description": "Variable-list fallback description.",
                "target": "",
                "unit": "",
                "hard_edits": "",
            }
        ]
    )

    text = build_feature_description(group)

    assert text.startswith("Description: Short SAS label. Detailed codebook question text.")
    assert "English Instructions" not in text
    assert "Variable-list fallback description" not in text


def test_feature_description_uses_variable_description_as_fallback() -> None:
    group = pd.DataFrame(
        [
            {
                "variable_name": "TESTVAR",
                "sas_label": "",
                "english_text": "",
                "variable_description": "Variable-list fallback description.",
                "target": "",
                "unit": "",
                "hard_edits": "",
            }
        ]
    )

    text = build_feature_description(group)

    assert text == "Description: Variable-list fallback description."


def test_feature_text_json_reader_aligns_by_feature_name(tmp_path) -> None:
    feature_texts = tmp_path / "feature_semantics.json"
    feature_texts.write_text(
        """
{
  "B": {"feature_semantic": "Type: categorical.\\nDescription: Second."},
  "A": {"feature_semantic": "Type: numerical.\\nDescription: First."}
}
""".strip(),
        encoding="utf-8",
    )

    assert read_feature_texts(feature_texts, feature_names=["A", "B"]) == [
        "Type: numerical.\nDescription: First.",
        "Type: categorical.\nDescription: Second.",
    ]


def test_cycle_aware_category_labels_are_applied() -> None:
    frame = pd.DataFrame(
        {
            "cycle": ["2011-2012", "2013-2014", "2015-2016"],
            "RIAGENDR": [1.0, 2.0, 3.0],
        }
    )
    value_labels = {"RIAGENDR": {"1": "Male", "2": "Female"}}
    cycle_value_labels = {
        "RIAGENDR": {
            "2011-2012": {"1": "Male 2011"},
            "2013-2014": {"2": "Female 2013"},
        }
    }

    transformed = apply_value_labels([frame], value_labels, cycle_value_labels)[0]

    assert transformed["RIAGENDR"].tolist() == ["Male 2011", "Female 2013", 3.0]


def test_missing_reason_masks_nonresponse_and_quantile_bins() -> None:
    train = pd.DataFrame(
        {
            "BMXBMI": [20.0, 25.0, 30.0, 777.0],
            "BMXBMI__missing_reason": [0, 0, 0, 7],
            "ALQ101": ["Yes", "No", "Yes", "Refused"],
            "ALQ101__missing_reason": [0, 0, 0, 7],
        }
    )
    preprocessor = NameValuePreprocessor.infer_from_dataframe(
        train[["BMXBMI", "ALQ101"]],
        numerical_columns=["BMXBMI"],
        categorical_columns=["ALQ101"],
        continuous_quantile_bins=2,
    )

    batch = preprocessor.fit_transform(train)

    assert batch.missing_mask[:, 0].tolist() == [False, False, False, True]
    assert batch.missing_reason_codes[:, 0].tolist() == [0, 0, 0, 7]
    assert batch.numeric_values[3, 0].item() == 0.0
    assert batch.continuous_bin_codes[:, 0].tolist() == [1, 2, 2, 0]
    assert batch.missing_mask[:, 1].tolist() == [False, False, False, True]


def test_tokenized_tabular_dataset_loads_named_rows_and_batches(tmp_path) -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    preprocessor = NameValuePreprocessor.infer_from_dataframe(
        train,
        numerical_columns=["x"],
        categorical_columns=["cat"],
    )
    batch = preprocessor.fit_transform(train)
    metadata = {
        "feature_names": preprocessor.feature_names,
        "feature_type_ids": preprocessor.feature_type_ids(),
        "categorical_cardinalities": preprocessor.categorical_cardinalities(),
        "continuous_bin_cardinality": preprocessor.continuous_bin_cardinality(),
        "missing_reason_cardinality": preprocessor.missing_reason_cardinality(),
    }
    torch.save(transformed_batch_to_dict(batch), tmp_path / "train_tokens.pt")
    torch.save(metadata, tmp_path / "tokenizer_metadata.pt")

    dataset = TokenizedTabularDataset.from_dir(tmp_path, "train")
    row = dataset[1]

    assert len(dataset) == 3
    assert dataset.feature_names == ["x", "cat"]
    assert dataset.n_features == 2
    assert set(row) == {
        "numeric_values",
        "continuous_bin_codes",
        "categorical_codes",
        "feature_context_codes",
        "missing_reason_codes",
        "missing_mask",
        "observed_mask",
        "row_idx",
    }
    assert row["numeric_values"].shape == (2,)
    assert row["row_idx"].item() == 1

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    payload = next(iter(loader))
    loader_batch = batch_from_dataset_tensors(payload, torch.device("cpu"))

    assert payload["row_idx"].tolist() == [0, 1]
    assert loader_batch.numeric_values.shape == (2, 2)
    assert loader_batch.categorical_codes.tolist() == batch.categorical_codes[:2].tolist()
    assert loader_batch.feature_context_codes.tolist() == batch.feature_context_codes[:2].tolist()


def test_category_ids_are_distinct_for_different_feature_contexts() -> None:
    train = pd.DataFrame({"question": ["Yes", "Yes", "No", "No"]})
    context_codes = pd.DataFrame({"question": [0, 1, 0, 1]})
    preprocessor = NameValuePreprocessor.infer_from_dataframe(
        train,
        categorical_columns=["question"],
        feature_context_texts_by_feature={
            "question": [
                "Type: categorical.\nDescription: First cycle question.",
                "Type: categorical.\nDescription: Later cycle question.",
            ]
        },
    )

    batch = preprocessor.fit_transform(train, context_codes)
    category_texts = preprocessor.categorical_value_texts()

    assert batch.feature_context_codes[:, 0].tolist() == [0, 1, 0, 1]
    assert batch.categorical_codes[0, 0].item() != batch.categorical_codes[1, 0].item()
    assert batch.categorical_codes[2, 0].item() != batch.categorical_codes[3, 0].item()
    assert "First cycle question" in category_texts["question"]["0"][2]
    assert "Later cycle question" in category_texts["question"]["1"][2]


def test_all_observed_categorical_labels_are_kept() -> None:
    train = pd.DataFrame({"cat": ["Common", "Common", "Single observed label"]})
    preprocessor = NameValuePreprocessor.infer_from_dataframe(
        train,
        categorical_columns=["cat"],
    )

    batch = preprocessor.fit_transform(train)
    labels = preprocessor.categorical_labels("cat")
    single_observed_code = labels.index((0, "Single observed label"))
    other_code = labels.index((0, "OTHER"))

    assert single_observed_code != other_code
    assert batch.categorical_codes[:, 0].tolist() == [
        labels.index((0, "Common")),
        labels.index((0, "Common")),
        single_observed_code,
    ]


def test_tokenization_plan_contains_context_rows() -> None:
    train = pd.DataFrame({"cycle": ["2011-2012", "2013-2014"], "question": ["Yes", "No"]})
    context_codes = pd.DataFrame({"question": [0, 1]})
    preprocessor = NameValuePreprocessor.infer_from_dataframe(
        train[["question"]],
        categorical_columns=["question"],
        feature_context_texts_by_feature={
            "question": [
                "Type: categorical.\nDescription: First cycle question.",
                "Type: categorical.\nDescription: Later cycle question.",
            ]
        },
    )
    preprocessor.fit_transform(train[["question"]], context_codes)
    metadata = type(
        "Metadata",
        (),
        {"feature_context_cycle_maps": {"question": {"2011-2012": 0, "2013-2014": 1}}},
    )()

    plan = build_tokenization_plan(
        preprocessor=preprocessor,
        metadata=metadata,
        model_columns=["question"],
        numerical_columns=[],
        categorical_columns=["question"],
        classification_report={"label_map_categorical_columns": ["question"], "label_map_numerical_columns": []},
        target_columns=[],
        split_frames={"train": train, "val": train.head(0), "test": train.head(0)},
    )
    assert plan["feature_name"].tolist() == ["question", "question"]
    assert plan["context_code"].tolist() == [0, 1]
    assert plan["n_context_category_texts"].tolist() == [3, 3]
    assert "First cycle question" in plan.loc[0, "feature_context_text"]
    assert "Later cycle question" in plan.loc[1, "feature_context_text"]


def test_tokenization_check_contains_row_feature_outputs() -> None:
    train = pd.DataFrame(
        {
            "SEQN": [1, 2],
            "cycle": ["2011-2012", "2013-2014"],
            "question": ["Yes", "No"],
            "numeric": [10.0, 20.0],
        }
    )
    context_codes = pd.DataFrame({"question": [0, 1], "numeric": [0, 0]})
    preprocessor = NameValuePreprocessor.infer_from_dataframe(
        train[["question", "numeric"]],
        categorical_columns=["question"],
        numerical_columns=["numeric"],
        continuous_quantile_bins=2,
        feature_context_texts_by_feature={
            "question": [
                "Type: categorical.\nDescription: First cycle question.",
                "Type: categorical.\nDescription: Later cycle question.",
            ],
            "numeric": ["Type: numerical.\nDescription: Numeric feature."],
        },
    )
    batch = preprocessor.fit_transform(train[["question", "numeric"]], context_codes)

    check = build_tokenization_check(
        preprocessor=preprocessor,
        batches={"train": batch},
        split_frames={"train": train},
        model_columns=["question", "numeric"],
        numerical_columns=["numeric"],
    )

    assert {
        "split",
        "SEQN",
        "feature_name",
        "numeric_mean",
        "numeric_std",
        "quantile_edges",
        "continuous_bin_code",
        "category_text",
        "missing_mask",
    } <= set(check)
    question_rows = check[(check["split"] == "train") & (check["feature_name"] == "question")]
    numeric_rows = check[(check["split"] == "train") & (check["feature_name"] == "numeric")]
    assert question_rows["feature_context_code"].tolist() == [0, 1]
    assert question_rows["categorical_code"].nunique() == 2
    assert question_rows["category_text"].str.contains("Answer:").all()
    assert numeric_rows["continuous_bin_code"].tolist() == [1, 2]
    assert numeric_rows["quantile_edges"].nunique() == 1


def test_registry_target_columns_are_not_feature_tokens() -> None:
    frame = pd.DataFrame(
        {
            "SEQN": [1, 2, 3],
            "split": ["train", "train", "train"],
            "x": [1.0, 2.0, 3.0],
            "label_feature": [4.0, 5.0, 6.0],
            "label_ckm_stage": [0, None, 2],
            "x__missing_reason": [0, 0, 0],
        }
    )
    assert model_feature_columns(frame, target_columns=["label_ckm_stage"]) == ["x", "label_feature"]

def test_registry_drop_columns_are_not_feature_tokens(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "SEQN": [1, 2],
            "split": ["train", "train"],
            "x": [1.0, 2.0],
            "leak": [3.0, 4.0],
            "label_task": [0, 1],
        }
    )
    registry = tmp_path / "task_registry.json"
    registry.write_text(
        """
{
  "task": {
    "target_column": "label_task",
    "cols_to_drop": ["leak", "missing_from_tokens"]
  }
}
""",
        encoding="utf-8",
    )

    assert load_registry_drop_columns(registry) == ["leak", "missing_from_tokens"]
    assert model_feature_columns(
        frame,
        target_columns=load_registry_target_columns(registry),
        drop_columns=load_registry_drop_columns(registry),
    ) == ["x"]


def test_registry_target_columns_load_uniquely_in_task_order(tmp_path) -> None:
    registry = tmp_path / "task_registry.json"
    registry.write_text(
        """
{
  "A": {"target_column": "label_a"},
  "B": {"target_column": "label_b"},
  "C": {"target_column": "label_a"}
}
""",
        encoding="utf-8",
    )

    assert load_registry_target_columns(registry) == ["label_a", "label_b"]


def test_binary_valued_columns_are_treated_as_categorical() -> None:
    frame = pd.DataFrame(
        {
            "bool_feature": [True, False, True],
            "zero_one_feature": [0, 1, 0],
            "numeric_feature": [1.2, 3.4, 5.6],
        }
    )

    categorical_columns, binary_columns = infer_categorical_and_binary_columns(frame)

    assert categorical_columns == ["bool_feature", "zero_one_feature"]
    assert binary_columns == []

    preprocessor = NameValuePreprocessor.infer_from_dataframe(frame)
    assert [spec.feature_type for spec in preprocessor.feature_specs] == [
        FeatureType.CATEGORICAL,
        FeatureType.CATEGORICAL,
        FeatureType.NUMERICAL,
    ]


def test_low_cardinality_numeric_columns_are_not_categorical_without_label_maps() -> None:
    frame = pd.DataFrame(
        {
            "coded_numeric_without_map": [1, 2, 3, 1],
            "binary_without_map": [0, 1, 0, 1],
        }
    )

    categorical_columns, binary_columns = infer_categorical_and_binary_columns(frame)

    assert categorical_columns == ["binary_without_map"]
    assert binary_columns == []


def test_value_label_maps_drive_numerical_and_categorical_classification(tmp_path) -> None:
    label_maps = tmp_path / "nhanes_category_label_maps.json"
    label_maps.write_text(
        """
{
  "2011-2012|FILE|range_feature": {
    ".": "Missing",
    "0 to 10": "Range of Values",
    "777": "Refused"
  },
  "2011-2012|FILE|enum_feature": {
    ".": "Missing",
    "1": "Yes",
    "2": "No"
  }
}
""",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "range_feature": [1.0, 2.0, 3.0],
            "enum_feature": [1.0, 2.0, 1.0],
            "fallback_numeric": [1.2, 3.4, 5.6],
            "fallback_binary": [0, 1, 0],
        }
    )

    numerical_columns, categorical_columns, report = infer_columns_from_value_labels(
        frame,
        value_label_specs=load_value_label_specs(label_maps),
    )

    assert numerical_columns == ["range_feature"]
    assert categorical_columns == ["enum_feature"]
    assert report["missing_label_map_columns"] == ["fallback_binary", "fallback_numeric"]


def test_glu_prefixed_features_use_unprefixed_codebook_labels(tmp_path) -> None:
    label_maps = tmp_path / "nhanes_category_label_maps.json"
    label_maps.write_text(
        """
{
  "2015-2016|GLU_I|WTSAF2YR": {
    ".": "Missing",
    "0": "No Lab Result",
    "13612.331812 to 521632.18583": "Range of Values"
  },
  "2019-2020|P_GLU|WTSAFPRP": {
    ".": "Missing",
    "0": "No Lab Result",
    "4808.069916 to 741259.18875": "Range of Values"
  }
}
""",
        encoding="utf-8",
    )
    specs = load_value_label_specs(label_maps)
    frame = pd.DataFrame(
        {
            "GLU_WTSAF2YR": [1.0, 2.0, 3.0],
            "GLU_WTSAFPRP": [4.0, 5.0, 6.0],
        }
    )

    numerical_columns, categorical_columns, report = infer_columns_from_value_labels(
        frame,
        value_label_specs=specs,
    )

    assert numerical_columns == ["GLU_WTSAF2YR", "GLU_WTSAFPRP"]
    assert categorical_columns == []
    assert report["missing_label_map_columns"] == []


def test_glu_prefixed_features_use_unprefixed_metadata(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "cycle": "2015-2016",
                "module": "GLU",
                "file_name": "GLU_I",
                "variable_name": "WTSAF2YR",
                "variable_description": "Fasting weight",
                "sas_label": "Fasting Subsample 2 Year MEC Weight",
                "english_text": "Fasting Subsample 2 Year MEC Weight",
                "target": "Both males and females 12 YEARS - 150 YEARS",
                "unit": "",
                "hard_edits": "",
                "value_labels_json": '{".": "Missing", "0 to 10": "Range of Values"}',
            }
        ]
    ).to_csv(metadata_path, index=False)

    metadata = load_feature_metadata(metadata_path, ["GLU_WTSAF2YR"], category_label_maps_path=None)

    assert "GLU_WTSAF2YR" in metadata.descriptions
    assert "Fasting Subsample" in metadata.descriptions["GLU_WTSAF2YR"]


def test_codebook_parser_normalizes_lowercase_variable_suffixes() -> None:
    soup = BeautifulSoup(
        """
<html><body>
  <div class="Codebook">
    <h3 class="vartitle">MCQ240n - Age when lung cancer first diagnosed</h3>
    <dl>
      <dt>Variable Name:</dt><dd>MCQ240n</dd>
      <dt>SAS Label:</dt><dd>Age when lung cancer first diagnosed</dd>
      <dt>English Text:</dt><dd>How old was SP when lung cancer was first diagnosed?</dd>
    </dl>
    <table>
      <tr><th>Code or Value</th><th>Value Description</th><th>Count</th><th>Cumulative</th></tr>
      <tr><td>40 to 77</td><td>Range of Values</td><td>10</td><td>10</td></tr>
      <tr><td>80</td><td>80 years or older</td><td>1</td><td>11</td></tr>
      <tr><td>77777</td><td>Refused</td><td>0</td><td>11</td></tr>
    </table>
  </div>
</body></html>
""",
        "lxml",
    )

    _sections, variables = parse_codebook_page(soup)

    assert "MCQ240N" in variables
    assert variables["MCQ240N"]["value_labels"]["40 to 77"] == "Range of Values"


def test_mixed_range_features_are_sanitized_before_tokenization(tmp_path) -> None:
    label_maps = tmp_path / "nhanes_category_label_maps.json"
    label_maps.write_text(
        """
{
  "2011-2012|FILE|range_feature": {
    ".": "Missing",
    "1 to 10": "Range of Values",
    "15": "15 drinks or more",
    "80": "80 years or older",
    "666": "Less than 1 year",
    "777": "Refused",
    "888": "Could not obtain",
    "999": "Don't know"
  }
}
""",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "cycle": ["2011-2012"] * 8,
            "range_feature": [5.0, 15.0, 80.0, 666.0, 777.0, 888.0, 999.0, 123.0],
            "range_feature__missing_reason": [0] * 8,
        }
    )

    sanitize_range_feature_values(
        [frame],
        value_label_specs=load_value_label_specs(label_maps),
        missing_reason_codes={
            "not_missing": 0,
            "unknown_missing": 6,
            "refused": 7,
            "dont_know": 8,
        },
    )

    assert frame["range_feature"].tolist()[:4] == [5.0, 15.0, 80.0, 1.0]
    assert pd.isna(frame.loc[4, "range_feature"])
    assert pd.isna(frame.loc[5, "range_feature"])
    assert pd.isna(frame.loc[6, "range_feature"])
    assert pd.isna(frame.loc[7, "range_feature"])
    assert frame["range_feature__missing_reason"].tolist() == [0, 0, 0, 0, 7, 6, 8, 6]


def test_task_registry_v2_targets_are_produced_by_label_builder() -> None:
    registry = Path("src/downstream_tasks/task_registry_v2.json")

    target_columns = load_registry_target_columns(registry)
    labels = build_labels(
        pd.DataFrame(
            {
                "cycle": ["2017-2018"],
                "MCQ010": [1],
                "MCQ160A": [1],
                "MCQ160C": [1],
                "MCQ160E": [1],
                "MCQ160F": [1],
                "MCQ160G": [1],
                "MCQ160K": [2],
                "MCQ160L": [1],
                "MCQ160M": [1],
                "MCQ160N": [1],
                "MCQ160O": [2],
                "MCQ220": [1],
            }
        )
    )

    assert target_columns == [
        "label_arthritis",
        "label_asthma",
        "label_thyroid_condition",
        "label_any_cancer",
        "label_copd",
        "label_gout",
        "label_liver_condition",
        "label_stroke",
        "label_myocardial_infarction",
        "label_emphysema",
        "label_chd",
    ]
    assert set(target_columns).issubset(labels.columns)
