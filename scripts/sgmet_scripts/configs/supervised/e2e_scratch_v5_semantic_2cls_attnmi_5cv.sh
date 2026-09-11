#!/usr/bin/env bash

RUNNER="supervised_5cv"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"


# Separate name prevents overwriting the earlier fold-0 calibration run.
#EXPERIMENT_NAME="sgmet_e2e_scratch_semantic_2cls_attnmi_b1_t1_lam0p005_fd10_cd90_5cv"
EXPERIMENT_NAME="sgmet_e2e_scratch_semantic_2cls_attnmi_b1_t1_lam0p005_fd10_cd90_5cv_random_seed42"

# Data
BASE_TOKEN_DIR="data/processed/tokenized_nhanes_v5"
#CLUSTER_CSV_NAME="feature_clusters_biolord_v5_k7_leiden.csv"
CLUSTER_CSV_NAME="feature_clusters_random_seed42_v5_k7.csv"

NUM_CLUSTERS=7
ACTIVE_CLUSTERS="0,1,2,3,4,5,6"
IGNORE_CLUSTERS="none"

# Architecture and initialization
EXPERT_INIT="random"
TRAIN_EXPERTS=1
NUM_SUMMARY_TOKENS=2
SUMMARY_DIAGNOSTICS=1
USE_CLUSTER_EMBEDDING=0

# Attention-MI auxiliary objective
SUMMARY_AUX_LOSS="attention_mi"
SUMMARY_AUX_WEIGHT="${SUMMARY_AUX_WEIGHT:-0.005}"
SUMMARY_AUX_WARMUP_EPOCHS="${SUMMARY_AUX_WARMUP_EPOCHS:-5}"
SUMMARY_MI_BETA="${SUMMARY_MI_BETA:-1.0}"
SUMMARY_MI_TEMPERATURE="${SUMMARY_MI_TEMPERATURE:-1.0}"

# Unused by this loss, but forwarded by the generic runner
SUMMARY_ATTENTION_MARGIN="0.90"
SUMMARY_OUTPUT_MARGIN="0.90"

# Keep identical to the existing two-CLS baseline
EPOCHS="${EPOCHS:-80}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"

LR="${LR:-0.0008341106432362088}"
EXPERT_LR="${EXPERT_LR:-0.0008341106432362088}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.1916299962955147e-06}"

FEATURE_DROPOUT_PROB_MAX="${FEATURE_DROPOUT_PROB_MAX:-0.10}"
CLUSTER_DROPOUT_PROB_MAX="${CLUSTER_DROPOUT_PROB_MAX:-0.90}"

SEED=42