#!/usr/bin/env bash

RUNNER="pretrain_hpo_one_cluster"
EXPERIMENT_NAME="sgmet_v5_k7_pretrain_2cls_attnmi_hpo_fold0_cluster5"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# Data: fold 0 only, expert / cluster 5 only.
BASE_TOKEN_DIR="data/processed/tokenized_nhanes_v5"
FOLD=0
CLUSTER_ID=5
CLUSTER_CSV_NAME="feature_clusters_biolord_v5_k7_leiden.csv"
NUM_CLUSTERS=7

# Architecture.
NUM_SUMMARY_TOKENS=2
D_MODEL=64
N_LAYERS=2
N_HEADS=4
DROPOUT=0.1

# Masked-value reconstruction pretraining.
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FEATURE_MASK_PROB_MAX=0.25
SCHEMA_DROPOUT_PROB_MAX=0.10
CATEGORICAL_WEIGHT=1.0
CAT_LOSS="focal"
FOCAL_GAMMA=1.0
VAL_MASK_PASSES=3

# 2-CLS Attention-MI regularization.
SUMMARY_AUX_LOSS="attention_mi"
SUMMARY_AUX_WEIGHT=0.005
SUMMARY_AUX_WARMUP_EPOCHS=5
SUMMARY_MI_BETA=1.0
SUMMARY_MI_TEMPERATURE=1.0

# These are unused by Attention-MI but are kept explicit for the shared interface.
SUMMARY_ATTENTION_MARGIN=0.9
SUMMARY_OUTPUT_MARGIN=0.9

# Optimization schedule kept fixed during HPO.
LR_SCHEDULE="cosine"
WARMUP_FRAC=0.05
MIN_LR_RATIO=0.05

# HPO: optimizer hyperparameters only.
OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
OPTUNA_LR_LOW="1e-4"
OPTUNA_LR_HIGH="3e-3"
OPTUNA_WEIGHT_DECAY_LOW="1e-7"
OPTUNA_WEIGHT_DECAY_HIGH="1e-3"
OPTUNA_STARTUP_TRIALS=8
OPTUNA_PRUNER_STARTUP_TRIALS=5
OPTUNA_PRUNER_WARMUP_STEPS=10
OPTUNA_STUDY_NAME="sgmet_v5_k7_pretrain_2cls_attnmi_fold0_cluster5_lr_wd"

SEED=45
DEVICE="${DEVICE:-mps}"

MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
