#!/usr/bin/env bash
RUNNER="supervised_5cv"
EXPERIMENT_NAME="sgmet_e2e_scratch_random_groups_1cls_fd10_cd90"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# Random-grouping ablation. It matches the semantic 1-CLS setup except
# for the feature-to-cluster assignment CSV.
EXPERT_INIT="random"
TRAIN_EXPERTS=1
CLUSTER_CSV_NAME="feature_clusters_biolord_v4_k8_leiden_random_balanced_seed42.csv"
NUM_SUMMARY_TOKENS=1
SUMMARY_DIAGNOSTICS=0

EPOCHS="${EPOCHS:-80}"
LR="${LR:-8e-4}"
EXPERT_LR="${EXPERT_LR:-8e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
FEATURE_DROPOUT_PROB_MAX="${FEATURE_DROPOUT_PROB_MAX:-0.10}"
CLUSTER_DROPOUT_PROB_MAX="${CLUSTER_DROPOUT_PROB_MAX:-0.90}"
