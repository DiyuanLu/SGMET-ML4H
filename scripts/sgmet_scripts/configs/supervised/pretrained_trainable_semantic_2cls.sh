#!/usr/bin/env bash
RUNNER="supervised_5cv"
EXPERIMENT_NAME="sgmet_pretrained_trainable_semantic_2cls_fd10_cd90"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

# Requires experts pretrained with NUM_SUMMARY_TOKENS=2.
EXPERT_INIT="pretrained"
TRAIN_EXPERTS=1
PRETRAIN_ROOT="${PRETRAIN_ROOT:-outputs/${DATE_TAG}_sgmet_pretrain_semantic_2cls}"
CLUSTER_CSV_NAME="feature_clusters_biolord_v4_k8_leiden.csv"
NUM_SUMMARY_TOKENS=2
SUMMARY_DIAGNOSTICS=1

EPOCHS="${EPOCHS:-80}"
LR="${LR:-8e-4}"
EXPERT_LR="${EXPERT_LR:-8e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
FEATURE_DROPOUT_PROB_MAX="${FEATURE_DROPOUT_PROB_MAX:-0.10}"
CLUSTER_DROPOUT_PROB_MAX="${CLUSTER_DROPOUT_PROB_MAX:-0.90}"
