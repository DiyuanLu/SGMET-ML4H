#!/usr/bin/env bash
RUNNER="supervised_5cv"
EXPERIMENT_NAME="sgmet_pretrained_frozen_semantic_1cls_fd10_cd90"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

EXPERT_INIT="pretrained"
TRAIN_EXPERTS=0
PRETRAIN_ROOT="${PRETRAIN_ROOT:-outputs/20260712_5cv_pretrained_branches_cosine50}"
CLUSTER_CSV_NAME="feature_clusters_biolord_v4_k8_leiden.csv"
NUM_SUMMARY_TOKENS=1
SUMMARY_DIAGNOSTICS=0

EPOCHS="${EPOCHS:-80}"
LR="${LR:-8e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
FEATURE_DROPOUT_PROB_MAX="${FEATURE_DROPOUT_PROB_MAX:-0.10}"
CLUSTER_DROPOUT_PROB_MAX="${CLUSTER_DROPOUT_PROB_MAX:-0.90}"
