#!/usr/bin/env bash
RUNNER="pretrain_5cv"
EXPERIMENT_NAME="sgmet_pretrain_semantic_1cls_smoke"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
CLUSTER_CSV_NAME="feature_clusters_biolord_v4_k8_leiden.csv"
NUM_SUMMARY_TOKENS=1
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-3}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-2}"
VAL_MASK_PASSES="${VAL_MASK_PASSES:-2}"

# Values are ordered by cluster ID 0,1,2,3,4,5,6.
CLUSTER_LRS="1.5e-3,4.0e-4,4.0e-4,4.0e-4,8.0e-4,4.0e-4,4.0e-4"
CLUSTER_CAT_WEIGHTS="2.0,1.0,3.0,2.0,3.0,1.0,1.0"
