# Mapping from the previous scripts

| Previous script | What it executed | Replacement config |
|---|---|---|
| `train_5cv_supervised_e2e_from_scratch.sh` | Random grouping **and** random expert initialization, with all experts trained end-to-end. This changed two factors simultaneously. | `supervised/e2e_scratch_random_m1`; use `supervised/e2e_scratch_semantic_m1` for the isolated no-pretraining condition. |
| `train_5cv_supervised_trainable_experts.sh` | Loads semantic pretrained experts and fine-tunes them with LR `8e-5`; fusion/head use `8e-4`. | `supervised/pretrained_trainable_semantic_m1` |
| `train_5cv_supervised_frozen_experts.sh` | Loads semantic pretrained experts, freezes them, and trains fusion/head only. | `supervised/pretrained_frozen_semantic_m1` |
| `pretrain_5cv_all_experts.sh` | Runs 35 full expert-pretraining jobs: 5 folds × 7 active experts, 50 epochs each, with cluster-specific LR and categorical weights. | `pretrain/semantic_m1_full` |
| `pretrain_5cv_all_experts_smoke.sh` | Runs the same 35 combinations but only 2 epochs, 3 train batches and 2 validation batches. | `pretrain/semantic_m1_smoke` |
| `hpo_supervised_frozen_experts_one_fold.sh` | Fold-0 Optuna search for fusion/head LR and weight decay with frozen pretrained experts. The old file referenced the obsolete `train_supervised_frozen_experts.py`; the new runner uses unified `train_supervised_downstream.py`. | `hpo/frozen_semantic_m1_fold0` |
| `inference_sgmet_cluster_ablation.sh` | Runs full-schema and seven leave-one-cluster-out conditions across 5 folds, then aggregates paired performance drops. | `inference/cluster_ablation` |
