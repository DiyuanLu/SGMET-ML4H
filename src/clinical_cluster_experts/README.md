# Clinical Cluster Experts (SGMET)

This package contains the core implementation of **SGMET**, a semantic-grouped Transformer for patient-level representation learning from tokenized tabular health data.

```text
tokenized patient table
    → enriched feature tokens
    → one Transformer expert per clinical feature group
    → one or more CLS summary tokens per expert
    → patient-level fusion Transformer
    → multitask binary prediction head
```

It supports expert pretraining, frozen or trainable pretrained experts, fully supervised end-to-end training, multiple expert summary tokens, five-fold evaluation, leave-one-cluster-out inference, and summary-token diagnostics.

Normal experiments should be launched through `scripts/sgmet_scripts/`. This folder contains the underlying Python implementation.

---

## Important Files

| File | Purpose |
|---|---|
| `model.py` | SGMET architecture: expert encoders, expert bank, fusion Transformer, prediction head, and full-model wrappers. |
| `token_builder.py` | Builds enriched feature tokens from semantic, value, type, and missingness channels. |
| `pretrain_expert_reconstruction.py` | Masked numerical/categorical reconstruction pretraining for one expert. |
| `train_supervised_downstream.py` | Supervised multitask training with frozen, trainable, pretrained, or random experts. |
| `inference_5cv.py` | Single-fold inference and five-fold evaluation. |
| `utils.py` | Device, cluster-assignment, embedding, batch, and checkpoint-loading utilities. |
| `summary_diagnostics.py` | Computes streaming diagnostics for multiple expert summary tokens. |
| `summary_auxiliary_losses.py` | Shared auxiliary losses for encouraging complementary expert summary tokens during pretraining and supervised training. Supports `none`, `attention_margin`, `attention_mi`, and `output_margin`. |
---

## Architecture

### 1. Enriched feature tokens

`FeatureTokenBuilder` converts each patient-feature cell into a dense token:
$$ z_{ij} =
\text{semantic}_{ij}
+
\text{value}_{ij}
+
\text{type}_{j}
+
\text{missingness}_{ij}.
$$

### 2. Feature-group experts

Each feature is assigned to a cluster in the cluster CSV. A `ClusterBranch`:

1. selects the features belonging to one cluster;
2. builds their enriched feature tokens;
3. sends them through a cluster-specific `ClusterExpertEncoder`.

The expert Transformer has no positional encoding, so the non-CLS feature tokens are treated as an unordered set.

### 3. Configurable CLS summary tokens

The number of expert summary tokens is controlled by:

```bash
--num-summary-tokens M
```

The default is `1`. For `M` summary tokens, one expert receives:

```text
[CLS_1, ..., CLS_M, feature_token_1, ..., feature_token_A]
```

and returns the first `M` encoded outputs.

For two summary tokens:

```text
[CLS_1, CLS_2, z_1, z_2, ...]
    → expert Transformer
    → [g_1, g_2]
```

The CLS tokens are stored as one trainable tensor of shape `[1, M, d_model]`. Its values are sampled independently; additional tokens are not copied from the first token.

`M=1` preserves the original one-token model interface and checkpoint shape.

### 4. Optional summary-token auxiliary losses

When an expert uses multiple summary tokens, the training scripts can optionally encourage the slots to learn complementary representations. The auxiliary loss is selected with: 

```bash 
--summary-aux-loss none|attention_margin|attention_mi|output_margin
# The default is `none`, so the original one-CLS and multi-CLS training behavior remains unchanged unless an auxiliary loss is explicitly enabled.

# Common arguments are:
--summary-aux-weight <lambda>
--summary-aux-warmup-epochs <epochs>
--summary-attention-margin <margin>
--summary-output-margin <margin>
--summary-mi-beta <beta>
--summary-mi-temperature <temperature>
```

The available objectives are:
- `none`: no summary-token regularization.
- `attention_margin`: penalizes excessive cosine overlap between CLS-to-feature attention distributions.
- `attention_mi`: encourages different CLS slots to specialize on different features while keeping all slots used overall.
- `output_margin`: penalizes excessive cosine similarity between the output summary-token representations.

For attention-based losses, the final-layer CLS-to-feature attention is obtained differentiably so that the auxiliary objective can update the expert encoder.


### 5. Expert bank and fusion

`ExpertBank` contains one branch per cluster. Its effective output is:

```text
M = 1: [batch, clusters, d_model]
M > 1: [batch, clusters, summary_tokens, d_model]
```

`PatientFusionTransformer` flattens cluster and summary-slot dimensions:

```text
[batch, clusters, summary_tokens, d_model]
    → [batch, clusters × summary_tokens, d_model]
```

With seven active groups and two summary tokens per group, fusion receives 14 active group-summary tokens. A patient-level fusion CLS token produces the final patient embedding, which is passed to `DownstreamPredictionHead`.

A cluster is masked from fusion when no feature in that group is available for the patient. Optional cluster embeddings give all summary slots from the same group a shared group identity.

---

## Missingness and availability

The code distinguishes three concepts:

| Concept | Meaning | Model behavior |
|---|---|---|
| Patient-level missingness | The feature exists, but this patient has no value. | The token remains visible because feature identity and missing reason can be informative. |
| Schema/view unavailability | The feature is absent, dropped, padded, or intentionally ignored. | The token is excluded through the Transformer attention mask. |
| Missing target label | A downstream task label is unavailable. | It contributes to neither loss nor metrics. |

Feature dropout modifies feature availability, while cluster dropout removes complete active expert branches from the batch-level fusion input. Neither operation rewrites patient-level missingness.

---

## Expert pretraining

`pretrain_expert_reconstruction.py` pretrains one cluster branch at a time using masked value reconstruction.

```text
local cluster batch
    → optional schema dropout
    → select observed reconstruction targets
    → hide their value channels
    → encode remaining available tokens
    → reconstruct only artificially masked values
```

Key behavior:

- targets must be non-missing and schema-available;
- true patient-missing tokens may remain visible;
- schema-dropped features are neither visible nor targets;
- at least one observed context value is retained when possible;
- numerical and categorical values use separate reconstruction heads;
- categorical loss supports CE, weighted CE, focal, and weighted focal loss.

For multiple summary tokens, each masked-feature query attends over all expert summaries before reconstruction. With one summary token, this reduces to the original path.

Pretraining supports the same configurable summary-token auxiliary losses as supervised training. The optimization objective is:
```text
reconstruction loss + effective auxiliary weight × summary auxiliary loss
```
The auxiliary weight can be linearly warmed up during the first training epochs with `--summary-aux-warmup-epochs`. 

### Example: two CLS tokens with Attention-MI

```bash
python -m src.clinical_cluster_experts.pretrain_expert_reconstruction \
  --token-dir data/processed/tokenized_nhanes_v5/cv_splits/fold0 \
  --cluster-csv data/processed/tokenized_nhanes_v5/cv_splits/fold0/feature_clusters_biolord_v5_k7_leiden.csv \
  --num-clusters 7 \
  --cluster-id 0 \
  --num-summary-tokens 2 \
  --summary-aux-loss attention_mi \
  --summary-aux-weight 0.005 \
  --summary-aux-warmup-epochs 5 \
  --summary-mi-beta 1.0 \
  --summary-mi-temperature 1.0 \
  --epochs 50 \
  --batch-size 128 \
  --d-model 64 \
  --n-layers 2 \
  --n-heads 4 \
  --feature-mask-prob-max 0.25 \
  --schema-dropout-prob-max 0.10 \
  --device mps \
  --output-dir outputs/example_pretrain_cluster0 \
  --log-dir runs/example_pretrain_cluster0
```

### Outputs

```text
cluster_<id>_best.pt
cluster_<id>_last.pt
cluster_<id>_metrics.csv
```

---

## Supervised training

All downstream modes use `train_supervised_downstream.py`.

| Mode | Important arguments | Trainable modules |
|---|---|---|
| End-to-end from scratch | `--allow-random-active-branches --train-experts` | Experts, fusion, head |
| Pretrained frozen experts | `--branch-checkpoint ... --strict-branch-load` | Fusion and head |
| Pretrained trainable experts | Checkpoints plus `--train-experts --expert-lr ...` | Experts, fusion, head |
| Continue supervised training | `--init-supervised-checkpoint ...` | Determined by `--train-experts` |

The trainer also supports:

- one or multiple summary tokens per expert;
- optional `none`, `attention_margin`, `attention_mi`, or `output_margin` summary-token auxiliary losses;
- separate expert and fusion/head learning rates;
- AdamW parameter grouping;
- linear warmup and cosine decay;
- weighted BCE or focal loss;
- feature and cluster dropout;
- early stopping and validation checkpoint selection;
- Optuna search over learning rate, weight decay, and maximum cluster dropout;
- optional validation summary-token diagnostics.

### Cluster dropout

`--cluster-dropout-prob-max` defines the maximum fraction of active clusters that may be removed from a training batch. For `K` active clusters: `max_clusters_to_drop = min(floor(cluster_dropout_prob_max × K), K - 1)`. For each training batch, the trainer then:
1. samples an integer number of clusters to drop uniformly from 0 through max_clusters_to_drop;
2. randomly chooses exactly that many active clusters;
3. removes those complete expert branches from the fusion input for that batch.

At least one cluster is always retained.


### End-to-end two-CLS Attention-MI example

```bash
python -m src.clinical_cluster_experts.train_supervised_downstream \
  --token-dir data/processed/tokenized_nhanes_v5/cv_splits/fold0 \
  --cluster-csv data/processed/tokenized_nhanes_v5/cv_splits/fold0/feature_clusters_biolord_v5_k7_leiden.csv \
  --num-clusters 7 \
  --active-clusters 0,1,2,3,4,5,6 \
  --ignore-clusters none \
  --allow-random-active-branches \
  --train-experts \
  --num-summary-tokens 2 \
  --summary-aux-loss attention_mi \
  --summary-aux-weight 0.005 \
  --summary-aux-warmup-epochs 5 \
  --summary-mi-beta 1.0 \
  --summary-mi-temperature 1.0 \
  --epochs 80 \
  --batch-size 128 \
  --d-model 64 \
  --expert-n-layers 2 \
  --expert-n-heads 4 \
  --fusion-n-layers 2 \
  --fusion-n-heads 4 \
  --lr 8e-4 \
  --expert-lr 8e-4 \
  --weight-decay 5e-4 \
  --binary-loss focal \
  --focal-gamma 1.0 \
  --feature-dropout-prob-max 0.10 \
  --cluster-dropout-prob-max 0.90 \
  --selection-metric macro_auroc \
  --early-stopping-patience 15 \
  --summary-diagnostics \
  --device mps \
  --output-dir outputs/example_supervised/fold0 \
  --log-dir runs/example_supervised/fold0 \
  --no-cluster-embedding
```
The supervised optimization loss is: `supervised prediction loss + effective auxiliary weight × summary auxiliary loss`

### Loading pretrained branches

Supply one checkpoint for each active cluster:

```bash
--branch-checkpoint 0=path/to/cluster_0_best.pt \
--branch-checkpoint 1=path/to/cluster_1_best.pt \
... \
--branch-checkpoint 6=path/to/cluster_6_best.pt \
--strict-branch-load
```

Cluster IDs, feature indices, and summary-token counts must match the current model configuration.

### Outputs

```text
best.pt
last.pt
supervised_frozen_metrics.csv      # frozen experts
supervised_finetune_metrics.csv    # trainable experts
```

With `--summary-diagnostics`:

```text
summary_token_token_stats.csv
summary_token_pair_stats.csv
```

TensorBoard events are written below the supplied `--log-dir` in a timestamped run directory.

---

## Summary-token diagnostics

`summary_diagnostics.py` collects dataset-level statistics using streaming sufficient statistics on CPU. It does not retain all patient embeddings.

For each available group and summary slot, it records:

- mean L2 norm;
- mean per-dimension variance across patients;
- pairwise cosine similarity between summary slots;
- pairwise overlap between final-layer CLS-to-feature attention distributions.

Interpretation:

- cosine similarity near one may indicate representational collapse;
- attention overlap near one means that two slots attend to nearly the same features;
- near-zero variance may indicate a dead or patient-invariant token;
- norm is less informative by itself because expert outputs pass through LayerNorm.

The optional dataset-level summary diagnostics are normally collected on validation or test data rather than training batches.

Attention-based auxiliary losses are different: `attention_margin` and `attention_mi` request differentiable final-layer CLS-to-feature attention during training so that the regularizer can update the expert encoder. The attention replay disables attention dropout for the returned attention map while retaining the autograd graph.

For diagnostics and auxiliary losses, attention is averaged across heads, restricted to available features, and renormalized over feature positions. Diagnostic attention-overlap statistics exclude rows where the comparison is not informative.


Recommended use:

| Stage | Recommendation |
|---|---|
| Training batches | Do not collect: dropout and parameter updates make the measurement noisy. |
| Validation | Enable for selected runs to monitor collapse across epochs. |
| Test inference | Collect once from the selected checkpoint for final reporting. |
| Broad HPO | Normally disable to reduce evaluation cost. |

Flags:

```bash
--summary-diagnostics                     # supervised validation
--summary-diagnostics-dir <directory>     # single-fold inference
--save-summary-diagnostics                # five-fold inference
```

---

## Inference and five-fold evaluation

`inference_5cv.py` supports single-fold evaluation and a five-fold wrapper. It reports macro and per-task metrics and can optionally save predictions, metrics, model parameter reports, and summary-token diagnostics.

The number of summary tokens is normally read from the supervised checkpoint. Legacy checkpoints without this field fall back to one token.

### Five-fold example

```bash
python -m src.clinical_cluster_experts.inference_5cv \
  --run-cv \
  --cv-root data/processed/tokenized_nhanes_v5/cv_splits \
  --cluster-csv-name feature_clusters_biolord_v5_k7_leiden.csv \
  --checkpoint-root outputs/example_supervised \
  --split test \
  --num-clusters 7 \
  --active-clusters 0,1,2,3,4,5,6 \
  --ignore-clusters none \
  --batch-size 128 \
  --device mps \
  --no-cluster-embedding \
  --save-cv-results \
  --save-summary-diagnostics
```

Main five-fold outputs:

```text
macro_metrics_per_fold.csv
per_target_metrics_per_fold.csv
per_target_metrics_mean_std.csv
cv_metrics.json
```

---

## Expected data layout

The current v5 data are organized as self-contained fold directories:

```text
data/processed/tokenized_nhanes_v5/
└── cv_splits/
    ├── fold0/
    ├── fold1/
    ├── fold2/
    ├── fold3/
    └── fold4/
```
Each fold directory contains its own tokens, targets, metadata, embeddings, and cluster assignment:

```text
tokenizer_metadata.pt
train_tokens.pt
val_tokens.pt
test_tokens.pt
train_targets.pt
val_targets.pt
test_targets.pt
feature_bge_embeddings.pt
category_bge_embeddings.pt
feature_clusters_biolord_v5_k7_leiden.csv
```

The cluster CSV must contain:

```text
feature_name,cluster_id
```

Every tokenizer feature must occur exactly once. Numeric cluster IDs are preserved without remapping.

The current v5 semantic clustering contains seven clusters, all of which are active:
```bash
--num-clusters 7 \
--active-clusters 0,1,2,3,4,5,6 \
--ignore-clusters none
```

`--num-clusters` remains eight because cluster ID 7 is still present in the assignment file.

---

## Experiment launchers

See `scripts/sgmet_scripts/README.md` for all available experiment configurations.