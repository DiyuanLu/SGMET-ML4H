# SGMET experiment scripts

This folder contains the shell configurations and shared runners used to launch
SGMET pretraining, supervised training, HPO, and inference experiments.

The main idea is simple:

- **`configs/` describes what to run**: the experimental condition and its parameters.
- **`runners/` describes how to run it**: fold loops, checkpoint loading, output creation, and Python commands.
- **`run.sh` is the single entry point** for every experiment.

The launcher automatically finds the repository root, activates the `di-lab`
Conda environment, and works even when called outside the repository root.

## Folder structure

```text
scripts/sgmet_scripts/
├── README.md
├── run.sh
├── lib/
│   └── common.sh
├── runners/
│   ├── supervised_5cv.sh
│   ├── pretrain_5cv.sh
│   ├── supervised_hpo_one_fold.sh
│   └── cluster_ablation_5cv.sh
└── configs/
    ├── supervised/
    ├── pretrain/
    ├── hpo/
    └── inference/
```

### Naming convention

1. **`cls`** states the number of learnable CLS summary tokens produced by each feature-group expert. E.g `1cls`: one CLS token per expert; the original SGMET architecture. `2cls`: two independently initialized CLS tokens per expert; the fusion Transformer receives two representation tokens from each active group.

2. **`fd`** state the maximum feature-dropout. Our currently use SGMET 0.10
3. **`cd`** state the maximum cluster-dropout. Our currently use SGMET 0.90

## Available experiments

### Supervised five-fold experiments

| Config name | Expert initialization | Grouping | Expert training | CLS tokens/group | Purpose |
|---|---|---|---|---:|---|
| `supervised/e2e_scratch_random_1cls` | Random | Random balanced | End-to-end | 1 | Random-grouping ablation |
| `supervised/e2e_scratch_semantic_1cls` | Random | Semantic | End-to-end | 1 | Standard fully supervised SGMET without pretraining |
| `supervised/e2e_scratch_semantic_2cls` | Random | Semantic | End-to-end | 2 | Multi-CLS ablation without expert pretraining; records validation token diagnostics |
| `supervised/pretrained_frozen_semantic_1cls` | Pretrained | Semantic | Frozen | 1 | Original frozen-expert downstream setting |
| `supervised/pretrained_trainable_semantic_1cls` | Pretrained | Semantic | Fine-tuned | 1 | End-to-end downstream fine-tuning of pretrained experts |
| `supervised/pretrained_trainable_semantic_2cls` | Pretrained with 2 CLS tokens | Semantic | Fine-tuned | 2 | Multi-CLS model with pretrained, trainable experts; records token diagnostics |

### Expert-pretraining experiments

Each full pretraining config runs all seven active experts for all five folds.
The smoke configs use only a few batches to validate the pipeline.

| Config name | CLS tokens/group | Scope | Purpose |
|---|---:|---|---|
| `pretrain/semantic_1cls_smoke` | 1 | 2 epochs, limited batches | Fast integration test for 1-CLS expert pretraining |
| `pretrain/semantic_1cls_full` | 1 | Full five-fold pretraining | Produce 1-CLS expert checkpoints |
| `pretrain/semantic_2cls_smoke` | 2 | 2 epochs, limited batches | Fast integration test for 2-CLS expert pretraining |
| `pretrain/semantic_2cls_full` | 2 | Full five-fold pretraining | Produce 2-CLS expert checkpoints |

### HPO and inference

| Config name | Purpose |
|---|---|
| `hpo/frozen_semantic_1cls_fold0` | Fold-0 Optuna search for learning rate and weight decay using frozen pretrained 1-CLS experts |
| `inference/cluster_ablation` | Five-fold full-schema and leave-one-cluster-out inference, followed by result aggregation |

## Running experiments

Run commands from the repository root or any other directory.

### List all available configs

```bash
bash scripts/sgmet_scripts/run.sh --list
```

### Useful controls:
| Variable | Effect |
|---|---|
| `DRY_RUN=1` | Print commands only |
| `BACKGROUND=1` | Launch with `nohup` and write log/PID files |
| `DATE_TAG=YYYYMMDD` | Override the output date prefix |
| `FOLDS=0,1,...` | Select folds |
| `MAX_TRAIN_BATCHES=N` | Limit training batches per epoch |
| `MAX_VAL_BATCHES=N` | Limit validation batches per epoch |
| `PRETRAIN_ROOT=...` | Select pretrained expert checkpoints |
| `CHECKPOINT_ROOT=...` | Select supervised checkpoints for inference |
| `DEVICE=cpu/mps/cuda` | Select the execution device |


### Preview an experiment

- `DRY_RUN=1` prints the resolved Python commands without executing them. 
- Try fold0. emove FOLDS=0 to check all fold. 
- Using temporary output paths prevents the dry run from creating experiment folders under your real outputs/ and runs/ directories.
```bash
DRY_RUN=1 \
FOLDS=0 \
OUT_ROOT=/tmp/sgmet_dryrun_outputs \
TB_ROOT=/tmp/sgmet_dryrun_runs \
bash scripts/sgmet_scripts/run.sh \
  supervised/e2e_scratch_semantic_1cls
```

### Run a quick supervised smoke test

```bash
FOLDS=0 EPOCHS=1 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=2 \
OUT_ROOT=/tmp/sgmet_smoke_outputs \
TB_ROOT=/tmp/sgmet_smoke_runs \
bash scripts/sgmet_scripts/run.sh \
  supervised/e2e_scratch_semantic_1cls
```

### Run in the foreground

```bash
bash scripts/sgmet_scripts/run.sh \
  supervised/e2e_scratch_semantic_2cls
```

### Run in the background

```bash
DATE_TAG=20260730 BACKGROUND=1 \
bash scripts/sgmet_scripts/run.sh \
  supervised/e2e_scratch_semantic_2cls
```

The launcher writes:

```text
logs/<date>_<experiment-name>.out
logs/<date>_<experiment-name>.pid
```

Useful checks after launching e.g.: 
```bash
tail -n 60 -f logs/<date>_<experiment-name>.out
```

On macOS, background runs use `caffeinate` when available.

### Pretraine two-CLS experts

Run the smoke test first:

```bash
bash scripts/sgmet_scripts/run.sh pretrain/semantic_2cls_smoke
```

Then run full pretraining:

```bash
BACKGROUND=1 bash scripts/sgmet_scripts/run.sh \
  pretrain/semantic_2cls_full
```

A downstream 2-CLS model must load expert checkpoints that were also pretrained
with `NUM_SUMMARY_TOKENS=2`.

### Fine-tune pretrained two-CLS experts

Pass the pretraining output root when it differs from the config default:

```bash
PRETRAIN_ROOT=outputs/20260730_sgmet_pretrain_semantic_2cls \
BACKGROUND=1 bash scripts/sgmet_scripts/run.sh \
  supervised/pretrained_trainable_semantic_2cls
```

### Run the cluster-removal ablation

```bash
CHECKPOINT_ROOT=outputs/<supervised-experiment> \
  bash scripts/sgmet_scripts/run.sh inference/cluster_ablation
```

To save multi-CLS cosine similarity, variance, norm, and attention-overlap
statistics during inference:

```bash
CHECKPOINT_ROOT=outputs/<supervised-experiment> \
SAVE_SUMMARY_DIAGNOSTICS=1 \
  bash scripts/sgmet_scripts/run.sh inference/cluster_ablation
```

## Temporary overrides

Values written in configs or runners as `${VARIABLE:-default}` can be overridden
without editing the file. Common examples are:

```bash
DATE_TAG=20260730 \
FOLDS=0,1 \
EPOCHS=10 \
LR=5e-4 \
BACKGROUND=1 \
  bash scripts/sgmet_scripts/run.sh \
  supervised/e2e_scratch_semantic_2cls
```


## Outputs and reproducibility

Each experiment output root contains:

```text
resolved_experiment_config.sh
run_manifest.txt
```

The manifest records the timestamp, repository path, Git commit, and current Git
working-tree status. Fold-specific model checkpoints and metrics are written
under `fold0/`, `fold1/`, and so on.

The project keeps `NUM_CLUSTERS=8` because the cluster CSV contains IDs `0` to
`7`. Clusters `0` to `6` are active, while cluster `7` is loaded and deliberately
ignored. The model therefore uses seven active feature groups.

## Adding a new experiment

Most new ablations require only a new config file.

### 1. Copy the closest existing config

For example:

```bash
cp scripts/sgmet_scripts/configs/supervised/e2e_scratch_semantic_2cls.sh \
   scripts/sgmet_scripts/configs/supervised/e2e_scratch_semantic_3cls.sh
```

### 2. Change only the experimental factors

At minimum, update:

```bash
EXPERIMENT_NAME="sgmet_e2e_scratch_semantic_3cls_fd10_cd90"
NUM_SUMMARY_TOKENS=3
```

Keep all unrelated settings unchanged when the goal is a controlled ablation.
Use a descriptive comment to state exactly how the new config differs from its
reference condition.

### 3. Check the required runner variables

A supervised config normally defines:

```bash
RUNNER="supervised_5cv"
EXPERIMENT_NAME="..."
EXPERT_INIT="random"       # or "pretrained"
TRAIN_EXPERTS=1            # 0 freezes pretrained experts
CLUSTER_CSV_NAME="..."
NUM_SUMMARY_TOKENS=1
SUMMARY_DIAGNOSTICS=0
```

A pretraining config normally defines:

```bash
RUNNER="pretrain_5cv"
EXPERIMENT_NAME="..."
CLUSTER_CSV_NAME="..."
NUM_SUMMARY_TOKENS=1
```

It also provides the per-cluster learning rates and categorical-loss weights.

### 4. Validate before launching

First check shell syntax:

```bash
bash -n scripts/sgmet_scripts/configs/supervised/e2e_scratch_semantic_3cls.sh
```

Then inspect the resolved command:

```bash
DRY_RUN=1 FOLDS=0 \
  bash scripts/sgmet_scripts/run.sh \
  supervised/e2e_scratch_semantic_3cls
```

Finally, run a limited smoke test before starting all five folds.

### When to add a new runner

Add a new file under `runners/` only when the execution procedure itself changes,
for example when a new experiment requires a different fold loop or a different
Python entry point. Parameter changes and ordinary ablations belong in
`configs/`, not in new runners.
