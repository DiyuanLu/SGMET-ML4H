# SGMET: Semantic-Group Modular Encoders for Robust Health Representation Learning with Tabular Data

Code accompanying the ML4H 2026 submission "SGMET: Semantic-Group Modular
Encoders for Robust Health Representation Learning with Tabular Data".

SGMET enriches each health-survey feature with its value, type, structured
missingness, and BioLORD codebook semantics; groups features into named
clinical domains discovered from those semantics; and fuses domain-specific
encoders for multi-task disease prediction that degrades gracefully when an
entire clinical domain is missing at inference time.

This release contains only the NHANES-side method, training, and evaluation
code. It does not include:

- **Any data.** NHANES is public but redistribution-restricted; the scripts
  under [`scripts/`](scripts) and [`src/nhanes`](src/nhanes) rebuild the
  processed tables and tokenized tensors from the public source.
- **The KNHANES cross-survey transfer experiments.** These results are not
  part of this submission's reported claims (see the manuscript) and are
  omitted here.
- **Manuscript figure-generation scripts.** Scripts that redraw paper figures
  from already-computed summary statistics are not included, since several of
  them read point estimates copied from the manuscript text rather than
  recomputing them from raw run outputs.

## Repository layout

```text
src/
  nhanes/                  NHANES download and feature-table preparation
  tokenizer/               Meta-data enriched feature tokenization (Sec. 3.2)
  clustering/               BioLORD-embedding semantic domain discovery (Sec. 3.3)
  clinical_cluster_experts/ SGMET domain encoders, fusion, and end-to-end trainer (Sec. 3.4-3.5)
  name_value_transformer/  Flat (non-modular) parameter-matched comparison architecture
  baseline_models/         XGBoost pipeline and shared evaluation utilities
experiments/sgmet_v5/       Frozen experiment scripts: SGMET, flat transformer,
                             XGBoost, FT-Transformer, TabPFN, and the
                             leave-one-domain-out (LODO) robustness protocol
scripts/                    Data download/preparation wrappers and the
                             sgmet_scripts/ experiment launcher
tests/                      Unit tests for tokenization, clustering, and the
                             flat-transformer checkpoint/data utilities
```

Each subfolder's README documents its scripts in more detail.

## Setup

```bash
conda env create -f environment.yml
conda activate sgmet
```

## Quickstart

```bash
# 1. Download and prepare NHANES data
bash scripts/download_data.sh
bash scripts/download_codebooks.sh
bash scripts/prepare_data.sh

# 2. Tokenize features, discover semantic domains, and build CV folds
#    (see src/tokenizer, src/clustering, experiments/sgmet_v5/README.md)

# 3. Train SGMET (or the flat-transformer control) and the baselines
bash experiments/sgmet_v5/run_e2e_cv.sh /path/to/tokenized_nhanes
```

See [`experiments/sgmet_v5/README.md`](experiments/sgmet_v5/README.md) for the
full data-preparation-to-evaluation pipeline and baseline scripts, and
[`scripts/sgmet_scripts/README.md`](scripts/sgmet_scripts/README.md) for the
preconfigured experiment launcher.

## Tests

```bash
python -m pytest tests/
```

## Data availability

NHANES cycles are publicly available from the CDC
(https://wwwn.cdc.gov/nchs/nhanes/). No data is redistributed with this code.

## License

See [LICENSE](LICENSE).
