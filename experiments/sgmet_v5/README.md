# SGMET v5 experiment scripts

This folder reproduces the NHANES-only results reported in the paper: the
SGMET end-to-end recipe, the parameter-matched flat transformer, and the
XGBoost / FT-Transformer / TabPFN baselines (clean and missingness-augmented
variants), plus the leave-one-domain-out (LODO) robustness evaluation.

Large patient tensors are not included in this release; the dataset
preparation scripts below rebuild them from public NHANES data.

## Dataset contract

- Cohort: 39,658 NHANES adults, 149 physical feature columns.
- Five outer test folds; each outer-training pool is re-stratified into 64%
  train / 16% validation.
- Feature-to-domain maps are BioLORD-embedding cosine-kNN (`k=17`) plus Leiden
  clustering (`config/`, `cluster_maps/`).

## Building the dataset

1. Download and prepare NHANES data with the scripts in [`../../scripts`](../../scripts).
2. Tokenize the prepared table with [`src/tokenizer/tokenize_nhanes.py`](../../src/tokenizer/tokenize_nhanes.py).
3. Discover semantic feature domains with [`src/clustering`](../../src/clustering)
   (`build_feature_semantics.py`, `cluster_evaluation.py`, `sweep_cluster_umap.py`).
4. Materialize the five outer folds and re-stratified train/validation splits
   with `recarve_restratified_folds.py`.
5. Package a fold set for training/validation with `package_dataset.py` and
   sanity-check it with `validate_package.py`.

`build_active_maps.py` reproduces the active (post-exclusion) feature-cluster
maps from a fold token directory and the BioLORD feature-embedding file.

## SGMET and the flat-transformer baseline

Both are produced by the same end-to-end trainer,
[`src/clinical_cluster_experts/train_supervised_downstream.py`](../../src/clinical_cluster_experts/train_supervised_downstream.py),
switched between modular (`--num-clusters 7`) and flat (`--num-clusters 1`)
by one flag. From the repository root:

```bash
python experiments/sgmet_v5/validate_package.py /path/to/tokenized_nhanes
bash experiments/sgmet_v5/run_e2e_cv.sh /path/to/tokenized_nhanes
```

`run_e2e_cv.sh` runs all five K=7 folds first, then the K=10 folds, using the
fixed hyperparameters in [Section 4 of the paper](../../manuscript). See
[`scripts/sgmet_scripts/README.md`](../../scripts/sgmet_scripts/README.md) for
preconfigured launcher scripts covering pretraining, ablations, and HPO.

## Baselines

| Baseline | Clean | Missingness-augmented |
| --- | --- | --- |
| XGBoost (Optuna-tuned) | `xgboost_v5_fivefold_optuna25.py` | `xgboost_missingness_augmented_v5.py` |
| FT-Transformer | `ft_transformer_v5.py` | `ft_transformer_missingness_augmented_v5.py`, `ft_transformer_missingness_validation_v5.py` |
| TabPFN | `tabpfn_missingness_v5.py` (clean and augmented contexts) | |

`ft_reference_lodo_v5.py` and `221_ft_fivefold_test_reference.py` score the
FT-Transformer checkpoints under the leave-one-domain-out protocol and on the
held-out test folds, respectively. `xgboost_fit_common.py` holds shared
Optuna/XGBoost fitting utilities used by both XGBoost scripts.

TabPFN evaluation additionally requires
[`requirements-tabpfn-optional.txt`](requirements-tabpfn-optional.txt)
(the `tabpfn_client` API package) and a TabPFN API key.

## Robustness protocol

Both augmented baselines sample a per-batch cell-drop probability from
U(0, 0.10) and a removed-domain count uniformly from {0, ..., 6}; a domain's
removal applies to every patient in that (virtual) batch. Validation stays
clean for early stopping; LODO evaluation applies deterministic, one-domain
masks to the trained models without retraining. SGMET is trained only with
the augmented (feature- and domain-dropout) protocol, so it is compared
against each baseline's augmented variant.
