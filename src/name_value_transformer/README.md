# Flat Name-Value Transformer

This package defines the flat (non-modular) comparison architecture referenced
in the paper as the "matched flat transformer": all selected features enter a
single Transformer sequence at once, using the same BioLORD name/category
embeddings, value channels, missingness channels, and pre-norm Transformer
block as the SGMET domain encoders. There are no expert branches, group
tokens, cluster embeddings, or fusion Transformer.

## What is included

- `model.py`: the `FeatureTokenTransformer` architecture and prediction head.
- `checkpoint.py`, `data.py`: checkpoint (de)serialization and batch-assembly
  helpers, used by `src/baseline_models/baseline_model_evaluation.py` to load
  and evaluate flat-transformer checkpoints.

## Training

The standalone `train.py` / `pretrain.py` / `inference.py` entry points that
originally shipped with this package have been superseded by the unified
end-to-end trainer in
[`src/clinical_cluster_experts/train_supervised_downstream.py`](../clinical_cluster_experts/train_supervised_downstream.py),
which produces both the modular SGMET model and this flat baseline from the
same code path:

```bash
# SGMET (7 semantic domains)
python -m src.clinical_cluster_experts.train_supervised_downstream \
  --token-dir /path/to/tokenized_nhanes \
  --cluster-csv /path/to/feature_clusters_biolord_v5_k7_leiden.csv \
  --num-clusters 7 --ignore-clusters ""

# Parameter-matched flat transformer (1 domain = all features in one encoder)
python -m src.clinical_cluster_experts.train_supervised_downstream \
  --token-dir /path/to/tokenized_nhanes \
  --cluster-csv /path/to/feature_clusters_biolord_v5_k7_leiden.csv \
  --num-clusters 1 --ignore-clusters ""
```

See [`scripts/sgmet_scripts/README.md`](../../scripts/sgmet_scripts/README.md)
for the preconfigured launcher scripts used for both variants.
