#!/usr/bin/env python3
"""Paired clean-v5 TabPFN-3 contexts. API inference only, no weight training.

Fixed 10,000-patient, single-member budget in both arms. One corrupted view per
patient keeps context length, patient identities and labels matched. Complete
all five validation folds before held-out intact and domain-removal evaluation.
Per-task atomic results allow safe continuation without repeating finished work.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import xgboost_missingness_augmented_v5 as data


if not os.environ.get('TABPFN_TOKEN'):
    for raw in (data.PROJECT / '.env').read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            if key.strip() == 'TABPFN_TOKEN':
                os.environ['TABPFN_TOKEN'] = value.strip().strip('"').strip("'")
                break
assert os.environ.get('TABPFN_TOKEN'), 'TABPFN_TOKEN is not configured'
from importlib.metadata import version
from tabpfn_client import TabPFNClassifier
from tabpfn_client.constants import ModelVersion

OUT = data.PROJECT / 'outputs/tabpfn3_v5_clean_vs_augmented_context10000_e1_seed42'
ARMS = ('clean', 'augmented_context')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def context(fold, target, target_index):
    y = fold['outcomes']['train'][target]
    indices = np.flatnonzero(np.isfinite(y))
    if len(indices) > 10000:
        indices, _ = train_test_split(indices, train_size=10000,
                                     stratify=y[indices], random_state=42)
    values = fold['matrices']['train'][indices]
    keep = np.isfinite(values).any(axis=0)
    values = values[:, keep]
    augmented, _, audit = data.augment_training(
        values, fold['cluster_ids'][keep], replicas=1, virtual_batch_size=128,
        feature_dropout_max=.10, group_dropout_max=6/7,
        group_dropout_mode='uniform_count', seed=42000 + target_index)
    assert augmented.shape == values.shape
    retained = np.isfinite(augmented)
    assert np.array_equal(augmented[retained], values[retained])
    assert np.isnan(augmented)[np.isnan(values)].all()
    assert len(indices) == len(np.unique(indices))
    assert np.isfinite(augmented).any(axis=0).all(), 'Corruption lost a context feature'
    return indices, keep, values, augmented, y[indices].astype(int), audit


def preflight():
    files = [data.MANIFEST, data.CLUSTER_MAP, Path(data.__file__), Path(__file__)]
    checks = []
    for number in range(5):
        fold = data.load_fold(number)  # train + validation only
        folder = data.CV / f'fold{number}'
        indices = np.load(folder / 'restrat_indices.npz')
        for split in ('train', 'val', 'test'):
            assert len(np.unique(indices[split])) == len(indices[split])
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            assert not np.intersect1d(indices[a], indices[b]).size
        for split in ('train', 'val'):
            assert len(fold['matrices'][split]) == len(indices[split])
            files += [folder / f'{split}_{kind}.pt' for kind in ('tokens', 'targets')]
        files += [folder / 'restrat_indices.npz', folder / 'tokenizer_metadata.pt']
        assert len(fold['target_names']) == 11
        for ti, target in enumerate(fold['target_names']):
            local, keep, clean, augmented, y, audit = context(fold, target, number*100 + ti)
            assert set(y) == {0, 1}
            assert len(local) <= 10000 and 0 < clean.shape[1] <= 149
        checks.append({'fold': number, 'train_rows': len(indices['train']),
                       'val_rows': len(indices['val']), 'test_rows_from_indices': len(indices['test'])})
    return {'model': 'TabPFN-3', 'tabpfn_client_version': version('tabpfn-client'),
            'backend': 'Prior Labs API', 'seed': 42,
            'n_estimators': 1, 'max_context_patients': 10000,
            'physical_features': 149, 'groups': 7, 'downstream_tasks': 11,
            'folds': checks, 'arms': list(ARMS),
            'augmentation': 'one view per patient; batch128; p_feature~U(0,.10); n_groups~Uniform{0..6}',
            'missing_values': 'native NaN, no median imputation',
            'context_selection': 'same task-stratified train-only patients in both arms; features all-missing among those task-eligible patients removed identically',
            'validation': 'intact only, no HPO or performance-based decision to omit folds',
            'test': 'intact and each of seven domains removed, after all validation completes',
            'uncertainty': 'five folds at one seed, not 25 checkpoints or five independent cohorts',
            'historical_comparison': 'new pinned API version, not a reproduction of old unpinned API result',
            'files_sha256': {str(p): data.sha256(p) for p in files}}


def measure_many(model, scenarios, labels):
    eligible = np.isfinite(labels)
    y = labels[eligible].astype(int)
    assert set(y) == {0, 1}
    started = time.monotonic()
    probabilities = model.predict_proba(np.vstack([values[eligible] for _, values, _ in scenarios]))
    assert probabilities.shape == (len(y) * len(scenarios), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    positive = int(np.flatnonzero(model.classes_ == 1)[0])
    elapsed = time.monotonic() - started
    results = []
    for offset, (name, _, mask_hash) in enumerate(scenarios):
        p = probabilities[offset * len(y):(offset + 1) * len(y), positive]
        results.append((name, mask_hash, {'auroc': float(roc_auc_score(y, p)),
            'auprc': float(average_precision_score(y, p)), 'n': len(y),
            'positive': int(y.sum()), 'prevalence': float(y.mean()),
            'prediction_seconds_combined_request': elapsed}, p, np.flatnonzero(eligible)))
    return results


def run_fold(output, number, split):
    fold = data.load_fold(number)
    folder = data.CV / f'fold{number}'
    meta = torch.load(folder / 'tokenizer_metadata.pt', weights_only=False, map_location='cpu')
    if split == 'test':
        assert (output / 'VALIDATION_COMPLETE.json').exists()
        external = data.load_test_fold(number)
        values, outcomes = external['values'], external['outcomes']
    else:
        values, outcomes = fold['matrices']['val'], fold['outcomes']['val']
    scenarios = [('intact', None)]
    if split == 'test':
        scenarios += [(f'without_G{g}', g) for g in range(7)]
    original_indices = np.load(folder / 'restrat_indices.npz')['train']
    for ti, target in enumerate(fold['target_names']):
        local, keep, clean, augmented, y, audit = context(fold, target, number*100 + ti)
        categories = np.flatnonzero(meta['feature_type_ids'].numpy()[keep] != 0).tolist()
        for arm in ARMS:
            destination = output / f'fold{number}' / arm / target
            destination.mkdir(parents=True, exist_ok=True)
            expected = [destination / f'{split}_{name}.json' for name, _ in scenarios]
            if all(p.exists() for p in expected):
                continue
            save(output / 'STATUS.json', {'phase': split, 'fold': number, 'arm': arm,
                                         'target': target, 'updated_unix': time.time()})
            print(f'START {split} fold{number} {arm} {target}', flush=True)
            fitted = destination / 'context.joblib'
            started = time.monotonic()
            if fitted.exists():
                model = joblib.load(fitted)
            else:
                assert split == 'val', 'Test requires previously saved validation fit'
                model = TabPFNClassifier.create_default_for_version(
                    ModelVersion.V3, n_estimators=1, random_state=42,
                    categorical_features_indices=categories)
                model.fit(clean if arm == 'clean' else augmented, y)
                temporary = destination / 'temporary.joblib'
                joblib.dump(model, temporary)
                temporary.replace(fitted)
                np.savez_compressed(destination / 'context_indices.npz',
                                    local=local, source=original_indices[local])
                save(destination / 'context.json', {
                    'context_patients': len(local), 'retained_features': int(keep.sum()),
                    'retained_feature_names': np.asarray(fold['feature_names'])[keep].tolist(),
                    'source_indices_sha256': digest_array(original_indices[local]),
                    'labels_sha256': digest_array(y), 'model_sha256': data.sha256(fitted),
                    'context_sha256': digest_array(clean if arm == 'clean' else augmented),
                    'augmentation': audit if arm != 'clean' else None})
            fit_seconds = time.monotonic() - started
            prepared = []
            for name, group in scenarios:
                current, mask_hash = data.validation_view(values[:, keep], fold['cluster_ids'][keep],
                    'feature_fraction' if group is None else 'leave_one_group_out',
                    0 if group is None else group, 1900001 + number*10000)
                prepared.append((name, current, mask_hash))
            for name, mask_hash, result, probability, eligible in measure_many(model, prepared, outcomes[target]):
                result_path = destination / f'{split}_{name}.json'
                np.savez_compressed(destination / f'{split}_{name}_predictions.npz',
                                    probability=probability, eligible_rows=eligible)
                result.update({'fold': number, 'seed': 42, 'arm': arm, 'target': target,
                               'split': split, 'scenario': name, 'mask_sha256': mask_hash,
                               'fit_or_load_seconds': fit_seconds})
                save(result_path, result)
                print(f'DONE {split} fold{number} {arm} {target} {name}: '
                      f'AUROC={result["auroc"]:.6f} AUPRC={result["auprc"]:.6f} '
                      f'seconds={result["prediction_seconds_combined_request"]:.1f}', flush=True)
            del model
    rows = [json.loads(p.read_text()) for p in output.glob(f'fold{number}/*/*/{split}_*.json')]
    assert len(rows) == 22 * len(scenarios)
    save(output / f'fold{number}' / f'{split.upper()}_DONE.json', rows)


def summary(output, split):
    rows = [r for fold in range(5) for r in json.loads(
        (output / f'fold{fold}' / f'{split.upper()}_DONE.json').read_text())]
    result = []
    for arm in ARMS:
        for scenario in sorted({r['scenario'] for r in rows}):
            folds = []
            for fold in range(5):
                selected = [r for r in rows if r['arm'] == arm and r['scenario'] == scenario and r['fold'] == fold]
                assert len(selected) == 11
                folds.append({m: float(np.mean([r[m] for r in selected])) for m in ('auroc', 'auprc')})
            result.append({'arm': arm, 'scenario': scenario, 'fold_values': folds,
                           **{f'{m}_{stat}': float(getattr(np, stat)([r[m] for r in folds],
                                  **({'ddof': 1} if stat == 'std' else {})))
                              for m in ('auroc', 'auprc') for stat in ('mean', 'std')}})
    save(output / f'{split}_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = preflight()
    if args.preflight:
        print(json.dumps(config, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'CONFIG.json').exists():
        assert json.loads((output / 'CONFIG.json').read_text()) == config, 'Configuration/source/data changed'
    else:
        save(output / 'CONFIG.json', config)
    started = time.time()
    for fold in range(1 if args.pilot_only else 5):
        run_fold(output, fold, 'val')
        if fold == 0:
            save(output / 'PILOT_COMPLETE.json', {'finite': True, 'paired_tasks': 11,
                'elapsed_seconds_this_invocation': time.time()-started,
                'note': 'Pipeline check, not proof of augmentation benefit. No test data loaded.'})
    if args.pilot_only:
        return
    validation = summary(output, 'val')
    save(output / 'VALIDATION_COMPLETE.json', validation)
    for fold in range(5):
        run_fold(output, fold, 'test')
    test = summary(output, 'test')
    save(output / 'DONE.json', {'validation': validation, 'test': test,
                              'elapsed_seconds_this_invocation': time.time()-started})


if __name__ == '__main__':
    main()
