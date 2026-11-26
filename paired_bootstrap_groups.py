#!/usr/bin/env python3
"""Split test rows by a membership file and bootstrap paired model differences."""
import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys
import numpy as np

csv.field_size_limit(sys.maxsize)
METRICS = ('accuracy', 'precision', 'recall', 'f1')


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f, strict=True)
        if not r.fieldnames or len(r.fieldnames) != len(set(r.fieldnames)):
            raise ValueError(f'Invalid CSV headers: {path}')
        return r.fieldnames, list(r)


def metrics(counts):
    # counts columns: true label, model A prediction, model B prediction
    classes = np.array([(y, a, b) for y in (0, 1) for a in (0, 1) for b in (0, 1)])
    result = []
    for pred in (1, 2):
        tp = counts[:, (classes[:, 0] == 1) & (classes[:, pred] == 1)].sum(axis=1)
        fp = counts[:, (classes[:, 0] == 0) & (classes[:, pred] == 1)].sum(axis=1)
        tn = counts[:, (classes[:, 0] == 0) & (classes[:, pred] == 0)].sum(axis=1)
        fn = counts[:, (classes[:, 0] == 1) & (classes[:, pred] == 0)].sum(axis=1)
        def divide(num, den):
            return np.divide(num, den, out=np.full(len(num), np.nan), where=den != 0)
        result.append({'accuracy': divide(tp + tn, tp + fp + tn + fn), 'precision': divide(tp, tp + fp),
                       'recall': divide(tp, tp + fn), 'f1': divide(2 * tp, 2 * tp + fp + fn)})
    return result


def paired(y, pa, pb, resamples, seed):
    n = len(y)
    if not n:
        return {'count': 0, 'status': 'No rows in group; bootstrap not performed'}
    counts = np.bincount(y * 4 + pa * 2 + pb, minlength=8)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=resamples)
    estimate_a, estimate_b = metrics(counts.reshape(1, -1))
    draw_a, draw_b = metrics(draws)
    out = {}
    for m in METRICS:
        diff = draw_b[m] - draw_a[m]
        undefined = int((~np.isfinite(diff)).sum())
        ea, eb = estimate_a[m][0], estimate_b[m][0]
        out[m] = {'model_a': float(ea) if np.isfinite(ea) else None,
                  'model_b': float(eb) if np.isfinite(eb) else None,
                  'difference_b_minus_a': float(eb - ea) if np.isfinite(ea) and np.isfinite(eb) else None,
                  'confidence_interval': [float(x) for x in np.quantile(diff, [.025, .975])] if not undefined else None,
                  'undefined_resamples': undefined}
    return {'count': n, 'positive_count': int(y.sum()), 'negative_count': int(n - y.sum()), 'metrics': out}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test', required=True, type=Path)
    p.add_argument('--model', action='append', nargs=2, metavar=('NAME', 'PREDICTIONS'), required=True)
    p.add_argument('--indices', required=True, type=Path, help='JSON array of 1-based data row numbers in duplicated group')
    p.add_argument('--definition', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.output_dir.exists(): p.error('Output directory already exists; choose a new one')
    if len(a.model) < 2 or len({x[0] for x in a.model}) != len(a.model) or a.resamples < 1: p.error('Supply at least two uniquely named models and positive resample count')
    fields, test = read(a.test)
    if not {'body', 'label'} <= set(fields): p.error('Test CSV requires body and label')
    ids = json.loads(a.indices.read_text())
    if not isinstance(ids, list) or any(type(i) is not int or i < 1 or i > len(test) for i in ids) or len(ids) != len(set(ids)):
        p.error('Invalid 1-based row indices')
    selected = set(ids)
    def binary(s):
        s = s.strip()
        if s in ('1', '1.0'): return 1
        if s in ('0', '0.0'): return 0
        raise ValueError(f'Unexpected binary label: {s!r}')
    truth = np.array([binary(r['label']) for r in test], dtype=np.int8)
    predictions = {}
    for name, path in a.model:
        _, rows = read(path)
        if len(rows) != len(test): raise ValueError(f'{name}: record count differs from test')
        for i, (t, r) in enumerate(zip(test, rows), 1):
            if t['body'] != r['body'] or binary(t['label']) != binary(r['label']):
                raise ValueError(f'{name}: body or label mismatch at row {i}; files may not be aligned')
        predictions[name] = np.array([binary(r['predicted_label']) for r in rows], dtype=np.int8)
    a.output_dir.mkdir(parents=True)
    mask = np.array([i in selected for i in range(1, len(test) + 1)])
    result = {'definition': a.definition, 'method': 'paired unstratified row bootstrap via joint label/prediction multinomial counts',
              'resamples': a.resamples, 'confidence_level': .95, 'seed': a.seed,
              'test_csv': str(a.test), 'test_sha256': hashlib.sha256(a.test.read_bytes()).hexdigest(),
              'models': {name: {'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for name, path in a.model},
              'notes': ['Intervals are unadjusted for multiple comparisons.', 'Row resampling assumes independence, which may fail for duplicated or related texts.', 'A null interval indicates one or more bootstrap resamples had undefined metrics.'],
              'groups': {}}
    for group, group_mask in [('duplicated', mask), ('non-duplicated', ~mask)]:
        with (a.output_dir / f'{group}.csv').open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            w.writerows(row for row, use in zip(test, group_mask) if use)
        result['groups'][group] = {'count': int(group_mask.sum()), 'comparisons': {}}
        for ma, mb in combinations(predictions, 2):
            result['groups'][group]['comparisons'][f'{ma} vs {mb}'] = paired(truth[group_mask], predictions[ma][group_mask], predictions[mb][group_mask], a.resamples, a.seed)
    with (a.output_dir / 'paired-bootstrap.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(f'Saved {a.output_dir}; group sizes: duplicated={mask.sum()}, non-duplicated={(~mask).sum()}')


if __name__ == '__main__':
    main()
