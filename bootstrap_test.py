#!/usr/bin/env python3
"""Estimate binary classifier metric uncertainty using row bootstrap samples."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

from analyze_results import calculate_metrics

ROOT = Path(__file__).resolve().parent
METRICS = ('accuracy', 'precision', 'recall', 'f1')


def load_counts(path, label_column, positive_label, prediction_column=None):
    """Count paired labels and predictions while preserving their relationship."""
    csv.field_size_limit(sys.maxsize)
    pairs = Counter()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames
        if not fields or any(not name for name in fields) or len(set(fields)) != len(fields):
            raise ValueError('CSV must have unique, nonempty headers')
        if label_column not in fields:
            raise ValueError(f'Ground-truth label column {label_column!r} is missing')
        if prediction_column is None:
            prediction_column = 'predicted_label' if 'predicted_label' in fields else 'model_response'
        if prediction_column not in fields:
            raise ValueError(f'Prediction column {prediction_column!r} is missing; supply --prediction-column')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed record near line {reader.line_num}')
            truth, predicted = row[label_column].strip(), row[prediction_column].strip()
            if not truth or not predicted:
                raise ValueError(f'Empty label or prediction near line {reader.line_num}')
            pairs[truth, predicted] += 1
    if not pairs:
        raise ValueError('No prediction records found')
    other = {value for pair in pairs for value in pair if value != positive_label}
    if len(other) > 1:
        raise ValueError('Expected binary labels; predictions must use the same two label values as the data')
    negative = next(iter(other), None)
    counts = (pairs[positive_label, positive_label], pairs[negative, positive_label],
              pairs[negative, negative], pairs[positive_label, negative])
    return counts, negative, prediction_column


def bootstrap_draws(counts, resamples, seed):
    n = sum(counts)
    rng = np.random.default_rng(seed)
    # Multinomial confusion counts exactly equal paired-row resampling for
    # statistics calculated solely from the binary confusion matrix.
    tp, fp, tn, fn = rng.multinomial(n, np.array(counts, dtype=np.float64) / n, size=resamples).T
    with np.errstate(divide='ignore', invalid='ignore'):
        return {'accuracy': (tp + tn) / n,
                'precision': np.divide(tp, tp + fp, out=np.full(resamples, np.nan), where=(tp + fp) != 0),
                'recall': np.divide(tp, tp + fn, out=np.full(resamples, np.nan), where=(tp + fn) != 0),
                'f1': np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(resamples, np.nan), where=(2 * tp + fp + fn) != 0)}


def intervals(draws, point, resamples, confidence):
    tail = (1 - confidence) / 2
    results = {}
    for name, values in draws.items():
        valid = values[np.isfinite(values)]
        undefined = resamples - len(valid)
        # Undefined samples make unconditional percentiles ill-defined. Never
        # silently discard them or give an artificial value.
        interval = None if undefined else [float(x) for x in np.quantile(valid, [tail, 1 - tail])]
        results[name] = {'estimate': point[name], 'confidence_interval': interval,
                         'undefined_resamples': undefined}
    return results


def bootstrap_metrics(counts, *, resamples=10000, confidence=0.95, seed=42):
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1, exclusive')
    if len(counts) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts) or not sum(counts):
        raise ValueError('Confusion counts must contain four nonnegative integers with a positive total')
    point = calculate_metrics(*counts)
    draws = bootstrap_draws(counts, resamples, seed)
    return {'method': 'unstratified paired-row percentile bootstrap', 'resamples': resamples,
            'confidence_level': confidence, 'seed': seed, 'point_metrics': point,
            'metrics': intervals(draws, point, resamples, confidence)}


def compare_independent(counts_a, counts_b, *, resamples=10000, confidence=0.95, seed=42):
    a = bootstrap_metrics(counts_a, resamples=resamples, confidence=confidence, seed=seed)
    b = bootstrap_metrics(counts_b, resamples=resamples, confidence=confidence, seed=seed + 1)
    draws_a = bootstrap_draws(counts_a, resamples, seed)
    draws_b = bootstrap_draws(counts_b, resamples, seed + 1)
    differences = {name: draws_b[name] - draws_a[name] for name in METRICS}
    point_diff = {name: (b['point_metrics'][name] - a['point_metrics'][name]
                         if a['point_metrics'][name] is not None and b['point_metrics'][name] is not None else None)
                  for name in METRICS}
    return b, {'direction': 'model_b_minus_model_a', 'method': 'independent unstratified percentile bootstrap',
               'metrics': intervals(differences, point_diff, resamples, confidence),
               'notes': ['Datasets are assumed independent; shared examples require paired resampling.',
                         'Differences combine model and dataset effects and do not isolate model performance.']}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='CSV containing ground-truth labels and model predictions')
    parser.add_argument('--compare', help='second model predictions CSV on an independent dataset')
    parser.add_argument('--compare-label-column', help='ground-truth column in the second CSV (default: --label-column)')
    parser.add_argument('--compare-prediction-column', help='prediction column in the second CSV (default: auto-detect)')
    parser.add_argument('--compare-positive-label', help='positive class in the second CSV (default: first positive label)')
    parser.add_argument('--label-column', help='ground-truth column (default: config.yaml)')
    parser.add_argument('--prediction-column', help='prediction column (default: predicted_label, then model_response)')
    parser.add_argument('--positive-label', help='value representing the positive class (default: config.yaml)')
    parser.add_argument('--resamples', type=int, default=10000, help='number of bootstrap samples (default: 10000)')
    parser.add_argument('--confidence', type=float, default=0.95, help='confidence level between 0 and 1 (default: 0.95)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--output', help='new JSON output file (default: bootstrap_results.json beside predictions)')
    args = parser.parse_args()
    try:
        with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        data = config.get('data', config.get('train', {}))
        label = args.label_column or data.get('label_column', 'label')
        positive = args.positive_label if args.positive_label is not None else str(data.get('positive_label', '1.0'))
        if not positive.strip():
            raise ValueError('positive label cannot be empty')
        counts, negative, prediction = load_counts(args.predictions, label, positive, args.prediction_column)
        result = bootstrap_metrics(counts, resamples=args.resamples, confidence=args.confidence, seed=args.seed)
        result.update({'input_csv': str(Path(args.predictions).expanduser().resolve()),
                       'input_sha256': sha256(args.predictions), 'label_column': label,
                       'prediction_column': prediction, 'positive_label': positive, 'negative_label': negative,
                       'notes': ['Rows are assumed independent; repeated or grouped examples may require grouped resampling.',
                                 'An interval is null when one or more bootstrap samples have an undefined metric.']})
        if args.compare:
            second_label = args.compare_label_column or label
            second_positive = args.compare_positive_label if args.compare_positive_label is not None else positive
            if not second_positive.strip():
                raise ValueError('second positive label cannot be empty')
            second_counts, second_negative, second_prediction = load_counts(args.compare, second_label, second_positive, args.compare_prediction_column)
            second, comparison = compare_independent(counts, second_counts, resamples=args.resamples, confidence=args.confidence, seed=args.seed)
            second.update({'input_csv': str(Path(args.compare).expanduser().resolve()),
                           'input_sha256': sha256(args.compare), 'label_column': second_label,
                           'prediction_column': second_prediction, 'positive_label': second_positive,
                           'negative_label': second_negative})
            result['model_b'] = second
            result['comparison'] = comparison
        output = Path(args.output).expanduser() if args.output else Path(args.predictions).with_name('bootstrap_results.json')
        input_paths = [Path(args.predictions).expanduser().resolve()]
        if args.compare:
            input_paths.append(Path(args.compare).expanduser().resolve())
        if output.resolve() in input_paths:
            raise ValueError('Output must differ from the inputs')
        with output.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
        print(json.dumps(result, indent=2, allow_nan=False))
        print(f'Saved bootstrap results to {output}')
    except (OSError, csv.Error, UnicodeError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
