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


def bootstrap_metrics(counts, *, resamples=10000, confidence=0.95, seed=42):
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError('resamples must be a positive integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be between 0 and 1, exclusive')
    if len(counts) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts) or not sum(counts):
        raise ValueError('Confusion counts must contain four nonnegative integers with a positive total')
    n = sum(counts)
    point = calculate_metrics(*counts)
    # Sampling the four confusion categories is exactly equivalent to sampling
    # paired (ground truth, prediction) rows independently with replacement.
    rng = np.random.default_rng(seed)
    tp, fp, tn, fn = rng.multinomial(n, np.array(counts, dtype=np.float64) / n, size=resamples).T
    with np.errstate(divide='ignore', invalid='ignore'):
        draws = {'accuracy': (tp + tn) / n,
                 'precision': np.divide(tp, tp + fp, out=np.full(resamples, np.nan), where=(tp + fp) != 0),
                 'recall': np.divide(tp, tp + fn, out=np.full(resamples, np.nan), where=(tp + fn) != 0),
                 'f1': np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(resamples, np.nan), where=(2 * tp + fp + fn) != 0)}
    tail = (1 - confidence) / 2
    results = {}
    for name, values in draws.items():
        valid = values[np.isfinite(values)]
        undefined = resamples - len(valid)
        # Undefined replicates make unconditional percentiles ill-defined. Do
        # not silently drop them or assign an artificial value.
        interval = None if undefined else [float(x) for x in np.quantile(valid, [tail, 1 - tail])]
        results[name] = {'estimate': point[name], 'confidence_interval': interval,
                         'undefined_resamples': undefined}
    return {'method': 'unstratified paired-row percentile bootstrap', 'resamples': resamples,
            'confidence_level': confidence, 'seed': seed, 'point_metrics': point, 'metrics': results}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='CSV containing ground-truth labels and model predictions')
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
        output = Path(args.output).expanduser() if args.output else Path(args.predictions).with_name('bootstrap_results.json')
        if output.resolve() == Path(args.predictions).expanduser().resolve():
            raise ValueError('Output must differ from the input')
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
