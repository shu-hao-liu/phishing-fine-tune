#!/usr/bin/env python3
"""Estimate binary classifier metric uncertainty using row bootstrap samples."""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import zip_longest
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


def paired_counts(path_a, path_b, label_a, label_b, positive_a, positive_b,
                  prediction_a, prediction_b, pair_columns=None):
    """Verify aligned rows and count (truth, A prediction, B prediction)."""
    counts = Counter()
    with Path(path_a).open(encoding='utf-8-sig', newline='') as a, \
         Path(path_b).open(encoding='utf-8-sig', newline='') as b:
        ra, rb = csv.DictReader(a, strict=True), csv.DictReader(b, strict=True)
        fields_a, fields_b = ra.fieldnames or [], rb.fieldnames or []
        excluded = {label_a, label_b, prediction_a, prediction_b,
                    'predicted_label', 'phishing_probability', 'model_response'}
        columns = pair_columns or [key for key in fields_a if key in fields_b and key not in excluded]
        if not columns:
            raise ValueError('Paired comparison needs a shared example identifier or text column; use --pair-column')
        if any(key not in fields_a or key not in fields_b or key in excluded for key in columns):
            raise ValueError('Paired key columns must exist in both files and identify examples, not labels or predictions')
        for index, (row_a, row_b) in enumerate(zip_longest(ra, rb), start=1):
            if row_a is None or row_b is None:
                raise ValueError('Paired input files must have the same number of rows')
            if any(row_a[k] != row_b[k] for k in columns):
                raise ValueError(f'Paired input rows differ at record {index}; align the files first')
            truth_a = row_a[label_a].strip() == positive_a
            truth_b = row_b[label_b].strip() == positive_b
            if truth_a != truth_b:
                raise ValueError(f'Paired ground truths differ at record {index}')
            counts[truth_a, row_a[prediction_a].strip() == positive_a,
                   row_b[prediction_b].strip() == positive_b] += 1
    return counts, columns


def compare_paired(joint, *, resamples=10000, confidence=0.95, seed=42):
    categories = [(truth, pred_a, pred_b) for truth in (False, True)
                  for pred_a in (False, True) for pred_b in (False, True)]
    counts = [joint.get(key, 0) for key in categories]
    n = sum(counts)
    if not n:
        raise ValueError('No paired records found')
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(n, np.array(counts, dtype=np.float64) / n, size=resamples)
    def confusion(prediction_index, matrix):
        return [sum((matrix[..., i] for i, category in enumerate(categories)
                     if category[0] == truth and category[prediction_index] == predicted),
                    np.zeros(matrix.shape[:-1], dtype=np.int64))
                for truth, predicted in ((True, True), (False, True), (False, False), (True, False))]
    def metrics(values):
        tp, fp, tn, fn = values
        with np.errstate(divide='ignore', invalid='ignore'):
            def divide(top, bottom):
                return np.divide(top, bottom, out=np.full(resamples, np.nan), where=bottom != 0)
            return {'accuracy': (tp + tn) / n, 'precision': divide(tp, tp + fp),
                    'recall': divide(tp, tp + fn), 'f1': divide(2 * tp, 2 * tp + fp + fn)}
    draws_a, draws_b = metrics(confusion(1, sampled)), metrics(confusion(2, sampled))
    orig = np.asarray(counts, dtype=np.int64)
    point_a = calculate_metrics(*(int(x) for x in confusion(1, orig)))
    point_b = calculate_metrics(*(int(x) for x in confusion(2, orig)))
    point_diff = {key: (point_b[key] - point_a[key]
                        if point_a[key] is not None and point_b[key] is not None else None) for key in METRICS}
    difference = {key: draws_b[key] - draws_a[key] for key in METRICS}
    comparison = {'direction': 'model_b_minus_model_a', 'method': 'paired unstratified percentile bootstrap',
                  'metrics': intervals(difference, point_diff, resamples, confidence),
                  'notes': ['Each sample resamples the same aligned examples for both models.',
                            'Rows are assumed independent; repeated or grouped examples may require grouped resampling.']}
    return point_a, point_b, intervals(draws_a, point_a, resamples, confidence), intervals(draws_b, point_b, resamples, confidence), comparison


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='CSV containing ground-truth labels and model predictions')
    parser.add_argument('--compare', help='second model predictions CSV')
    parser.add_argument('--paired', action='store_true', help='compare models on aligned rows from the same dataset')
    parser.add_argument('--pair-column', action='append', help='shared column identifying each paired example (repeatable; default: all shared input columns)')
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
        if args.paired and not args.compare:
            raise ValueError('--paired requires --compare')
        if args.pair_column and not args.paired:
            raise ValueError('--pair-column requires --paired')
        if args.compare:
            second_label = args.compare_label_column or label
            second_positive = args.compare_positive_label if args.compare_positive_label is not None else positive
            if not second_positive.strip():
                raise ValueError('second positive label cannot be empty')
            second_counts, second_negative, second_prediction = load_counts(args.compare, second_label, second_positive, args.compare_prediction_column)
            if args.paired:
                joint, pair_columns = paired_counts(args.predictions, args.compare, label, second_label,
                                                    positive, second_positive, prediction, second_prediction,
                                                    args.pair_column)
                point_a, point_b, metrics_a, metrics_b, comparison = compare_paired(joint, resamples=args.resamples, confidence=args.confidence, seed=args.seed)
                if point_a != result['point_metrics'] or point_b != calculate_metrics(*second_counts):
                    raise ValueError('Paired records do not match the input counts')
                result['metrics'] = metrics_a
                second = {'method': 'paired-row percentile bootstrap', 'resamples': args.resamples,
                          'confidence_level': args.confidence, 'seed': args.seed,
                          'point_metrics': point_b, 'metrics': metrics_b}
                comparison['pair_columns'] = pair_columns
            else:
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
