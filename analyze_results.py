#!/usr/bin/env python3
"""Calculate binary classification metrics from a prediction CSV."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent


def calculate_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {'count': total, 'accuracy': ratio(tp + tn, total), 'precision': precision,
            'recall': recall, 'f1': f1,
            'true_positives': tp, 'false_positives': fp,
            'true_negatives': tn, 'false_negatives': fn}


def analyze_predictions(path, label_column, positive_label, prediction_column=None):
    csv.field_size_limit(sys.maxsize)
    counts = Counter()
    truths = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, strict=True)
        if not reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(f'Ground-truth label column {label_column!r} is missing')
        if prediction_column is None:
            prediction_column = 'predicted_label' if 'predicted_label' in reader.fieldnames else 'model_response'
        if prediction_column not in reader.fieldnames:
            raise ValueError(f'Prediction column {prediction_column!r} is missing')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed record near line {reader.line_num}')
            truth, predicted = row[label_column].strip(), row[prediction_column].strip()
            if not truth or not predicted:
                raise ValueError(f'Empty label or prediction near line {reader.line_num}')
            truths.add(truth)
            counts[(truth, predicted)] += 1
    if not counts:
        raise ValueError('No prediction records found')
    if positive_label not in truths or len(truths) != 2:
        raise ValueError(f'Ground truth must have exactly two labels including {positive_label!r}; found {sorted(truths)}')
    negative_label = next(label for label in truths if label != positive_label)
    unexpected = {pred for _, pred in counts} - truths
    if unexpected:
        raise ValueError('Predictions must match the ground-truth labels exactly; unmatched values found. Generative responses are not automatically classified.')
    result = calculate_metrics(counts[positive_label, positive_label], counts[negative_label, positive_label],
                               counts[negative_label, negative_label], counts[positive_label, negative_label])
    result.update({'positive_label': positive_label, 'negative_label': negative_label,
                   'label_column': label_column, 'prediction_column': prediction_column})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='predictions.csv produced by test.py')
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--prediction-column', default=None)
    parser.add_argument('--output', default=None, help='JSON output path (default: metrics.json next to predictions)')
    args = parser.parse_args()
    try:
        with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        data = config.get('data', config.get('train', {}))
        metrics = analyze_predictions(args.predictions, data.get('label_column', 'label'), str(data.get('positive_label', '1.0')), args.prediction_column)
        output = Path(args.output) if args.output else Path(args.predictions).with_name('metrics.json')
        with output.open('x', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            f.write('\n')
        print(json.dumps(metrics, indent=2))
        print(f'Saved metrics to {output}')
    except (OSError, csv.Error, UnicodeError, ValueError, KeyError, TypeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
