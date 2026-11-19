#!/usr/bin/env python3
"""Calculate ROC-AUC from a labeled predictions CSV on a local CPU."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent


def read_scores(path, label_column='label', score_column='phishing_probability', positive_label='1.0'):
    """Read numeric scores and binary labels; larger scores mean more positive."""
    csv.field_size_limit(sys.maxsize)
    observations = []
    labels = Counter()
    with Path(path).expanduser().open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames
        if not fields or any(not x for x in fields) or len(set(fields)) != len(fields):
            raise ValueError('CSV must have unique, nonempty headers')
        for column in (label_column, score_column):
            if column not in fields:
                raise ValueError(f'Required column {column!r} is missing')
        if label_column == score_column:
            raise ValueError('Label and score columns must differ')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed record near line {reader.line_num}')
            label, raw_score = row[label_column].strip(), row[score_column].strip()
            if not label or not raw_score:
                raise ValueError(f'Empty label or score near line {reader.line_num}')
            try:
                score = float(raw_score)
            except ValueError as exc:
                raise ValueError(f'Invalid numeric score near line {reader.line_num}') from exc
            if not math.isfinite(score):
                raise ValueError(f'Non-finite score near line {reader.line_num}')
            observations.append((score, label == positive_label))
            labels[label] += 1
    if not observations:
        raise ValueError('No records found')
    if len(labels) != 2 or positive_label not in labels:
        raise ValueError(f'The CSV needs two label classes including positive label {positive_label!r}; found {sorted(labels)}')
    negative_label = next(label for label in labels if label != positive_label)
    return observations, labels, negative_label


def calculate_roc_auc(observations):
    """Calculate ROC vertices and trapezoidal AUC; equal-score ties stay together."""
    ranked = sorted(observations, key=lambda x: x[0], reverse=True)
    positives = sum(positive for _, positive in ranked)
    negatives = len(ranked) - positives
    if positives == 0 or negatives == 0:
        raise ValueError('ROC-AUC requires at least one positive and one negative example')
    curve = [{'threshold': None, 'false_positive_rate': 0.0, 'true_positive_rate': 0.0}]
    tp = fp = i = 0
    while i < len(ranked):
        score = ranked[i][0]
        while i < len(ranked) and ranked[i][0] == score:
            if ranked[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        curve.append({'threshold': score, 'false_positive_rate': fp / negatives, 'true_positive_rate': tp / positives})
    auc = math.fsum((b['false_positive_rate'] - a['false_positive_rate']) *
                    (a['true_positive_rate'] + b['true_positive_rate']) / 2
                    for a, b in zip(curve, curve[1:]))
    return auc, curve


def sha256(path):
    value = hashlib.sha256()
    with Path(path).expanduser().open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_plot(path, curve, auc):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError('Plotting requires matplotlib: python3 -m pip install matplotlib') from exc
    destination = Path(path).expanduser()
    suffix = destination.suffix.lower().lstrip('.')
    if suffix not in ('png', 'pdf', 'svg'):
        raise ValueError('Plot filename must end in .png, .pdf, or .svg')
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.plot([p['false_positive_rate'] for p in curve], [p['true_positive_rate'] for p in curve],
            color='#176a6a', linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#8c98a3', label='Random ranking')
    ax.set(xlabel='False positive rate', ylabel='True positive rate (recall)', title='ROC curve',
           xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(loc='lower right')
    fig.tight_layout()
    try:
        with destination.open('xb') as f:
            fig.savefig(f, format=suffix, dpi=200)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='labeled predictions CSV; higher scores indicate the positive class')
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--label-column', help='label column (default: config.yaml)')
    parser.add_argument('--score-column', default='phishing_probability', help='numeric positive-class score (default: phishing_probability)')
    parser.add_argument('--positive-label', help='positive label (default: config.yaml)')
    parser.add_argument('--output', help='save summary to a new JSON file')
    parser.add_argument('--curve-csv', help='save ROC vertices to a new CSV file')
    parser.add_argument('--plot', help='save an ROC curve to a new PNG, PDF, or SVG (requires matplotlib)')
    args = parser.parse_args()
    try:
        with open(args.config, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        data = config.get('data', config.get('train', {}))
        label_column = args.label_column or data.get('label_column', 'label')
        positive = args.positive_label if args.positive_label is not None else str(data.get('positive_label', '1.0'))
        if not positive.strip():
            raise ValueError('Positive label cannot be empty')
        outputs = [Path(value).expanduser().resolve() for value in (args.output, args.curve_csv, args.plot) if value]
        input_path = Path(args.predictions).expanduser().resolve()
        if len(outputs) != len(set(outputs)) or input_path in outputs:
            raise ValueError('Input and output paths must be distinct')
        for output in outputs:
            if output.exists():
                raise FileExistsError(f'Output already exists: {output}')
        observations, labels, negative = read_scores(input_path, label_column, args.score_column, positive)
        auc, curve = calculate_roc_auc(observations)
        summary = {'roc_auc': auc, 'count': len(observations), 'positive_count': labels[positive],
                   'negative_count': labels[negative], 'positive_label': positive, 'negative_label': negative,
                   'label_column': label_column, 'score_column': args.score_column,
                   'score_direction': 'higher_is_positive', 'input_csv': str(input_path),
                   'input_sha256': sha256(input_path)}
        if args.output:
            with Path(args.output).expanduser().open('x', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, allow_nan=False)
                f.write('\n')
        if args.curve_csv:
            with Path(args.curve_csv).expanduser().open('x', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=('threshold', 'false_positive_rate', 'true_positive_rate'), lineterminator='\n')
                writer.writeheader()
                writer.writerows(curve)
        if args.plot:
            save_plot(args.plot, curve, auc)
        print(json.dumps(summary, indent=2, allow_nan=False))
        for path in outputs:
            print(f'Saved: {path}')
    except (OSError, csv.Error, UnicodeError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
