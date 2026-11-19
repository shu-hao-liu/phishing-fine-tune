#!/usr/bin/env python3
"""Exact two-sided McNemar test for two classifiers on the same examples."""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUTPUT_COLUMNS = {'predicted_label', 'phishing_probability', 'model_response'}


def exact_mcnemar(only_a_correct, only_b_correct):
    """Exact binomial two-sided p-value, with a stable log10 for tiny values."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (only_a_correct, only_b_correct)):
        raise ValueError('Discordant counts must be nonnegative integers')
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0, 0.0
    k = min(only_a_correct, only_b_correct)
    # Sum P(X <= k) relative to its largest term to avoid underflow.
    relative_sum = term = 1.0
    for i in range(k, 0, -1):
        term *= i / (n - i + 1)
        relative_sum += term
    log_tail_term = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
    log_p = min(0.0, math.log(2) + log_tail_term + math.log(relative_sum))
    return math.exp(log_p), log_p / math.log(10)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare_csvs(path_a, path_b, *, label_column='label', prediction_a=None, prediction_b=None, match_columns=None):
    csv.field_size_limit(sys.maxsize)
    labels, predictions = set(), set()
    counts, matches = Counter(), Counter()
    with Path(path_a).expanduser().open(encoding='utf-8-sig', newline='') as fa, \
         Path(path_b).expanduser().open(encoding='utf-8-sig', newline='') as fb:
        a, b = csv.DictReader(fa, strict=True), csv.DictReader(fb, strict=True)
        fields_a, fields_b = a.fieldnames, b.fieldnames
        for fields in (fields_a, fields_b):
            if not fields or any(not x for x in fields) or len(fields) != len(set(fields)):
                raise ValueError('Each CSV needs unique, nonempty column names')
            if label_column not in fields:
                raise ValueError(f'Missing ground-truth column {label_column!r}')
        prediction_a = prediction_a or ('predicted_label' if 'predicted_label' in fields_a else 'model_response')
        prediction_b = prediction_b or ('predicted_label' if 'predicted_label' in fields_b else 'model_response')
        if prediction_a not in fields_a or prediction_b not in fields_b:
            raise ValueError('A prediction column is missing; use --prediction-a or --prediction-b')
        if label_column in (prediction_a, prediction_b):
            raise ValueError('Prediction and ground-truth columns must differ')
        forbidden = OUTPUT_COLUMNS | {label_column, prediction_a, prediction_b}
        if match_columns is None:
            key = next((name for name in ('body', 'id', 'record_id', 'sample_id', 'text')
                        if name in fields_a and name in fields_b and name not in forbidden), None)
            if key is None:
                raise ValueError('No shared body, text, or ID column; supply --match-column')
            match_columns = [name for name in fields_a if name in fields_b and name not in forbidden]
        if not match_columns or any(not key or key not in fields_a or key not in fields_b or key in forbidden for key in match_columns):
            raise ValueError('Match columns must exist in both files and cannot be labels or predictions')
        for index, (row_a, row_b) in enumerate(zip_longest(a, b), start=1):
            if row_a is None or row_b is None:
                raise ValueError('The two CSVs have different numbers of rows')
            if None in row_a or None in row_b or any(v is None for v in row_a.values()) or any(v is None for v in row_b.values()):
                raise ValueError(f'Malformed record at row {index}')
            key_a, key_b = tuple(row_a[k] for k in match_columns), tuple(row_b[k] for k in match_columns)
            if not any(value.strip() for value in key_a):
                raise ValueError(f'Empty matching key at row {index}')
            if key_a != key_b:
                raise ValueError(f'Rows are not aligned at record {index}; align files first or select the correct match columns')
            label_a, label_b = row_a[label_column].strip(), row_b[label_column].strip()
            pa, pb = row_a[prediction_a].strip(), row_b[prediction_b].strip()
            if not label_a or not label_b or not pa or not pb:
                raise ValueError(f'Missing ground truth or prediction at record {index}')
            if label_a != label_b:
                raise ValueError(f'Ground-truth labels differ at record {index}')
            labels.add(label_a)
            predictions.update((pa, pb))
            matches[key_a] += 1
            counts[(pa == label_a, pb == label_a)] += 1
    total = sum(counts.values())
    if not total:
        raise ValueError('The CSVs have no records')
    if len(labels) > 2 or len(labels | predictions) > 2 or (len(labels) == 2 and not predictions <= labels):
        raise ValueError('Expected binary labels and predictions matching those labels exactly')
    only_a, only_b = counts[True, False], counts[False, True]
    p_value, log10_p = exact_mcnemar(only_a, only_b)
    return {'method': 'exact two-sided McNemar test', 'count': total,
            'both_correct': counts[True, True], 'only_a_correct': only_a,
            'only_b_correct': only_b, 'both_incorrect': counts[False, False],
            'accuracy_a': (counts[True, True] + only_a) / total,
            'accuracy_b': (counts[True, True] + only_b) / total,
            'accuracy_difference_b_minus_a': (only_b - only_a) / total,
            'discordant_pairs': only_a + only_b, 'p_value': p_value,
            'log10_p_value': log10_p, 'match_columns': match_columns,
            'duplicate_match_keys': sum(count - 1 for count in matches.values()),
            'label_column': label_column, 'prediction_column_a': prediction_a,
            'prediction_column_b': prediction_b,
            'notes': ['The test compares overall error rates, not F1, precision, recall or direction-specific errors.',
                      'Rows must be independent; repeated or related examples can invalidate the p-value.',
                      'p_value may underflow to zero for extremely small values; log10_p_value remains informative.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_a', help='first model predictions CSV')
    parser.add_argument('model_b', help='second model predictions CSV for the same examples in the same order')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--prediction-a', help='prediction column in model A CSV')
    parser.add_argument('--prediction-b', help='prediction column in model B CSV')
    parser.add_argument('--match-column', action='append', help='matching example column; can be repeated (default: shared input columns)')
    parser.add_argument('--output', help='save results to a new JSON file')
    args = parser.parse_args()
    try:
        result = compare_csvs(args.model_a, args.model_b, label_column=args.label_column,
                              prediction_a=args.prediction_a, prediction_b=args.prediction_b,
                              match_columns=args.match_column)
        result.update({'model_a_csv': str(Path(args.model_a).expanduser().resolve()),
                       'model_b_csv': str(Path(args.model_b).expanduser().resolve()),
                       'model_a_sha256': sha256(Path(args.model_a).expanduser()),
                       'model_b_sha256': sha256(Path(args.model_b).expanduser())})
        if args.output:
            output = Path(args.output).expanduser()
            if output.resolve() in (Path(args.model_a).expanduser().resolve(), Path(args.model_b).expanduser().resolve()):
                raise ValueError('Output must differ from the input files')
            with output.open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2, allow_nan=False)
                f.write('\n')
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.output:
            print(f'Saved result to {output}')
    except (OSError, csv.Error, UnicodeError, ValueError, TypeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
