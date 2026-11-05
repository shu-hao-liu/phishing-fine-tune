#!/usr/bin/env python3
"""Format a CSV for the configured phishing training and testing scripts."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parent
OPTIONAL_COLUMNS = ('subject', 'urls', 'language', 'source')


def format_csv(source, output, *, mode, text_source, label_source, text_target,
               label_target, positive_label, negative_label, positive_source=None,
               negative_source=None, unlabeled=False):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if source == output:
        raise ValueError('Input and output must be different files')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if mode != 'test' and unlabeled:
        raise ValueError('Only test CSVs can be unlabeled')
    if (positive_source is None) != (negative_source is None):
        raise ValueError('Provide both --positive and --negative when mapping labels')
    if positive_label == negative_label or (positive_source is not None and positive_source == negative_source):
        raise ValueError('Positive and negative labels must differ')
    if text_target == label_target or (not unlabeled and text_source == label_source):
        raise ValueError('Text and label columns must be different')
    if text_target in OPTIONAL_COLUMNS or label_target in OPTIONAL_COLUMNS:
        raise ValueError('Configured text and label columns must not conflict with optional output columns')
    labels = {positive_label: positive_label, negative_label: negative_label}
    if positive_source is not None:
        labels = {positive_source: positive_label, negative_source: negative_label}
    columns = ['subject', text_target, 'urls'] + ([] if unlabeled else [label_target]) + ['language', 'source']
    csv.field_size_limit(sys.maxsize)
    temp = None
    counts = Counter()
    try:
        with source.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, strict=True)
            if not reader.fieldnames or any(not name for name in reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('Input must have a header with unique, nonempty column names')
            required = [text_source] + ([] if unlabeled else [label_source])
            missing = [name for name in required if name not in reader.fieldnames]
            if missing:
                raise ValueError(f'Missing input columns: {missing}')
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=output.parent, prefix='.format-', suffix='.csv', delete=False) as f_out:
                temp = Path(f_out.name)
                writer = csv.DictWriter(f_out, fieldnames=columns, lineterminator='\n')
                writer.writeheader()
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f'Malformed row ending at line {reader.line_num}')
                    text = row[text_source]
                    if not text.strip():
                        raise ValueError(f'Empty text at line {reader.line_num}')
                    formatted = {name: row.get(name, '') for name in OPTIONAL_COLUMNS}
                    formatted[text_target] = text
                    if not unlabeled:
                        value = row[label_source].strip()
                        if value not in labels:
                            raise ValueError(f'Unexpected or empty label {value!r} at line {reader.line_num}; specify --positive and --negative for other labels')
                        formatted[label_target] = labels[value]
                        counts[labels[value]] += 1
                    else:
                        counts['unlabeled'] += 1
                    writer.writerow(formatted)
        if not counts:
            raise ValueError('Input CSV has no records')
        if mode == 'train' and set(counts) != {positive_label, negative_label}:
            raise ValueError('Training CSV must contain both positive and negative labels')
        # Link atomically, so an existing output is never replaced.
        os.link(temp, output)
        return counts
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='source CSV')
    parser.add_argument('output', help='new output CSV; existing files are never overwritten')
    parser.add_argument('--mode', choices=('train', 'validate', 'test'), required=True)
    parser.add_argument('--text-column', help='source text column (default: configured text column)')
    parser.add_argument('--label-column', help='source label column (default: configured label column)')
    parser.add_argument('--positive', help='source label meaning phishing')
    parser.add_argument('--negative', help='source label meaning legitimate')
    parser.add_argument('--negative-output', default='0.0', help='output label for legitimate (default: 0.0)')
    parser.add_argument('--unlabeled', action='store_true', help='format a test CSV without labels or metrics')
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    args = parser.parse_args()
    try:
        with open(args.config, encoding='utf-8') as f:
            settings = yaml.safe_load(f)['train']
        text_target = settings.get('text_column', 'body')
        label_target = settings.get('label_column', 'label')
        positive = str(settings.get('positive_label', '1.0'))
        counts = format_csv(args.input, args.output, mode=args.mode,
                            text_source=args.text_column or text_target,
                            label_source=args.label_column or label_target,
                            text_target=text_target, label_target=label_target,
                            positive_label=positive, negative_label=args.negative_output,
                            positive_source=args.positive, negative_source=args.negative,
                            unlabeled=args.unlabeled)
        print(f'Saved {sum(counts.values())} rows to {args.output}. Label counts: {dict(counts)}')
        if not args.unlabeled and len(counts) < 2:
            print('Note: this dataset has one class; the current results analyzer requires both classes to report metrics.')
    except (OSError, ValueError, KeyError, TypeError, csv.Error, yaml.YAMLError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
