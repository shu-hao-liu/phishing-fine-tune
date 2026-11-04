#!/usr/bin/env python3
"""Create equal, stratified validation and test sets from an inference CSV."""
import argparse
from collections import Counter, defaultdict
import csv
import os
from pathlib import Path
import random
import sys
import tempfile

from split_csv import RecordingLines, signature

ROOT = Path(__file__).resolve().parent


def split(source_path, validation_path, test_path, seed=42, label_column='label'):
    source = Path(source_path).expanduser().resolve()
    validation, test = Path(validation_path).expanduser().resolve(), Path(test_path).expanduser().resolve()
    if len({source, validation, test}) != 3:
        raise ValueError('Source, validation, and test paths must differ')
    if validation.exists() or test.exists():
        raise FileExistsError('Validation or test output already exists; neither will be overwritten')
    if not validation.parent.is_dir() or not test.parent.is_dir():
        raise FileNotFoundError('Output directories must already exist')
    csv.field_size_limit(sys.maxsize)
    before = signature(source)
    indices = defaultdict(list)
    temps = []
    installed = []
    try:
        with source.open(encoding='utf-8', newline='') as f:
            reader = csv.reader(f, strict=True)
            header = next(reader)
            if len(set(header)) != len(header) or label_column not in header:
                raise ValueError('Missing label column or duplicate header names')
            label_idx = header.index(label_column)
            for index, row in enumerate(reader):
                if len(row) != len(header) or not row[label_idx].strip():
                    raise ValueError(f'Malformed or unlabeled record near line {reader.line_num}')
                indices[row[label_idx].strip()].append(index)
            if len(indices) != 2:
                raise ValueError(f'Expected exactly two label values, found {sorted(indices)}')
            rng = random.Random(seed)
            chosen = set()
            for label in sorted(indices):
                group = indices[label]
                chosen.update(rng.sample(group, len(group) // 2))
            if not chosen or sum(map(len, indices.values())) == len(chosen):
                raise ValueError('Both sets must contain data')
            f.seek(0)
            lines = RecordingLines(f)
            reader = csv.reader(lines, strict=True)
            next(reader)
            header_text = lines.take()
            counts = {'validation': Counter(), 'test': Counter()}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=validation.parent, prefix='.split-validation-', suffix='.csv', delete=False) as va:
                temps.append(Path(va.name))
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=test.parent, prefix='.split-test-', suffix='.csv', delete=False) as te:
                    temps.append(Path(te.name))
                    va.write(header_text); te.write(header_text)
                    for index, row in enumerate(reader):
                        raw = lines.take()
                        key = 'validation' if index in chosen else 'test'
                        (va if key == 'validation' else te).write(raw)
                        counts[key][row[label_idx].strip()] += 1
        if signature(source) != before:
            raise RuntimeError('Source changed during split; outputs were not installed')
        os.link(temps[0], validation); installed.append(validation)
        os.link(temps[1], test); installed.append(test)
        return counts
    except BaseException:
        for path in installed:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temps:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(ROOT / 'data' / 'inference.csv'))
    parser.add_argument('--validation', default=str(ROOT / 'data' / 'validation.csv'))
    parser.add_argument('--test', default=str(ROOT / 'data' / 'test.csv'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label-column', default='label')
    args = parser.parse_args()
    try:
        counts = split(args.input, args.validation, args.test, args.seed, args.label_column)
        for name, items in counts.items():
            print(f'{name}: {sum(items.values())} records; labels: {dict(items)}')
    except (OSError, UnicodeError, csv.Error, ValueError, RuntimeError, StopIteration) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
