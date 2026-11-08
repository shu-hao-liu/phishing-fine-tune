#!/usr/bin/env python3
"""Split the formatted FCN CSV into reproducible stratified 80/10/10 datasets."""
import csv
from collections import Counter, defaultdict
import os
from pathlib import Path
import random
import sys
import tempfile

from split_csv import signature

ROOT = Path(__file__).resolve().parent


def split(source, destinations, seed=42):
    source = Path(source).resolve()
    destinations = {name: Path(path).resolve() for name, path in destinations.items()}
    if set(destinations) != {'train', 'validation', 'test'} or len({source, *destinations.values()}) != 4:
        raise ValueError('Source and three output paths must be distinct')
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError('An output already exists; none will be overwritten')
    csv.field_size_limit(sys.maxsize)
    before = signature(source)
    rows_by_label = defaultdict(list)
    temps, installed = {}, []
    try:
        with source.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, strict=True)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)) or not {'body', 'label'}.issubset(fields):
                raise ValueError('Input must have unique headers including body and label')
            for row in reader:
                if None in row or any(v is None for v in row.values()) or not row['body'].strip() or row['label'].strip() not in ('0.0', '1.0'):
                    raise ValueError(f'Invalid body or label near line {reader.line_num}')
                rows_by_label[row['label'].strip()].append(row)
        if set(rows_by_label) != {'0.0', '1.0'} or min(map(len, rows_by_label.values())) < 10:
            raise ValueError('Both classes need at least ten records')
        rng = random.Random(seed)
        groups = {name: [] for name in destinations}
        for label in sorted(rows_by_label):
            rows = rows_by_label[label]
            rng.shuffle(rows)
            test_n = len(rows) // 10
            val_n = len(rows) // 10
            groups['test'].extend(rows[:test_n])
            groups['validation'].extend(rows[test_n:test_n + val_n])
            groups['train'].extend(rows[test_n + val_n:])
        counts = {}
        for name, dest in destinations.items():
            rng.shuffle(groups[name])
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=dest.parent, prefix='.split-fcn-', suffix='.csv', delete=False) as f:
                temps[name] = Path(f.name)
                writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(groups[name])
            counts[name] = Counter(row['label'].strip() for row in groups[name])
        if signature(source) != before:
            raise RuntimeError('Source changed during splitting')
        for name, dest in destinations.items():
            os.link(temps[name], dest)
            installed.append(dest)
        return counts
    except BaseException:
        for path in installed:
            path.unlink()
        raise
    finally:
        for path in temps.values():
            path.unlink(missing_ok=True)


def main():
    data = ROOT / 'data'
    try:
        counts = split(data / 'FCN_Phishing_Email_dataset_formatted.csv',
                       {name: data / f'FCN_{name}.csv' for name in ('train', 'validation', 'test')})
        for name, labels in counts.items():
            print(f'{name}: {sum(labels.values())} rows; labels: {dict(labels)}')
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
