#!/usr/bin/env python3
"""Move the final 20% of CSV records into a separate CSV, retaining the header."""
import argparse
import csv
import os
from pathlib import Path
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


class RecordingLines:
    """Remember the exact physical lines consumed by csv.reader for each record."""
    def __init__(self, stream):
        self.stream = stream
        self.lines = []

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.stream)
        self.lines.append(line)
        return line

    def take(self):
        value = ''.join(self.lines)
        self.lines.clear()
        return value


def signature(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def split(train_path, inference_path):
    train = Path(train_path).expanduser().resolve()
    output = Path(inference_path).expanduser().resolve()
    if train == output:
        raise ValueError('Training and inference paths must differ')
    if output.exists():
        raise FileExistsError(f'Inference output already exists: {output}')
    if not output.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {output.parent}')
    csv.field_size_limit(sys.maxsize)
    before = signature(train)
    train_tmp = output_tmp = None
    try:
        with train.open('r', encoding='utf-8', newline='') as source:
            reader = csv.reader(source, strict=True)
            try:
                next(reader)
            except StopIteration:
                raise ValueError('CSV is empty; a header row is required') from None
            count = sum(1 for _ in reader)
            # Round up for a partial record so every nonempty dataset has a holdout.
            inference_count = (count + 4) // 5
            train_count = count - inference_count
            source.seek(0)
            lines = RecordingLines(source)
            reader = csv.reader(lines, strict=True)
            next(reader)
            header = lines.take()
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=train.parent, prefix='.split-train-', delete=False) as train_file:
                train_tmp = Path(train_file.name)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=output.parent, prefix='.split-inference-', delete=False) as output_file:
                    output_tmp = Path(output_file.name)
                    train_file.write(header)
                    output_file.write(header)
                    for index, _ in enumerate(reader):
                        (train_file if index < train_count else output_file).write(lines.take())
            if signature(train) != before:
                raise RuntimeError('Training CSV changed during the split; no output was installed')
            os.chmod(train_tmp, stat.S_IMODE(train.stat().st_mode))
            # Linking creates the destination exclusively; existing files are never overwritten.
            os.link(output_tmp, output)
            try:
                os.replace(train_tmp, train)
                train_tmp = None
            except BaseException:
                output.unlink()
                raise
        return train_count, inference_count
    finally:
        for temp in (train_tmp, output_tmp):
            if temp is not None:
                temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default=str(ROOT / 'train.csv'), help='CSV to split (default: project folder/train.csv)')
    parser.add_argument('--inference', default=str(ROOT / 'inference.csv'), help='new inference CSV (default: project folder/inference.csv)')
    args = parser.parse_args()
    try:
        train_count, inference_count = split(args.train, args.inference)
    except (OSError, UnicodeError, csv.Error, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    print(f'Training records: {train_count}; inference records: {inference_count}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
