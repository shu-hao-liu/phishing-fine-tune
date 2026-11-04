#!/usr/bin/env python3
"""Keep only subject, body, urls, label, language and source in CSV files."""
import argparse
import csv
import os
from pathlib import Path
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
COLUMNS = ('subject', 'body', 'urls', 'label', 'language', 'source')


def signature(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def clean(path):
    path = Path(path).expanduser().resolve()
    backup = path.with_name(path.stem + '.original.csv')
    before = signature(path)
    csv.field_size_limit(sys.maxsize)
    temporary = None
    try:
        with path.open('rb') as probe:
            has_bom = probe.read(3) == b'\xef\xbb\xbf'
        encoding = 'utf-8-sig' if has_bom else 'utf-8'
        with path.open('r', encoding=encoding, newline='') as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f'{path}: CSV has no header') from None
            if len(set(header)) != len(header):
                raise ValueError(f'{path}: duplicate column names')
            missing = [column for column in COLUMNS if column not in header]
            if missing:
                raise ValueError(f'{path}: missing required columns: {missing}')
            if tuple(header) == COLUMNS:
                return 0, None
            if backup.exists():
                raise FileExistsError(f'Backup already exists: {backup}; move it before cleaning again')
            indices = [header.index(column) for column in COLUMNS]
            with tempfile.NamedTemporaryFile(mode='w', encoding=encoding, newline='', dir=path.parent, prefix='.clean-', suffix='.csv', delete=False) as dest:
                temporary = Path(dest.name)
                writer = csv.writer(dest, lineterminator='\n')
                writer.writerow(COLUMNS)
                count = 0
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(f'{path}: malformed record ending near line {reader.line_num}')
                    writer.writerow([row[i] for i in indices])
                    count += 1
        if signature(path) != before:
            raise RuntimeError(f'{path}: file changed during cleaning; no changes applied')
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        # Same-directory hard link retains the original without making a full copy.
        os.link(path, backup)
        try:
            if signature(path) != before:
                raise RuntimeError(f'{path}: file changed during cleaning; no changes applied')
            os.replace(temporary, path)
            temporary = None
        except BaseException:
            backup.unlink()
            raise
        return count, backup
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', default=[str(ROOT / 'train.csv'), str(ROOT / 'inference.csv')], help='CSV files to clean (default: project train.csv and inference.csv)')
    args = parser.parse_args()
    try:
        for name in args.files:
            count, backup = clean(name)
            if backup is None:
                print(f'{name}: already has only the requested columns.', flush=True)
            else:
                print(f'{name}: kept {count} records; original backed up to {backup}', flush=True)
    except (OSError, csv.Error, UnicodeError, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
