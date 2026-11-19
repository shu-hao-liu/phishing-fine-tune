#!/usr/bin/env python3
"""Remove chosen columns from CSV files after creating byte-for-byte backups."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            value.update(chunk)
    return value.digest()


def fingerprint(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def remove_url_column(paths, columns=('urls',)):
    """Preserve all other values and row order; never replace an existing backup."""
    columns = tuple(columns)
    if not columns or any(not isinstance(c, str) or not c.strip() for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('Columns must be a nonempty list of distinct column names')
    csv.field_size_limit(sys.maxsize)
    sources = [Path(p).expanduser().resolve() for p in paths]
    if len(sources) != len(set(sources)):
        raise ValueError('The same CSV was supplied more than once')
    plans, skipped, temps, modified = [], [], [], []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f'CSV not found: {source}')
        with source.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, strict=True)
            header = next(reader, None)
        if not header or len(set(header)) != len(header):
            raise ValueError(f'{source}: missing or duplicate column names')
        selected = [column for column in columns if column in header]
        if not selected:
            skipped.append(source)
            continue
        if len(selected) == len(header):
            raise ValueError(f'{source}: refusing to remove every column')
        suffix = '.before-url-removal.csv' if columns == ('urls',) else '.before-column-removal.csv'
        backup = source.with_name(source.stem + suffix)
        if backup.exists():
            raise FileExistsError(f'Backup already exists: {backup}')
        plans.append({'source': source, 'backup': backup, 'fingerprint': fingerprint(source),
                      'mode': stat.S_IMODE(source.stat().st_mode), 'columns': selected})
    if {p['backup'] for p in plans} & set(sources):
        raise ValueError('A backup path conflicts with an input path')

    # Create every backup before modifying any source CSV. A backup is a real
    # copy, so later writes to the source cannot alter it through a hard link.
    for plan in plans:
        with plan['source'].open('rb') as src, plan['backup'].open('xb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(plan['source'], plan['backup'])
        if fingerprint(plan['source']) != plan['fingerprint'] or digest(plan['source']) != digest(plan['backup']):
            raise RuntimeError(f"Source changed while making backup: {plan['source']}")

    try:
        # Stage every result before replacing any original; parse values exactly
        # with the CSV module, including fields containing commas and newlines.
        for plan in plans:
            with plan['backup'].open('rb') as f:
                probe = f.read(8192)
            bom = probe.startswith(b'\xef\xbb\xbf')
            line_ending = '\r\n' if b'\r\n' in probe else '\n'
            with plan['backup'].open('r', encoding='utf-8-sig', newline='') as src:
                reader = csv.reader(src, strict=True)
                header = next(reader)
                keep = [i for i, name in enumerate(header) if name not in plan['columns']]
                with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig' if bom else 'utf-8', newline='',
                                                 dir=plan['source'].parent, prefix='.remove-urls-', suffix='.csv', delete=False) as dst:
                    staged = Path(dst.name)
                    temps.append(staged)
                    writer = csv.writer(dst, lineterminator=line_ending)
                    writer.writerow([header[i] for i in keep])
                    count = 0
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f"{plan['source']}: malformed row near line {reader.line_num}")
                        writer.writerow([row[i] for i in keep])
                        count += 1
                    dst.flush()
                    os.fsync(dst.fileno())
            os.chmod(staged, plan['mode'])
            plan.update(staged=staged, records=count)
        for plan in plans:
            if fingerprint(plan['source']) != plan['fingerprint'] or digest(plan['source']) != digest(plan['backup']):
                raise RuntimeError(f"Source changed before replacement: {plan['source']}")
        for plan in plans:
            os.replace(plan['staged'], plan['source'])
            modified.append(plan)
        return plans, skipped
    except BaseException:
        # Restore any files already replaced if a later replacement fails.
        for plan in reversed(modified):
            with tempfile.NamedTemporaryFile('wb', dir=plan['source'].parent, prefix='.restore-urls-', suffix='.csv', delete=False) as dst:
                restored = Path(dst.name)
                temps.append(restored)
                with plan['backup'].open('rb') as src:
                    shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
            shutil.copystat(plan['backup'], restored)
            os.replace(restored, plan['source'])
        raise
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='+', help='CSV files to update')
    parser.add_argument('--columns', default='urls', help='comma-separated column names (default: urls)')
    args = parser.parse_args()
    try:
        columns = tuple(name.strip() for name in args.columns.split(','))
        plans, skipped = remove_url_column(args.csv, columns)
        for plan in plans:
            print(f"Updated {plan['source']}: {plan['records']} rows; backup: {plan['backup']}")
        for source in skipped:
            print(f'Skipped {source}: none of the selected columns exist')
    except (OSError, csv.Error, UnicodeError, ValueError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
