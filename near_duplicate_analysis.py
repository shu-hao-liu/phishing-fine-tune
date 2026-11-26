#!/usr/bin/env python3
"""Find exact and high-Jaccard body matches between reference and test CSVs.

Uses exact prefix filtering for character-shingle Jaccard similarity; no approximate
nearest-neighbor index is used. Text is never written to the report.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path


def read_csv(path, column):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f'{path}: missing {column!r} column')
        return [r[column] or '' for r in reader]


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def shingles(text, size):
    return frozenset(text[i:i + size] for i in range(len(text) - size + 1)) if len(text) >= size else frozenset([text]) if text else frozenset()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'validation', 'test', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--column', default='body')
    p.add_argument('--threshold', type=float, default=.8)
    p.add_argument('--shingle-size', type=int, default=5)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--include-indices', action='store_true', help='Include 1-based matching data row numbers')
    a = p.parse_args()
    if not 0 < a.threshold <= 1 or a.shingle_size < 1:
        p.error('Threshold must be in (0, 1] and shingle size must be positive')
    if a.output.exists() and not a.overwrite:
        p.error(f'{a.output} exists; use --overwrite')
    paths = {'train': a.train, 'validation': a.validation, 'test': a.test}
    raw = {k: read_csv(v, a.column) for k, v in paths.items()}
    texts = {k: [normalize(t) for t in rows] for k, rows in raw.items()}
    # Index the smaller test set, then stream unique reference texts through it.
    # Shingle ordering is global and identical on both sides; prefix filtering
    # therefore cannot miss a pair that meets the requested Jaccard threshold.
    print('Building test index', flush=True)
    unique_test = list(set(texts['test']) - {''})
    grams = [shingles(t, a.shingle_size) for t in unique_test]
    freq = collections.Counter(g for item in grams for g in item)
    def ordered(item):
        return sorted(item, key=lambda g: (freq[g], g))
    index = collections.defaultdict(list)
    for i, item in enumerate(grams):
        seq = ordered(item)
        for gram in seq[:len(seq) - math.ceil(a.threshold * len(seq)) + 1]:
            index[gram].append(i)
    test_index = {t: i for i, t in enumerate(unique_test)}
    summary = {}
    for split in ('train', 'validation'):
        if split == 'validation' and a.validation.resolve() == a.train.resolve():
            summary[split] = dict(summary['train'])
            continue
        print(f'Comparing {split}', flush=True)
        refs = set(texts[split]) - {''}
        raw_ref = set(raw[split]) - {''}
        norm_match = set(unique_test) & refs
        near_match = set(norm_match)
        for rownum, ref in enumerate(refs, 1):
            gs = shingles(ref, a.shingle_size)
            n = len(gs)
            seq = ordered(gs)
            candidates = set()
            for gram in seq[:n - math.ceil(a.threshold * n) + 1]:
                candidates.update(index.get(gram, ()))
            for i in candidates:
                if unique_test[i] in near_match:
                    continue
                other = grams[i]
                m = len(other)
                if min(n, m) < a.threshold * max(n, m):
                    continue
                intersection = len(gs & other)
                if intersection / (n + m - intersection) >= a.threshold:
                    near_match.add(unique_test[i])
            if rownum % 5000 == 0:
                print(f'{split}: checked {rownum}/{len(refs)} unique reference bodies', flush=True)
        summary[split] = {
            'test_rows': len(texts['test']),
            'empty_test_bodies_excluded': sum(not t for t in texts['test']),
            'exact_raw_body_test_rows': sum(bool(t) and original in raw_ref for original, t in zip(raw['test'], texts['test'])),
            'exact_normalized_body_test_rows': sum(t in norm_match for t in texts['test']),
            'near_only_test_rows': sum(t in near_match and t not in norm_match for t in texts['test']),
            'exact_or_near_test_rows': sum(t in near_match for t in texts['test'])}
        if a.include_indices:
            summary[split]['exact_or_near_indices_1based'] = [i for i, t in enumerate(texts['test'], 1) if t in near_match]
            summary[split]['exact_normalized_indices_1based'] = [i for i, t in enumerate(texts['test'], 1) if t in norm_match]
    out = {'method': 'exact character-shingle Jaccard similarity with prefix filtering',
           'normalization': 'Unicode NFKC, casefold, whitespace collapsed and trimmed',
           'threshold': a.threshold, 'shingle_size': a.shingle_size, 'text_column': a.column,
           'unit': 'test rows with at least one match in each reference split; a row can match both splits',
           'interpretation': 'near_only excludes exact normalized text; matches are candidate similarities, not proof of leakage or duplicates',
           'inputs': {k: {'path': str(v), 'rows': len(raw[k]), 'sha256': hashlib.file_digest(v.open('rb'), 'sha256').hexdigest()} for k, v in paths.items()},
           'comparisons': summary}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved {a.output}', flush=True)


if __name__ == '__main__':
    main()
