#!/usr/bin/env python3
"""Prepare mutually exclusive match categories and run paired bootstraps."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('language', choices=('english', 'mandarin'))
a = p.parse_args()
language = a.language
root = Path('bootstrap-results')
report = json.loads((root / f'near-duplicates-{language}-indices.json').read_text())
paths = {k: Path(v['path']) for k, v in report['inputs'].items()}
def bodies(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [row['body'] for row in csv.DictReader(f)]
reference = set(bodies(paths['train'])) | set(bodies(paths['validation']))
test = bodies(paths['test'])
exact = {i for i, body in enumerate(test, 1) if body and body in reference}
normalized = set().union(*(set(report['comparisons'][s]['exact_normalized_indices_1based']) for s in ('train', 'validation')))
near = set().union(*(set(report['comparisons'][s]['exact_or_near_indices_1based']) for s in ('train', 'validation')))
assert exact <= normalized <= near
membership = {'exact-body': sorted(exact), 'normalized-only': sorted(normalized-exact), 'near-only': sorted(near-normalized)}
idx = root / f'{language}-match-categories.json'
if idx.exists(): raise SystemExit(f'{idx} already exists')
idx.write_text(json.dumps(membership, indent=2)+'\n')
summary=json.loads((root/'roc-mcnemar-20261119T035304150162Z/summary.json').read_text())
cmd=['python3','paired_bootstrap_groups.py','--test',str(paths['test']),'--indices',str(idx),'--definition','Exact body, normalized exact, or character 5-gram Jaccard >= 0.80 match against train or validation','--output-dir',str(root/f'paired-groups-{language}-categorized')]
for name, data in summary['roc_auc'][language].items():
    cmd += ['--model', name, data['input_csv']]
subprocess.run(cmd, check=True)
print({name: len(items) for name, items in membership.items()})
