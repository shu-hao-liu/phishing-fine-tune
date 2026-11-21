#!/usr/bin/env python3
"""Graph paired discordant counts and Holm-adjusted McNemar p-values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'bootstrap-results/roc-mcnemar-20261119T035304150162Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=ROOT / 'bootstrap-results/mcnemar-results.png')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('Output must have a .png extension')
    if args.output.exists() and not args.overwrite:
        parser.error(f'{args.output} exists; use --overwrite or a different output path')
    with (args.results_dir / 'summary.json').open(encoding='utf-8') as file:
        data = json.load(file)['mcnemar']
    with (args.results_dir / 'holm-adjusted-mcnemar.json').open(encoding='utf-8') as file:
        adjusted = {(r['dataset'], r['comparison']): r['holm_adjusted_p_value'] for r in json.load(file)['tests']}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, dataset in zip(axes, ('english', 'mandarin')):
        pairs = list(data[dataset].items())
        y = np.arange(len(pairs))
        for i, (name, row) in enumerate(pairs):
            a, b = name.split(' vs ')
            ax.barh(i - .18, row['only_a_correct'], height=.34, color='#62748a')
            ax.barh(i + .18, row['only_b_correct'], height=.34, color='#237a72')
            ax.text(row['only_a_correct'] + 2, i - .18, f"{a}: {row['only_a_correct']}", va='center', fontsize=9)
            ax.text(row['only_b_correct'] + 2, i + .18, f"{b}: {row['only_b_correct']}", va='center', fontsize=9)
            p = adjusted[dataset, name]
            ax.text(.99, i, f'adjusted p = {p:.3g}', transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=9,
                    bbox={'facecolor':'white', 'alpha':.9, 'edgecolor':'none', 'pad':2})
        ax.set_yticks(y, [name.replace(' vs ', ' vs ') for name, _ in pairs])
        ax.invert_yaxis()
        ax.set_xlim(0, 300)
        ax.set_title(dataset.title(), fontweight='bold')
        ax.set_xlabel('Test entries where only the named model was correct')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('McNemar’s test: paired model disagreements', fontsize=16, fontweight='bold')
    fig.text(.5, .02, 'Exact two-sided tests; Holm adjustment across six comparisons. English: n = 10,869; Mandarin: n = 303.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
