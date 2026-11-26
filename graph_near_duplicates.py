#!/usr/bin/env python3
"""Plot cross-split exact and near-duplicate counts from saved JSON reports."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent / 'bootstrap-results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'near-duplicates-comparison.png')
    parser.add_argument('--overwrite', action='store_true')
    a = parser.parse_args()
    if a.output.suffix.lower() != '.png':
        parser.error('Output must be a PNG')
    if a.output.exists() and not a.overwrite:
        parser.error('Output exists; use --overwrite or choose a different name')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, language in zip(axes, ('english', 'mandarin')):
        base = json.loads((a.results_dir / f'near-duplicates-{language}.json').read_text())
        val = json.loads((a.results_dir / f'near-duplicates-{language}-validation.json').read_text())
        if (base['threshold'], base['shingle_size'], base['normalization']) != (val['threshold'], val['shingle_size'], val['normalization']):
            parser.error(f'Inconsistent settings for {language}')
        rows = [val['comparisons']['train'], base['comparisons']['train'], base['comparisons']['validation']]
        labels = ['Validation vs train', 'Test vs train', 'Test vs validation']
        x = np.arange(3)
        exact = [100 * row['exact_normalized_body_test_rows'] / row['test_rows'] for row in rows]
        near = [100 * row['near_only_test_rows'] / row['test_rows'] for row in rows]
        ax.bar(x, exact, color='#62748a', label='Exact after normalization')
        ax.bar(x, near, bottom=exact, color='#237a72', label='Near only')
        for i, row in enumerate(rows):
            count = row['exact_or_near_test_rows']
            ax.text(i, exact[i] + near[i] + .5, f"{count:,} / {row['test_rows']:,}\n({exact[i] + near[i]:.2f}%)", ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x, labels, rotation=15, ha='right')
        ax.set_title(language.title(), fontweight='bold')
        ax.set_ylim(0, 25)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Percentage of rows in the first-named split')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
    fig.suptitle('Cross-split similar email bodies', fontsize=16, fontweight='bold')
    fig.text(.5, .02, 'Normalized text; character 5-gram Jaccard similarity ≥ 0.80. Counts are rows with at least one match.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .87))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Saved {a.output}')


if __name__ == '__main__':
    main()
