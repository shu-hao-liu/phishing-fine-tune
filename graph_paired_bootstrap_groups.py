#!/usr/bin/env python3
"""Plot paired bootstrap metric differences by dataset and match group."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / 'bootstrap-results'
METRICS = ('accuracy', 'precision', 'recall', 'f1')
PAIRS = ('Original vs FFT', 'Original vs LoRA', 'FFT vs LoRA')
COLORS = ('#62748a', '#c87532', '#237a72')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT / 'paired-bootstrap-groups.png')
    p.add_argument('--group', choices=('duplicated', 'non-duplicated', 'exact-body', 'normalized-only', 'near-only'), action='append',
                   help='Group to plot, repeatable (default: duplicated and non-duplicated)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.output.suffix.lower() != '.png': p.error('Output must be a PNG')
    if args.output.exists() and not args.overwrite: p.error('Output exists; use --overwrite')
    results = {}
    for lang in ('english', 'mandarin'):
        path = args.results_dir / f'paired-groups-{lang}-categorized' / 'paired-bootstrap.json'
        results[lang] = json.loads(path.read_text(encoding='utf-8'))
    groups = args.group or ['duplicated', 'non-duplicated']
    fig, axes = plt.subplots(len(groups), 2, squeeze=False, figsize=(16, 5 * len(groups)))
    for i, group in enumerate(groups):
        for j, lang in enumerate(('english', 'mandarin')):
            ax = axes[i, j]
            data = results[lang]['groups'][group]
            ax.set_title(f"{lang.title()} — {group.replace('-', ' ')} (n = {data['count']:,})", loc='left', fontweight='bold')
            ax.axvline(0, color='#707c89', linewidth=1, linestyle='--')
            if not data['count']:
                ax.text(.5, .5, 'No rows in this group', transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_yticks([])
                ax.set_xlim(-1, 1)
            else:
                for metric_idx, metric in enumerate(METRICS):
                    for pair_idx, (pair, color) in enumerate(zip(PAIRS, COLORS)):
                        y = (len(METRICS) - 1 - metric_idx) * 4 + (1 - pair_idx)
                        value = data['comparisons'][pair]['metrics'][metric]
                        point, ci = value['difference_b_minus_a'], value['confidence_interval']
                        if point is None:
                            continue
                        ax.scatter(point * 100, y, s=28, color=color, zorder=3)
                        if ci is not None:
                            ax.plot([ci[0] * 100, ci[1] * 100], [y, y], lw=2.5, color=color)
                        else:
                            ax.annotate('CI undefined', (point * 100, y), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
                ax.set_yticks([i * 4 for i in range(3, -1, -1)], ['Accuracy', 'Precision', 'Recall', 'F1'])
                ax.margins(x=.15, y=.12)
            ax.set_xlabel('Difference in percentage points (second model − first model)')
            ax.grid(axis='x', alpha=.2)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker='o', color=color, label=pair, linewidth=2) for pair, color in zip(PAIRS, COLORS)]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle('Paired bootstrap: model differences and 95% confidence intervals', fontsize=16, fontweight='bold')
    fig.text(.5, .015, 'Dots = estimated difference; lines = unadjusted percentile intervals. Matched means exact or near match to train or validation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .89))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
