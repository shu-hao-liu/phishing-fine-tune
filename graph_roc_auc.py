#!/usr/bin/env python3
"""Plot saved ROC-AUC scores for each model and dataset."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / 'bootstrap-results/roc-mcnemar-20261119T035304150162Z/summary.json'
DEFAULT_OUTPUT = ROOT / 'bootstrap-results/roc-auc-scores.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Results summary JSON')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output PNG path')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing PNG')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('Output must be a PNG file')
    if args.output.exists() and not args.overwrite:
        parser.error(f'{args.output} already exists; use --overwrite or choose another output')
    with args.input.open(encoding='utf-8') as f:
        results = json.load(f)['roc_auc']
    datasets = ['english', 'mandarin']
    models = ['Original', 'FFT', 'LoRA']
    colors = ['#62748a', '#c87532', '#237a72']
    x = np.arange(len(datasets))
    width = .22
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [results[dataset][model]['roc_auc'] for dataset in datasets]
        bars = ax.bar(x + (i - 1) * width, values, width, label=model, color=color)
        ax.bar_label(bars, labels=[f'{value:.4f}' for value in values], padding=4, fontsize=10)
    ax.set_xticks(x, ['English\n(n = 10,869)', 'Mandarin\n(n = 303)'])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Test-set ROC-AUC by model and dataset', fontsize=15, fontweight='bold')
    ax.axhline(.5, color='#777777', linestyle='--', linewidth=1, label='Random ranking (0.5)')
    ax.legend(ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.19))
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
