#!/usr/bin/env python3
"""Graph single-model bootstrap JSON results; no model or GPU is needed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
METRICS = [('accuracy', 'Accuracy'), ('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1 score')]
COLORS = {'Original': '#62748a', 'FFT': '#c87532', 'LoRA': '#237a72'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', nargs=3, metavar=('DATASET', 'MODEL', 'JSON'), action='append',
                        help='Result to plot; repeat for each dataset and model. Defaults to English and Mandarin results.')
    parser.add_argument('--output', type=Path, default=ROOT / 'bootstrap-results' / 'confidence-intervals-graph.png',
                        help='Output PNG path')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing an existing PNG')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('Output must have a .png extension.')
    if args.output.exists() and not args.overwrite:
        parser.error(f'{args.output} already exists; choose another name or use --overwrite.')
    entries = args.result or [(language.title(), model, str(ROOT / 'bootstrap-results' / f'single-{language}-{model.lower()}.json'))
                              for language in ('english', 'mandarin') for model in ('Original', 'FFT', 'LoRA')]
    datasets = {}
    try:
        for dataset, model, filename in entries:
            results = datasets.setdefault(dataset, {})
            if model in results:
                raise ValueError(f'Duplicate dataset/model pair: {dataset}/{model}')
            with open(filename, encoding='utf-8') as f:
                data = json.load(f)
            for key, _ in METRICS:
                metric = data['metrics'][key]
                if metric['estimate'] is None:
                    raise ValueError(f'{filename}: {key} estimate is undefined')
                interval = metric['confidence_interval']
                if interval is not None and (len(interval) != 2 or interval[0] > interval[1]):
                    raise ValueError(f'{filename}: invalid {key} interval')
            results[model] = data
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    fig, axes = plt.subplots(len(METRICS), len(datasets), squeeze=False,
                             figsize=(8.5 * len(datasets), 11.4))
    fig.suptitle('Phishing detection: bootstrap confidence intervals', fontsize=20, fontweight='bold')
    fig.text(.08, .94, 'Dots show estimates; lines show confidence intervals.', fontsize=11)
    for col, (dataset, models) in enumerate(datasets.items()):
        for row, (metric, label) in enumerate(METRICS):
            ax = axes[row][col]
            vals = [data['metrics'][metric] for data in models.values()]
            endpoints = [x for v in vals for x in ([v['estimate']] + (v['confidence_interval'] or []))]
            low, high = min(endpoints), max(endpoints)
            padding = max((high - low) * .12, .02)
            ax.set_xlim(max(0, low - padding), min(1.01, high + padding))
            for idx, ((model, data), value) in enumerate(zip(models.items(), vals)):
                y = len(models) - idx - 1
                color = COLORS.get(model, f'C{idx % 10}')
                estimate, interval = value['estimate'], value['confidence_interval']
                if interval is not None:
                    lo, hi = interval
                    ax.plot([lo, hi], [y, y], color=color, lw=3, solid_capstyle='round')
                    text = f'{estimate * 100:.2f}%  [{lo * 100:.2f}, {hi * 100:.2f}]'
                else:
                    text = f'{estimate * 100:.2f}%  [undefined]'
                ax.scatter([estimate], [y], s=42, c=color, edgecolors='white', linewidths=.6, zorder=3)
                ax.text(1.02, y, text, transform=ax.get_yaxis_transform(), va='center', fontsize=9, clip_on=False)
            ax.set_yticks(list(range(len(models) - 1, -1, -1)), list(models))
            ax.set_ylim(-.6, len(models) - .4)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='x', alpha=.25)
            ax.set_axisbelow(True)
            ax.set_title(f'{dataset} / {label}', loc='left', fontweight='bold')
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis='y', length=0)
    fig.subplots_adjust(left=.09, right=.77, top=.90, bottom=.07, hspace=.65, wspace=.85)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor='white')
    plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
