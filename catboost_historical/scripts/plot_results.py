#!/usr/bin/env python3
"""Plot development/holdout RMSE, plus a fixed-iteration comparison."""
import argparse
import os
from pathlib import Path
from hdb_rent.data import OUTPUT_DIR
os.environ.setdefault('MPLCONFIGDIR', str(OUTPUT_DIR / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(OUTPUT_DIR / 'cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/controlled_sweep')
    a = p.parse_args()
    summary = pd.read_csv(a.input_dir/'scores_summary.csv')
    for metric, ylabel, filename in [('dev_mean_rmse','Mean development-fold RMSE','development_rmse.png'),
                                     ('holdout_rmse','2025 Q1 diagnostic RMSE','holdout_rmse.png')]:
        if metric not in summary or summary[metric].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(9,5))
        for name in sorted(summary.model.unique()):
            s = summary[summary.model==name].sort_values('iterations')
            ax.plot(s.iterations, s[metric], marker='o', label=name)
        ax.set_xlabel('CatBoost iterations')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + ' (lower is better)')
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.input_dir/filename, dpi=160)
        plt.close(fig)
    ablation = pd.read_csv(a.input_dir/'ablation_fixed_iterations.csv').sort_values('model')
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(ablation.model, ablation.dev_mean_rmse)
    ax.set_title(f'Controlled A/B/C/D ({int(ablation.iterations.iloc[0])} iterations)')
    ax.set_ylabel('Development mean RMSE')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(a.input_dir/'ablation_fixed_iterations.png', dpi=160)
    plt.close(fig)
    print(f'Plots saved to {a.input_dir}')


if __name__=='__main__':
    main()
