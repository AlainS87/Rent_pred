"""Run selected research stages. Training requires the explicit --refit flag."""
from runtime import PROJECT, ANALYSIS, CODE_DIR
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refit', action='store_true', help='Rebuild features and refit the original locked comparisons.')
    args = parser.parse_args()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    stages = ['build_features.py', 'associations.py', 'spatial_sensitivity.py', 'run_experiments.py', 'run_history_experiment.py'] if args.refit else []
    stages += ['macro_trend_sensitivity.py', 'distribution_study.py', 'distribution_original_dwelling.py']
    for stage in stages:
        subprocess.run([sys.executable, str(CODE_DIR / stage)], cwd=PROJECT, check=True)

if __name__ == '__main__':
    main()
