"""Permitted preprocessing values; no search or record-writing entry point."""

PREPROCESS_KNOBS = {'numeric_impute': ('median', 'mean', 'zero'), 'scale': ('standard', 'none', 'minmax'), 'categorical': ('onehot', 'ordinal', 'target', 'bin'), 'rare_min_count': (10, 30, 100)}
