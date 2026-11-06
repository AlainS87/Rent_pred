"""Explore existing auxiliary indicators; this command never trains a model."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from src.paths import DATA_ROOT, ARTIFACT_ROOT

def run(raw_dir, output_dir):
    ROOT = Path(__file__).resolve().parents[1]
    RAW = Path(raw_dir).resolve()
    OUT = Path(output_dir).resolve()
    OUT.mkdir(exist_ok=True, parents=True)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    from src.features import stocks as stocks_mod
    from src.features import coe as coe_mod
    rent = pd.read_csv(RAW / 'train.csv')
    rent['month'] = pd.PeriodIndex(rent.RENT_APPROVAL_DATE, freq='M')
    rent['town_key'] = rent.TOWN.astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    rent['type_key'] = rent.FLAT_TYPE.astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    rent['unit_rent'] = rent.MONTHLY_RENT / rent.FLOOR_AREA_SQM
    months = pd.period_range(rent.month.min(), rent.month.max(), freq='M')
    cells = rent.groupby(['month', 'town_key', 'type_key']).unit_rent.agg(['median', 'count'])
    med = cells['median'].unstack(['town_key', 'type_key']).reindex(months)
    counts = cells['count'].unstack(['town_key', 'type_key']).reindex(months).fillna(0)
    # Adjacent-month matching limits changes in town/flat-type mix. It is not a hedonic index.
    valid = (counts >= 5) & (counts.shift(1) >= 5)
    cell_changes = np.log(med / med.shift(1)).where(valid)
    growth = cell_changes.mean(axis=1)
    growth.iloc[0] = 0.0
    market = pd.DataFrame(index=months)
    market['rent_matched_cell_unit_index'] = 100 * np.exp(growth.cumsum())
    market['rent_unit_median'] = rent.groupby('month').unit_rent.median().reindex(months)
    market['rent_monthly_median'] = rent.groupby('month').MONTHLY_RENT.median().reindex(months)
    market['rent_monthly_mean'] = rent.groupby('month').MONTHLY_RENT.mean().reindex(months)
    market['matched_cells'] = valid.sum(axis=1)
    market['matched_rows_current'] = counts.where(valid).sum(axis=1)
    market['rental_rows'] = rent.groupby('month').size().reindex(months)
    stocks = pd.read_csv(RAW / 'auxiliary/sg-stock-prices.csv')
    stocks['DATE'] = pd.to_datetime(stocks.DATE)
    # Deliberately exclude every observation beyond the last labelled rental month.
    stocks = stocks[stocks.DATE <= months[-1].end_time].copy()
    signals = pd.DataFrame(index=months)
    for name, symbols in [('bank', stocks_mod.BANK_SYMBOLS), ('property', stocks_mod.PROPERTY_SYMBOLS)]:
        wide = stocks_mod._month_end_closes(stocks, symbols).reindex(months)
        close, ret = stocks_mod._basket_series(wide)
        signals[f'{name}_existing_close'] = close
        signals[f'{name}_equal_weight_monthly_return'] = ret
        # An explanatory rebased return index; not a change to any model input.
        complete = wide.notna().all(axis=1)
        start = complete[complete].index[0]
        idx = (1 + ret.loc[start:].fillna(0)).cumprod() * 100
        signals[f'{name}_equal_weight_return_index'] = idx.reindex(months)

    coe = pd.read_csv(RAW / 'auxiliary/sg-coe-prices.csv')
    monthly_coe = coe_mod.monthly_coe(coe)
    for cat in ['a', 'b', 'c', 'e']:
        s = monthly_coe.xs(cat, level='category').reindex(months)
        signals[f'coe_{cat}_premium'] = s.premium
        signals[f'coe_{cat}_bids_per_quota'] = s.bids / s.quota.where(s.quota > 0)

    price_cols = [c for c in signals if c.endswith(('close', 'return_index', 'premium'))]
    ratio_cols = [c for c in signals if c.endswith('bids_per_quota')]
    target_cols = ['rent_matched_cell_unit_index', 'rent_unit_median', 'rent_monthly_median']
    rows = []
    def record(signal, target, test, x, y):
        xy = pd.concat([x.rename('x'), y.rename('y')], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(xy) < 6:
            return
        rows.append({'signal': signal, 'target': target, 'test': test, 'n_overlapping_months': len(xy),
                     'first_month': str(xy.index.min()), 'last_month': str(xy.index.max()),
                     'pearson_r': float(xy.x.corr(xy.y)), 'spearman_r': float(xy.x.corr(xy.y, method='spearman'))})

    for target in target_cols:
        y = market[target]
        for col in price_cols + ratio_cols:
            x = signals[col]
            # Lag1 is a full calendar-month lag, as in the current feature mapping.
            record(col, target, 'lag1_level_vs_rent_level', x.shift(1), y)
            record(col, target, 'lag1_monthly_log_change_vs_rent_monthly_log_change', np.log(x).diff().shift(1), np.log(y).diff())
            record(col, target, 'lag1_yoy_log_change_vs_rent_yoy_log_change', np.log(x).diff(12).shift(1), np.log(y).diff(12))
            # This is outcome-based exploratory association, never a future input or fitted forecast.
            for h in [6, 12]:
                mom3 = np.log(x).diff(3).shift(1)
                future_growth = np.log(y.shift(-h) / y)
                record(col, target, f'known_lag1_3m_momentum_vs_next_{h}m_rent_growth', mom3, future_growth)
                for tag, mask in [('early_through_2022', months <= pd.Period('2022-12', 'M')),
                                  ('later_from_2023', months >= pd.Period('2023-01', 'M'))]:
                    record(col, target, f'{tag}_known_lag1_3m_momentum_vs_next_{h}m_rent_growth', mom3[mask], future_growth[mask])

    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'descriptive_correlations.csv', index=False)
    all_series = market.join(signals)
    all_series.index.name = 'month'
    all_series.to_csv(OUT / 'monthly_series.csv')
    partial_rows = []
    common_rows = []
    maturity_rows = []
    lead_cols = ['bank_equal_weight_return_index', 'property_equal_weight_return_index',
                 'coe_a_premium', 'coe_b_premium', 'coe_c_premium', 'coe_e_premium']
    rent_log = np.log(market[target_cols[0]])
    known_momentum = np.log(signals[lead_cols]).diff(3).shift(1)
    for h in [6, 12]:
        future = (rent_log.shift(-h) - rent_log).rename('future')
        own = rent_log.diff(3).rename('own')
        common = known_momentum.join(future).join(own).dropna()
        corr = common.corr()
        for col in lead_cols:
            rxf, rxz, rzf = corr.loc[col, 'future'], corr.loc[col, 'own'], corr.loc['own', 'future']
            # Analytical three-variable partial correlation; no prediction or regression is fitted.
            partial = (rxf-rxz*rzf)/np.sqrt((1-rxz**2)*(1-rzf**2))
            common_rows.append({'signal': col, 'future_horizon_months': h, 'n_overlapping_months': len(common),
                                'first_month': str(common.index.min()), 'last_month': str(common.index.max()),
                                'pearson_r': float(rxf), 'partial_r_controlling_own_3m_rent_growth': float(partial)})
            single = known_momentum[[col]].join(future).join(own).dropna()
            sc = single.corr()
            sx, sz, yz = sc.loc[col, 'future'], sc.loc[col, 'own'], sc.loc['own', 'future']
            partial_rows.append({'signal': col, 'future_horizon_months': h, 'n_overlapping_months': len(single),
                                 'pearson_r': float(sx),
                                 'partial_r_controlling_own_3m_rent_growth': float((sx-sz*yz)/np.sqrt((1-sz**2)*(1-yz**2)))})
        for cutoff in ['2022-03', '2022-10', '2023-05']:
            c = pd.Period(cutoff, 'M')
            mature = known_momentum.loc[known_momentum.index+h <= c].dropna()
            maturity_rows.append({'cutoff': cutoff, 'future_horizon_months': h,
                                  'mature_common_supervised_origins': len(mature),
                                  'warning': 'overlapping months, not independent training samples'})
    pd.DataFrame(common_rows).to_csv(OUT / 'common_window_momentum_correlations.csv', index=False)
    pd.DataFrame(partial_rows).to_csv(OUT / 'exploratory_partial_correlations.csv', index=False)
    pd.DataFrame(maturity_rows).to_csv(OUT / 'historical_origin_sample_maturity.csv', index=False)
    endpoint_rows = []
    for start, end in [('2022-03', '2023-07'), ('2023-05', '2024-09')]:
        for col in ['rent_matched_cell_unit_index', 'rent_monthly_mean'] + price_cols:
            src = all_series[col]
            a, b = src.loc[start], src.loc[end]
            endpoint_rows.append({'start': start, 'end': end, 'series': col, 'start_value': float(a), 'end_value': float(b),
                                  'percent_change': float((b/a - 1)*100) if pd.notna(a) and pd.notna(b) else None})
    pd.DataFrame(endpoint_rows).to_csv(OUT / 'endpoint_changes.csv', index=False)

    report = {
        'analysis_type': 'descriptive and exploratory; no fitting or model predictions',
        'label_period': [str(months.min()), str(months.max())],
        'stock_and_coe_observations_after_label_end_used': False,
        'rental_rows': len(rent), 'monthly_observations': len(months),
        'rent_index_method': 'chain matched town x normalised flat-type median rent/sqm; >=5 observations in both adjacent months; equal cell weights; no regression',
        'rent_index_limits': 'partly controls composition only; no fixed address/size/age adjustment; descriptive target, not official index or newly trained model',
        'matched_cell_count_range_excluding_first_month': [int(valid.sum(axis=1).iloc[1:].min()), int(valid.sum(axis=1).iloc[1:].max())],
        'common_trends_and_multiple_comparisons': 'level correlations are not predictive proof; signal/lag tables are exploratory, not independent validation or significance tests',
        'common_window_comparison': 'compare all six fixed baskets/COE categories over identical complete months; 33 overlapping origins for h6, 27 for h12',
        'partial_correlation_limits': 'analytical 3-variable statistic controls only one linear own-rent momentum term; not causal evidence or a predictive performance measure',
        'candidate_selection_status': 'post hoc exploratory hypotheses after viewing pilot outcomes, not pre-registered or independent validation',
        'forward_growth_windows': '6m and 12m overlap substantially; raw row counts are not independent sample sizes',
        'historical_origin_sample_maturity': maturity_rows,
        'early_origin_limit': '2022-03 has only 15 labelled monthly macro observations; property basket begins 2021-09',
        'as_of_limits': 'calendar lag is not a release-date/vintage audit; adjusted closes may reflect later corporate-action adjustments',
        'models_trained': 0, 'new_model_predictions_generated': 0,
        'inputs_sha256': {str(path.relative_to(RAW)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          [RAW/'train.csv', RAW/'auxiliary/sg-stock-prices.csv', RAW/'auxiliary/sg-coe-prices.csv']},
        'outputs': ['monthly_series.csv', 'descriptive_correlations.csv', 'endpoint_changes.csv', 'trend_comparison.png',
                    'common_window_momentum_correlations.csv', 'exploratory_partial_correlations.csv',
                    'historical_origin_sample_maturity.csv'],
    }
    (OUT / 'analysis_metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    focus = table[(table.target == target_cols[0]) & table.signal.isin(['bank_existing_close','property_existing_close','coe_a_premium','coe_b_premium','coe_c_premium','coe_e_premium'])]
    print('COMMON WINDOW', pd.DataFrame(common_rows).round(4).to_string(index=False))
    print('MATURE HISTORICAL ORIGINS', json.dumps(maturity_rows, ensure_ascii=False))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        t = all_series.index.to_timestamp()
        plot_cols = [('rent_matched_cell_unit_index', 'Rental index (matched cells)', '#162c48'),
                     ('bank_equal_weight_return_index', 'Bank shares (equal-weight returns)', '#2678b2'),
                     ('property_equal_weight_return_index', 'Developer shares (equal-weight returns)', '#cc6644'),
                     ('coe_a_premium', 'COE A premium', '#4b8b62')]
        base = pd.Period('2021-10', freq='M')
        for col, label, color in plot_cols:
            s = all_series[col].where(all_series.index >= base)
            axes[0].plot(t, 100*s/all_series.loc[base,col], label=label, color=color, lw=2)
            axes[1].plot(t, s.pct_change(12, fill_method=None), label=label, color=color, lw=2)
        axes[0].set_title('Existing auxiliary signals and rental trend | Oct 2021 = 100', loc='left', fontweight='bold')
        axes[0].set_ylabel('Rebased level')
        axes[0].legend(loc='upper left', ncol=2, fontsize=9)
        axes[1].set_title('Year-on-year changes | same-month co-movement, not a forecast', loc='left', fontweight='bold')
        axes[1].set_ylabel('12-month change')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].axhline(0, color='#7f8c8d', lw=.8)
        for ax in axes:
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top','right']].set_visible(False)
        fig.supxlabel('Only months with rental labels are analysed (through Mar 2025).')
        fig.savefig(OUT / 'trend_comparison.png', dpi=160)
        plt.close(fig)
    except ImportError:
        report['outputs'].remove('trend_comparison.png')
        report['plot_skipped'] = 'matplotlib not installed; no package was installed'
        (OUT / 'analysis_metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OUTPUT', OUT)

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_ROOT / 'external_signals')
    args = parser.parse_args()
    run(args.raw_dir, args.output_dir)

if __name__ == '__main__':
    main()
