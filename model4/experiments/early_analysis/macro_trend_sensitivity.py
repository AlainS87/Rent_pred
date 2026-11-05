"""Post-hoc macro association diagnostics, never forecasting feature selection.

The development cutoff, dwelling controls, lagged inputs and minimum 18 observed
months match associations.py. Existing tables and frozen model code are untouched.
Polynomial trends and HAC6 probe shared nonlinear trends and overlapping windows;
they cannot certify causal effects or a correctly specified short time series.
"""
from __future__ import annotations
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import time
import runtime
from runtime import ANALYSIS
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from associations import CUTOFF, MIN_MACRO_MONTHS, _bh, _control_design, _partial_macro, _residualizer
SPECIFICATIONS = [('linear_hac3', 1, 3), ('linear_hac6', 1, 6), ('quadratic_hac6', 2, 6), ('cubic_hac6', 3, 6)]

def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _controls(months: pd.PeriodIndex, positions: np.ndarray, degree: int) -> np.ndarray:
    t = (positions - positions.mean()) / max(float(positions.std()), 1)
    columns = [np.ones(len(months)), *[t ** power for power in range(1, degree + 1)], np.sin(2 * np.pi * months.month / 12), np.cos(2 * np.pi * months.month / 12)]
    return np.column_stack(columns)

def _partial_hac(x, y, controls, calendar_positions, bandwidth):
    rx = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
    ry = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return {'status': 'no_variation_after_time_controls'}
    denominator = float(rx @ rx)
    beta = float(rx @ ry / denominator)
    r = float(np.corrcoef(rx, ry)[0, 1])
    scores = rx * (ry - beta * rx)
    full = np.zeros(int(calendar_positions.max()) + 1)
    full[calendar_positions] = scores
    meat = float(scores @ scores)
    for lag in range(1, bandwidth + 1):
        if lag < len(full):
            meat += 2 * (1 - lag / (bandwidth + 1)) * float(full[lag:] @ full[:-lag])
    df = len(x) - controls.shape[1] - 1
    if df <= 0:
        return {'status': 'insufficient_residual_degrees_of_freedom'}
    variance = max(meat, 0) * len(x) / df / denominator ** 2
    se = float(np.sqrt(variance))
    p = float(2 * student_t.sf(abs(beta / se), df)) if se > 0 else np.nan
    critical = float(student_t.ppf(0.975, df))
    return {'status': 'tested', 'partial_correlation': r, 'coefficient_sgd_per_unit': beta, 'hac_standard_error': se, 'p_value': p, 'degrees_of_freedom': df, 'ci_low_95': beta - critical * se, 'ci_high_95': beta + critical * se}

def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def _direct_full_ols_hac(x, y, controls, calendar_positions, bandwidth):
    """Independent full-design sandwich check, rather than FWL scalar scores."""
    scale = float(np.std(x))
    standardized = (x - x.mean()) / scale
    design = np.column_stack([controls, standardized])
    assert np.linalg.matrix_rank(design) == design.shape[1]
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    errors = y - design @ coefficient
    score = design * errors[:, None]
    calendar_score = np.zeros((int(calendar_positions.max()) + 1, design.shape[1]))
    calendar_score[calendar_positions] = score
    meat = score.T @ score
    for lag in range(1, bandwidth + 1):
        if lag < len(calendar_score):
            cross = calendar_score[lag:].T @ calendar_score[:-lag]
            meat += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
    inverse = np.linalg.inv(design.T @ design)
    covariance = inverse @ meat @ inverse * len(x) / (len(x) - design.shape[1])
    return (float(coefficient[-1] / scale), float(np.sqrt(max(covariance[-1, -1], 0)) / scale))

def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = ANALYSIS / 'results'
    original_files = [results / 'monthly_macro_screen.csv', results / 'association_notes.json', results / 'controlled_spatial_associations.csv', results / 'adjusted_distance_bins.csv']
    protected_hashes = {path.name: _digest(path) for path in original_files}
    baseline_saved = pd.read_csv(results / 'monthly_macro_screen.csv').set_index('feature')
    full = pd.read_pickle(ANALYSIS / 'features' / 'train_features.pkl')
    dev = full.loc[full['month'].le(CUTOFF)].copy()
    del full
    assert dev.month.nunique() == 34 and dev.month.max() == '2023-10'
    target = dev.target.to_numpy(dtype=float)
    composition_residual = _residualizer(_control_design(dev, include_month=False))(target)
    composition = pd.DataFrame({'month': dev.month.to_numpy(), 'rent': target, 'composition_residual': composition_residual}).groupby('month').mean()
    families = json.loads((ANALYSIS / 'features' / 'feature_families.json').read_text())
    candidates = [feature for family in ('coe', 'stocks', 'stocks_adjusted_sensitivity') for feature in families[family]]
    assert len(candidates) == len(set(candidates)) == 1595
    monthly = pd.read_pickle(ANALYSIS / 'features' / 'macro_monthly.pkl').reindex(composition.index)
    months = pd.PeriodIndex(composition.index, freq='M')
    positions = np.array([month.ordinal - months[0].ordinal for month in months])
    original_controls = np.column_stack([np.ones(len(months)), positions, np.sin(2 * np.pi * months.month / 12), np.cos(2 * np.pi * months.month / 12)])
    family_of = {feature: family for family in ('coe', 'stocks', 'stocks_adjusted_sensitivity') for feature in families[family]}
    records = []
    for specification, degree, bandwidth in SPECIFICATIONS:
        controls = original_controls if specification == 'linear_hac3' else _controls(months, positions, degree)
        rows = []
        for feature in candidates:
            x = monthly[feature].to_numpy(dtype=float)
            valid = np.isfinite(x)
            n = int(valid.sum())
            row = {'feature': feature, 'family': family_of[feature], 'symbol': baseline_saved.at[feature, 'symbol'], 'specification': specification, 'trend_degree': degree, 'hac_calendar_lags': bandwidth, 'observed_months': n, 'missing_months': len(months) - n, 'development_months': len(months), 'sensitivity_only': family_of[feature] == 'stocks_adjusted_sensitivity'}
            if n < MIN_MACRO_MONTHS:
                row['status'] = 'insufficient_months_minimum_18'
            elif np.unique(x[valid]).size < 2:
                row['status'] = 'constant_in_development'
            else:
                y = composition.composition_residual.to_numpy()[valid]
                if specification == 'linear_hac3':
                    result = _partial_macro(x[valid], y, controls[valid], positions[valid])
                    result['hac_standard_error'] = result.pop('hac3_standard_error', np.nan)
                    result['p_value'] = result.pop('p_value_hac3', np.nan)
                    result.pop('p_value_iid_month', None)
                    if result['status'] == 'tested':
                        delta = student_t.ppf(0.975, result['degrees_of_freedom']) * result['hac_standard_error']
                        result['ci_low_95'] = result['coefficient_sgd_per_unit'] - delta
                        result['ci_high_95'] = result['coefficient_sgd_per_unit'] + delta
                else:
                    result = _partial_hac(x[valid], y, controls[valid], positions[valid], bandwidth)
                row.update(result)
            rows.append(row)
        spec_frame = pd.DataFrame(rows)
        spec_frame['bh_q_value'] = _bh(spec_frame['p_value'])
        spec_frame['bh_exploratory_05'] = spec_frame.bh_q_value.le(0.05)
        records.append(spec_frame)
    long = pd.concat(records, ignore_index=True)
    baseline = long.loc[long.specification.eq('linear_hac3')].set_index('feature').reindex(baseline_saved.index)
    assert baseline.status.equals(baseline_saved.status)
    replication = {}
    for actual, saved in (('partial_correlation', 'partial_correlation'), ('coefficient_sgd_per_unit', 'coefficient_sgd_per_unit'), ('hac_standard_error', 'hac3_standard_error'), ('p_value', 'p_value_hac3'), ('bh_q_value', 'bh_q_value_hac3')):
        a, b = (baseline[actual].to_numpy(), baseline_saved[saved].to_numpy())
        assert np.allclose(a, b, rtol=1e-08, atol=1e-08, equal_nan=True), actual
        replication[actual] = float(np.nanmax(np.abs(a - b)))
    identity = baseline[['family', 'symbol', 'observed_months', 'missing_months', 'sensitivity_only']].copy()
    fields = ['status', 'partial_correlation', 'coefficient_sgd_per_unit', 'hac_standard_error', 'p_value', 'bh_q_value', 'degrees_of_freedom', 'ci_low_95', 'ci_high_95']
    paired = identity.copy()
    per_spec = {}
    for specification, _, _ in SPECIFICATIONS:
        data = long.loc[long.specification.eq(specification)].set_index('feature').reindex(paired.index)
        per_spec[specification] = data
        paired = paired.join(data[fields].add_prefix(specification + '_'))
    summary_counts = []
    for (specification, family), part in long.groupby(['specification', 'family'], sort=False):
        tested = part.status.eq('tested')
        summary_counts.append({'specification': specification, 'family': family, 'candidate_count': len(part), 'tested': int(tested.sum()), 'bh_q_below_05': int(part.bh_q_value.lt(0.05).sum()), 'positive_slopes_tested': int((tested & part.coefficient_sgd_per_unit.gt(0)).sum()), 'negative_slopes_tested': int((tested & part.coefficient_sgd_per_unit.lt(0)).sum())})
    base_significant = baseline.bh_q_value.lt(0.05)
    stability = []
    for specification in ('linear_hac6', 'quadratic_hac6', 'cubic_hac6'):
        data = per_spec[specification]
        both = baseline.status.eq('tested') & data.status.eq('tested')
        same = np.sign(baseline.coefficient_sgd_per_unit).eq(np.sign(data.coefficient_sgd_per_unit))
        stability.append({'specification': specification, 'paired_tested': int(both.sum()), 'same_direction': int((both & same).sum()), 'direction_flipped': int((both & ~same).sum()), 'baseline_q_below_05': int(base_significant.sum()), 'baseline_significant_still_q_below_05': int((base_significant & data.bh_q_value.lt(0.05)).sum()), 'baseline_significant_same_direction': int((base_significant & both & same).sum())})
    example_names = ['stock_m44u_si_daily_vol_trailing_6m', 'coe_a_premium_mean_trailing_6m', 'coe_a_premium_last_trailing_6m']
    example_names += baseline.loc[baseline.status.eq('tested')].nsmallest(3, 'bh_q_value').index.tolist()
    example_names = list(dict.fromkeys((feature for feature in example_names if feature in paired.index)))
    examples = paired.loc[example_names].reset_index()
    verification = []
    for feature in example_names:
        x = monthly[feature].to_numpy(dtype=float)
        valid = np.isfinite(x)
        y = composition.composition_residual.to_numpy()[valid]
        for specification, degree, bandwidth in SPECIFICATIONS[1:]:
            beta, se = _direct_full_ols_hac(x[valid], y, _controls(months, positions, degree)[valid], positions[valid], bandwidth)
            saved = per_spec[specification].loc[feature]
            assert np.allclose([beta, se], [saved.coefficient_sgd_per_unit, saved.hac_standard_error], rtol=1e-07, atol=1e-08)
            verification.append({'feature': feature, 'specification': specification, 'direct_full_ols_beta': beta, 'direct_full_ols_hac_se': se, 'beta_absolute_difference': abs(beta - saved.coefficient_sgd_per_unit), 'se_absolute_difference': abs(se - saved.hac_standard_error), 'passed': True})
    summary = {'status': 'Post-hoc targeted exploratory diagnostic after inspecting the initial macro association screen', 'development_cutoff': CUTOFF, 'development_rows': len(dev), 'distinct_monthly_observations': len(months), 'confirmation_labels_used': False, 'model_column_selection_uses_this_diagnostic': False, 'original_model_features_splits_parameters_changed': False, 'minimum_observed_months': MIN_MACRO_MONTHS, 'candidate_count_each_specification': len(candidates), 'specifications': [{'name': name, 'trend_degree': degree, 'hac_calendar_lags': bandwidth} for name, degree, bandwidth in SPECIFICATIONS], 'composition_controls': 'Exact associations._control_design(include_month=False) and tiny Ridge0.001 residualizer: area/logarea/lease_age/flat_type/flat_model/town', 'annual_controls': 'Same annual sin/cos in every specification', 'hac_policy': 'Bartlett weights, score cross-products at actual calendar separation, finite-sample n/df correction; t reference remains approximate', 'bh_policy': 'Separately across finite p values of all candidates for each specification; counts are exploratory, not feature-selection thresholds', 'baseline_replication_max_absolute_errors': replication, 'independent_full_design_OLS_sandwich_checks': verification, 'counts_by_specification_and_family': summary_counts, 'stability_relative_to_linear_hac3': stability, 'examples_selected_for_diagnosis': example_names, 'examples_selection_caution': 'Explicit concern examples plus three smallest initial q values; selected post-hoc for illustration, never to choose models', 'limitations': 'Polynomial trend degree and HAC bandwidth are diagnostics, not known true data-generating parameters. Overlapping windows, serial dependence, common nonlinear trends, composition-fit uncertainty, multiple dependent hypotheses and only 34 distinct months still limit inference. Neither surviving nor disappearing q values establish causality or future forecasting value.', 'original_association_file_hashes_unchanged': protected_hashes, 'elapsed_seconds': round(time.perf_counter() - started, 3)}
    for path in original_files:
        assert _digest(path) == protected_hashes[path.name], path.name
    long.to_csv(results / 'macro_trend_sensitivity_long.csv', index=False)
    paired.reset_index().to_csv(results / 'macro_trend_sensitivity_paired.csv', index=False)
    pd.DataFrame(summary_counts).to_csv(results / 'macro_trend_sensitivity_counts.csv', index=False)
    examples.to_csv(results / 'macro_trend_sensitivity_examples.csv', index=False)
    (results / 'macro_trend_sensitivity_summary.json').write_text(json.dumps(_clean(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    notes = ['宏观关联的趋势/序列依赖敏感性（事后探索诊断）', '', '初始monthly_macro_screen的极小q值引起共享非线性趋势、6个月窗口重叠与HAC3覆盖不足的担忧，因此追加本诊断。', '仅使用截至2023-10的99,904条开发交易、34个不同月份；不读取确认集标签、不选择模型/特征，也不修改原始关联表或冻结实现。', '住宅composition residual完全复用associations.py的area/logarea/lease_age/flat_type/flat_model/town控制与tiny Ridge0.001。', '四个时间控制规格：线性trend+HAC3重现原表；线性trend+HAC6单独检查带宽改变；二次/三次trend+HAC6分别增加趋势灵活性。全部保留相同annual sin/cos，至少18个可用月。', '每个规格都保留全部1595候选的coverage/status，分别在该规格内进行BH；没有跨四个规格同时控制FDR。多规格q值的变化只是稳健性描述，没有检验哪一个趋势一定正确。', '']
    for specification, _, _ in SPECIFICATIONS:
        part = long.loc[long.specification.eq(specification)]
        notes.append(f"{specification}：tested={int(part.status.eq('tested').sum())}；q<0.05={int(part.bh_q_value.lt(0.05).sum())}。")
    notes.append('')
    for row in stability:
        notes.append(f"相对baseline，{row['specification']}：同方向{row['same_direction']}/{row['paired_tested']}；baseline的{row['baseline_q_below_05']}个q<.05中，仍q<.05者{row['baseline_significant_still_q_below_05']}。")
    notes += ['', '具体示例（事后为诊断选择，不可据此选预测特征）：']
    for feature in example_names:
        values = []
        for specification, _, _ in SPECIFICATIONS:
            row = per_spec[specification].loc[feature]
            values.append(f'{specification}: r={row.partial_correlation:.4f}, q={row.bh_q_value:.4g}')
        notes.append(feature + ' — ' + '；'.join(values))
    notes += ['', '最重要的限制：更长HAC带宽不一定提高标准误或减少q<.05，样本中的负自协方差可以产生相反结果。HAC6并不保证消除所有序列相关，三次趋势也不是已知真实趋势；rolling窗口重叠、未观测共同冲击、住宅控制拟合误差和34个月的有限支持仍然存在。', '某个q值消失，不证明因素现实中没有影响；某个q值保留，也不证明因果，更不保证未来预测有效。预测效果继续以锁定时间外评估为准。', f'原表SHA256检查通过；{len(verification)}个完整OLS矩阵sandwich检查与FWL标量结果一致。底表为results/macro_trend_sensitivity_{{long,paired,counts,examples}}.csv与summary.json。']
    (ANALYSIS / 'RESULTS_MACRO_TREND_SENSITIVITY_CN.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(json.dumps(_clean({'counts': summary_counts, 'stability': stability, 'baseline_replication': replication, 'seconds': summary['elapsed_seconds']}), indent=2), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
