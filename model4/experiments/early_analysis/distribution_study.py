"""Descriptive distribution figures, separate from locked predictive experiments.

All train labels may be summarized here because model configurations are locked.
The controls below are an in-sample association fit, not another predictive score
and not a causal adjustment.  Geographic panels give each observed block one
vote after aggregation; other plots explicitly retain transaction weighting.
"""
from __future__ import annotations
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import json
from pathlib import Path
import runtime
from runtime import ANALYSIS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from history_encoding import construct_history
FIGURES = ANALYSIS / 'figures'
RESULTS = ANALYSIS / 'results'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 10, 'axes.titleweight': 'bold', 'axes.edgecolor': '#d2dbe5', 'axes.labelcolor': '#25384c', 'text.color': '#132a41', 'xtick.color': '#41566c', 'ytick.color': '#41566c', 'figure.facecolor': 'white', 'axes.facecolor': '#fbfcfe', 'savefig.facecolor': 'white', 'pdf.fonttype': 42})

def _label(value: str) -> str:
    return str(value).replace('_', ' ').title().replace('Mrt', 'MRT').replace('Coe', 'COE').replace('Cbd', 'CBD')

def _first(frame: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in frame), None)

def _save(fig: plt.Figure, stem: str, note: str, manifest: list[dict]) -> None:
    fig.text(0.012, -0.014, note, ha='left', va='top', fontsize=9, color='#4c6279')
    for suffix in ('png', 'pdf'):
        fig.savefig(FIGURES / f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
    manifest.append({'figure': stem, 'png': str(FIGURES / f'{stem}.png'), 'note': note})
    plt.close(fig)

def _heatmap(ax, values: pd.DataFrame, title: str, cmap='viridis', vmin=None, vmax=None, annotate=False, fmt='.0f', norm=None, xlabel='', ylabel='', xtick_step=1):
    color = plt.get_cmap(cmap).copy()
    color.set_bad('#edf1f5')
    array = values.to_numpy(dtype=float)
    image = ax.imshow(np.ma.masked_invalid(array), aspect='auto', cmap=color, vmin=vmin if norm is None else None, vmax=vmax if norm is None else None, norm=norm)
    x = np.arange(0, len(values.columns), xtick_step)
    ax.set_xticks(x, [_label(values.columns[i]) for i in x], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(values.index)), [_label(value) for value in values.index])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(length=0)
    if annotate:
        limits = image.get_clim()
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                value = array[row, col]
                if np.isfinite(value):
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    text_color = 'white' if luminance < 0.52 else '#17324b'
                    ax.text(col, row, format(value, fmt), ha='center', va='center', fontsize=8, color=text_color)
    return image

def _composition_residual(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    numeric = [name for name in ('floor_area_sqm', 'log_floor_area') if name in frame]
    categorical = [name for name in ('flat_type', 'flat_model', 'town') if name in frame]
    inputs = frame[numeric + categorical].copy()
    for name in categorical:
        inputs[name] = inputs[name].fillna('__MISSING__').astype(str)
    transformer = ColumnTransformer([('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric), ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical)])
    association = make_pipeline(transformer, Ridge(alpha=100, solver='lsqr'))
    association.fit(inputs, frame['target'].to_numpy())
    prediction = association.predict(inputs)
    residual = frame['target'].to_numpy() - prediction
    note = {'fit_scope': 'All training transactions; descriptive in-sample association only', 'outcome': 'MONTHLY_RENT, SGD/month', 'numeric_controls': numeric, 'categorical_controls': categorical, 'alpha_fixed': 100, 'calendar_controls': 'None, so month patterns remain visible', 'residual_definition': 'Observed rent minus in-sample composition fit', 'causal_interpretation': False}
    return (residual, note)

def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_pickle(ANALYSIS / 'features' / 'train_features.pkl')
    if not {'month', 'target', 'town', 'flat_type', 'address_key'} <= set(frame):
        raise ValueError('Complete rebuilt training features are required')
    manifest: list[dict] = []
    date_range = f"{frame['month'].min()} to {frame['month'].max()}"
    scope = f'{len(frame):,} training transactions | {date_range}'
    flat_order = [name for name in ('1 room', '2 room', '3 room', '4 room', '5 room', 'executive') if name in frame['flat_type'].unique()]
    flat_order += sorted(set(frame['flat_type'].dropna()) - set(flat_order))
    towns = sorted(frame['town'].dropna().unique())
    grouped = frame.groupby(['town', 'flat_type'])['target'].agg(['median', 'count'])
    median = grouped['median'].unstack().reindex(index=towns, columns=flat_order)
    count = grouped['count'].unstack().reindex(index=towns, columns=flat_order).fillna(0)
    displayed = median.where(count >= 30)
    fig, axes = plt.subplots(1, 2, figsize=(17, 12), layout='constrained')
    fig.suptitle(f'Where do observed rents differ?\n{scope}', fontsize=18, fontweight='bold')
    im = _heatmap(axes[0], displayed, 'Median monthly rent (cells with n >= 30)', annotate=True, fmt=',.0f', xlabel='Flat type', ylabel='Town')
    fig.colorbar(im, ax=axes[0], shrink=0.75, label='SGD per month')
    im = _heatmap(axes[1], count.where(count > 0), 'Transaction sample size', 'cividis', annotate=True, fmt=',.0f', norm=LogNorm(vmin=1, vmax=max(2, count.max().max())), xlabel='Flat type')
    fig.colorbar(im, ax=axes[1], shrink=0.75, label='Transactions (log colour scale)')
    _save(fig, 'distribution_01_town_flat', 'One transaction = one vote. Grey cells lack support; raw differences mix time, dwelling composition and local markets.', manifest)
    grouped.reset_index().to_csv(RESULTS / 'distribution_town_flat.csv', index=False)
    frame = frame.copy()
    frame['composition_residual'], control_audit = _composition_residual(frame)
    month_town = frame.groupby(['town', 'month']).agg(rent_median=('target', 'median'), residual_mean=('composition_residual', 'mean'), transactions=('target', 'size'))
    calendar = sorted(frame['month'].unique())
    n = month_town['transactions'].unstack().reindex(index=towns, columns=calendar)
    raw = month_town['rent_median'].unstack().reindex(index=towns, columns=calendar).where(n >= 20)
    residual = month_town['residual_mean'].unstack().reindex(index=towns, columns=calendar).where(n >= 20)
    limit = max(1, np.nanquantile(np.abs(residual.to_numpy()), 0.95))
    fig, axes = plt.subplots(2, 1, figsize=(18, 15), layout='constrained')
    fig.suptitle(f'Rent changes across time and towns\n{scope}', fontsize=18, fontweight='bold')
    im = _heatmap(axes[0], raw, 'Observed median monthly rent', xtick_step=4, ylabel='Town')
    fig.colorbar(im, ax=axes[0], shrink=0.8, label='SGD per month')
    im = _heatmap(axes[1], residual, 'Mean residual after dwelling and town composition controls', 'RdBu_r', vmin=-limit, vmax=limit, xtick_step=4, xlabel='Approval month', ylabel='Town')
    fig.colorbar(im, ax=axes[1], shrink=0.8, label='SGD/month above or below composition fit')
    _save(fig, 'distribution_02_month_town', 'Cells require n >= 20. Controls: area, log area, flat type, flat model and town; all-train in-sample descriptive fit, no month control, no causal claim.', manifest)
    month_town.reset_index().to_csv(RESULTS / 'distribution_month_town.csv', index=False)
    (RESULTS / 'distribution_composition_fit.json').write_text(json.dumps(control_audit, indent=2), encoding='utf-8')
    lat = _first(frame, ['hdb_latitude', 'latitude'])
    lon = _first(frame, ['hdb_longitude', 'longitude'])
    if lat and lon:
        blocks = frame.groupby('address_key').agg(latitude=(lat, 'first'), longitude=(lon, 'first'), rent_median=('target', 'median'), area_median=('floor_area_sqm', 'median'), transactions=('target', 'size')).dropna(subset=['latitude', 'longitude'])
        blocks = blocks.loc[blocks['transactions'] >= 5]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.9), layout='constrained')
        fig.suptitle(f'Geographic rent hotspots and sample support\n{len(blocks):,} observed blocks with >= 5 transactions each | {date_range}', fontsize=17, fontweight='bold')
        for ax, column, title, unit, cmap in ((axes[0], 'rent_median', 'Median of block rent medians', 'SGD/month', 'viridis'), (axes[2], 'area_median', 'Median of block floor-area medians', 'Square metres', 'cividis')):
            image = ax.hexbin(blocks['longitude'], blocks['latitude'], C=blocks[column], gridsize=(35, 23), mincnt=3, reduce_C_function=np.median, cmap=cmap, linewidths=0.15, edgecolors='white')
            fig.colorbar(image, ax=ax, shrink=0.65, label=unit)
            ax.set_title(title)
        density = axes[1].hexbin(blocks['longitude'], blocks['latitude'], gridsize=(35, 23), mincnt=1, cmap='Blues', linewidths=0.15, edgecolors='white')
        axes[1].set_title('Observed block density')
        fig.colorbar(density, ax=axes[1], shrink=0.65, label='Distinct observed blocks per cell')
        for ax in axes:
            ax.set(xlabel='Longitude (degrees east)', ylabel='Latitude (degrees north)')
            ax.set_aspect(1 / np.cos(np.deg2rad(1.35)))
            ax.grid(alpha=0.18)
        _save(fig, 'distribution_03_geo_hotspots', 'One observed block = one vote. Rent/area cells require >= 3 blocks; no basemap. Density describes this sample, not population. Time and dwelling mix remain unadjusted.', manifest)
        blocks.reset_index().to_csv(RESULTS / 'distribution_block_hotspots.csv', index=False)
    primary = _first(frame, ['school_primary_nearest_km', 'school_level_primary_nearest_km'])
    secondary = _first(frame, ['school_secondary_nearest_km', 'school_level_secondary_s1_s5_nearest_km', 'school_level_secondary_s1_s4_nearest_km'])
    spatial_requested = [('target', 'Monthly rent'), ('floor_area_sqm', 'Area'), ('lease_age', 'Lease age'), ('building_age', 'Building age'), ('max_floor', 'Block max floor'), ('cbd_raffles_place_distance_km', 'CBD distance'), ('mrt_status_open_nearest_km', 'MRT distance'), (primary, 'Primary distance'), (secondary, 'Secondary distance'), ('mall_nearest_km', 'Mall distance'), ('hdb_latitude', 'Latitude'), ('hdb_longitude', 'Longitude')]
    spatial_cols = [(name, label) for name, label in spatial_requested if name and name in frame]
    spatial_corr = frame[[name for name, _ in spatial_cols]].rename(columns=dict(spatial_cols)).corr(method='spearman', min_periods=100)
    macro_requested = [('coe_a_premium_last_trailing_1m', 'COE A premium'), ('coe_b_bid_pressure_trailing_1m', 'COE B bid pressure'), ('stock_group_all_stock_return_1m', 'All-stock return'), ('stock_group_all_stock_portfolio_daily_vol_trailing_1m', 'All-stock volatility'), ('stock_group_banks_return_3m', 'Bank momentum 3m'), ('stock_group_property_return_3m', 'Property momentum 3m')]
    macro_cols = [(name, label) for name, label in macro_requested if name in frame]
    monthly = frame.groupby('month').agg(rent_median=('target', 'median'), adjusted_residual=('composition_residual', 'mean'))
    for name, label in macro_cols:
        monthly[label] = frame.groupby('month')[name].first()
    monthly = monthly.rename(columns={'rent_median': 'Monthly rent median', 'adjusted_residual': 'Composition residual'})
    monthly_corr = monthly.corr(method='spearman', min_periods=12)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9.7), layout='constrained')
    fig.suptitle('Which measured quantities move together?\nSpearman rank correlations; descriptive associations', fontsize=18, fontweight='bold')
    im = _heatmap(axes[0], spatial_corr, f'Dwelling and geography | {len(frame):,} transactions', 'RdBu_r', -1, 1, True, '.2f')
    im2 = _heatmap(axes[1], monthly_corr, f'Macro and rent | {len(monthly)} monthly observations', 'RdBu_r', -1, 1, True, '.2f')
    fig.colorbar(im2, ax=axes, shrink=0.65, label='Spearman rho')
    _save(fig, 'distribution_04_correlations', 'Macro inputs are already lagged by one month. Pairwise available observations; row repetition, correlated features and common trends do not establish causal effects.', manifest)
    spatial_corr.to_csv(RESULTS / 'distribution_spatial_spearman.csv')
    monthly_corr.to_csv(RESULTS / 'distribution_monthly_spearman.csv')
    monthly.to_csv(RESULTS / 'distribution_monthly_macro_inputs.csv')
    mrt = _first(frame, ['mrt_status_open_nearest_km', 'mrt_nearest_km'])
    facilities = [(mrt, 'Nearest snapshot-open MRT'), (primary, 'Nearest primary school'), (secondary, 'Nearest secondary school')]
    facilities = [(col, label) for col, label in facilities if col]
    boundaries = [0, 0.3, 0.5, 0.8, 1.2, 2, np.inf]
    distance_labels = ['< 0.3 km', '0.3-0.5 km', '0.5-0.8 km', '0.8-1.2 km', '1.2-2 km', '>= 2 km']
    distance_data = []
    if facilities:
        fig, axes = plt.subplots(1, len(facilities), figsize=(18, 6.8), layout='constrained', squeeze=False)
        fig.suptitle(f'Rent distribution by distance and flat type\n{scope}', fontsize=18, fontweight='bold')
        for ax, (col, label) in zip(axes[0], facilities):
            selected = frame[['flat_type', 'target', col]].copy()
            selected['distance_bin'] = pd.cut(selected[col], boundaries, labels=distance_labels, right=False)
            summary = selected.groupby(['distance_bin', 'flat_type'], observed=True)['target'].agg(['median', 'count'])
            n = summary['count'].unstack().reindex(index=distance_labels, columns=flat_order)
            values = summary['median'].unstack().reindex(index=distance_labels, columns=flat_order).where(n >= 50)
            image = _heatmap(ax, values, label, 'viridis', vmin=1500, vmax=4000, xlabel='Flat type', ylabel='Straight-line distance')
            for row in range(len(values)):
                for column in range(len(values.columns)):
                    value = values.iloc[row, column]
                    if pd.notna(value):
                        color = 'white' if image.norm(value) < 0.56 else '#17324b'
                        ax.text(column, row, f'{value:,.0f}\nn={n.iloc[row, column]:,.0f}', ha='center', va='center', fontsize=8, color=color)
            fig.colorbar(image, ax=ax, shrink=0.7, label='Median SGD/month')
            output = summary.reset_index()
            output['facility'] = label
            output['source_feature'] = col
            distance_data.append(output)
        _save(fig, 'distribution_05_distance_bins', 'Each cell requires >= 50 transactions; annotations show median and n. Same colour scale. Static supplied facilities; raw patterns retain age, town and time confounding.', manifest)
        pd.concat(distance_data, ignore_index=True).to_csv(RESULTS / 'distribution_distance_flat_bins.csv', index=False)
    hist, _, hist_lineage, hist_audit = construct_history(frame, frame.iloc[:0], None, frame['month'].max(), smoothing=20)
    coverage = pd.DataFrame(hist_audit['monthly_coverage'])
    coverage.to_csv(RESULTS / 'distribution_history_monthly_coverage.csv', index=False)
    groups = list(hist_audit['used_groups'])
    mean_columns = [hist_audit['group_feature_columns'][group][0] for group in groups]
    distributions = [hist[column].dropna().to_numpy() for column in mean_columns]
    short = {'address': 'Block address', 'town_flat_type': 'Town + flat type', 'street_flat_type': 'Street + flat type', 'postal_sector': 'Postal sector', 'nearest_snapshot_open_mrt': 'Nearest snapshot-open MRT', 'nearest_mrt_snapshot': 'Nearest snapshot MRT', 'nearest_mall': 'Nearest mall', 'nearest_school': 'Nearest school', 'nearest_primary_school': 'Nearest primary', 'nearest_secondary_school': 'Nearest secondary'}
    coverage_matrix = coverage.pivot(index='group', columns='month', values='seen_fraction').reindex(index=groups, columns=calendar)
    coverage_matrix.index = [short.get(group, _label(group)) for group in groups]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), layout='constrained', gridspec_kw={'width_ratios': [1, 1.35]})
    fig.suptitle(f'Past local-market encodings and history coverage\nEarlier months only | alpha = 20 | {scope}', fontsize=18, fontweight='bold')
    box = axes[0].boxplot(distributions, orientation='horizontal', tick_labels=[short.get(group, _label(group)) for group in groups], patch_artist=True, showfliers=False, widths=0.55, medianprops={'color': '#f5b942', 'linewidth': 2}, whiskerprops={'color': '#365b79'}, capprops={'color': '#365b79'})
    for patch in box['boxes']:
        patch.set(facecolor='#406f91', alpha=0.8, edgecolor='#365b79')
    axes[0].invert_yaxis()
    axes[0].set(title='Smoothed historical mean distributions', xlabel='Past monthly rent mean (SGD/month)')
    axes[0].grid(axis='x', alpha=0.2)
    image = _heatmap(axes[1], coverage_matrix * 100, 'Rows with prior observations in their group', 'viridis', vmin=0, vmax=100, xtick_step=5, xlabel='Training approval month')
    fig.colorbar(image, ax=axes[1], shrink=0.7, label='Previously seen group (%)')
    _save(fig, 'distribution_06_history_coverage', 'Training month T uses labels strictly before T. First-month mean is missing; unseen groups use global past mean. Entity encodings are local-market proxies, not amenity quality scores.', manifest)
    hist.describe().T.to_csv(RESULTS / 'distribution_history_feature_summary.csv')
    (RESULTS / 'distribution_history_audit.json').write_text(json.dumps(hist_audit, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(hist_lineage).to_csv(RESULTS / 'distribution_history_lineage.csv', index=False)
    notes = f'分布研究图表说明\n\n数据：{scope}。目标为月租金 MONTHLY_RENT，单位按新加坡住房租赁语境标示 SGD/月。所有图都是分布与关联描述，不作为已锁定主模型的调参或特征选择依据。\n\n01 town×flat_type：左图显示每个组合的交易加权租金中位数；少于30条交易的组合遮盖。右图显示真实样本数，颜色使用对数尺度。深色不一定表示房型品质更高，可能受到年份、面积、房龄和市场地段共同影响。\n\n02 month×town：上图为原始中位数；下图为控制面积、log面积、房型、flat model和town后的平均残差。控制模型只为描述，在完整训练集内拟合，未加入月份控制，因此仍保留总体租赁市场时间变化。残差为实际租金减构成模型拟合值，不是验证集预测误差，也不是因果效应；每格至少20条。\n\n03 经纬度热点：先把每栋楼的租金和面积分别压缩为中位数，再对地理栅格内楼栋取中位数，避免交易频繁的楼栋自动获得更大权重。楼栋至少5条交易；租金与面积栅格至少3栋楼。中间图是观察到的不同楼栋数，不是租约数、人口密度或全体楼栋密度。只有纯经纬度，没有外部底图。楼栋加权与租约加权会给出不同热点，这是权重问题，不代表其中一种错误。热点仍混合不同月份、房型、面积和房龄，不能断言经纬度造成租金改变。\n\n04 Spearman：左侧是关键房屋/空间变量的交易级秩相关；右侧每个月仅一条观察，避免把重复于数千租约的宏观变量当成数千个独立宏观样本。宏观输入按此前完整月滞后，月级最多51个样本；相关趋势与相关特征都可能制造表面关联。未列全1595个宏观候选，以便阅读。\n\n05 距离×房型：最近MRT、小学、中学的直线距离分箱，与房型交叉显示租金中位数和样本数，每格至少50条交易。使用统一色阶，不能把静态snapshot标为open解释成所有历史月份已开通，也不能把小学邻近理解为教育质量。年龄、town、时间与面积仍可能有残余混杂。\n\n06 历史target encoding：每条训练记录只读取更早月份标签，先计算整月特征，再更新该月标签。平滑为(组历史租金总和+20×全局过去均值)/(组历史样本数+20)。未见组退回全局过去均值，计数为0且覆盖标识为0；第一月没有过去标签，均值缺失。右图是各组已有历史观察的交易比例，左图是平滑后过去均值的分布。其数值反映本地历史租赁市场，并非学校/MRT/商场的独立质量。真正验证时历史固定在训练截止月，绝不更新held-out标签。\n\n输出：analysis/figures下有6组PNG和PDF；analysis/results/distribution_*保留图底表、历史覆盖、控制拟合说明与lineage。所有原CSV未修改。\n'
    (ANALYSIS / 'RESULTS_DISTRIBUTION_CN.txt').write_text(notes, encoding='utf-8')
    (RESULTS / 'distribution_figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'figures': len(manifest), 'rows': len(frame), 'groups': len(groups), 'scope': scope}, indent=2), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
