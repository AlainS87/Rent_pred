"""One descriptive figure for original dwelling attributes; no model fitting."""
from __future__ import annotations
from runtime import DATA_DIR, CODE_DIR, ANALYSIS
import json
import runtime
from runtime import ANALYSIS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    figures, results = (ANALYSIS / 'figures', ANALYSIS / 'results')
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    all_features = pd.read_pickle(ANALYSIS / 'features' / 'train_features.pkl')
    frame = all_features[['month', 'target', 'floor_area_sqm', 'lease_age', 'flat_type', 'flat_model']].copy()
    del all_features
    flat_order = [name for name in ('1 room', '2 room', '3 room', '4 room', '5 room', 'executive') if name in frame.flat_type.unique()]
    definitions = [('floor_area_sqm', [0, 45, 60, 75, 90, 105, 120, 135, 150, 225, np.inf], ['<45', '45-59', '60-74', '75-89', '90-104', '105-119', '120-134', '135-149', '150-224', '>=225'], 'Floor area within flat type', 'Floor-area bin (square metres)'), ('lease_age', [0, 10, 20, 30, 40, 50, 60, np.inf], ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '>=60'], 'Lease age within flat type', 'Lease-age bin (years at approval)')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': '#132a41', 'axes.labelcolor': '#304a60', 'axes.edgecolor': '#d6dfe8', 'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(20, 11), layout='constrained', gridspec_kw={'width_ratios': [1, 1, 1.25]})
    fig.suptitle(f'Original dwelling attributes and observed rent distributions\n{len(frame):,} training transactions | {frame.month.min()} to {frame.month.max()}', fontsize=19, fontweight='bold')
    tables = []
    image = None
    for ax, (feature, boundaries, labels, title, ylabel) in zip(axes[:2], definitions):
        selected = frame[['flat_type', 'target', feature]].copy()
        selected['bin'] = pd.cut(selected[feature], boundaries, labels=labels, right=False)
        table = selected.groupby(['bin', 'flat_type'], observed=True).target.agg(rent_median='median', transactions='size').reset_index()
        table.insert(0, 'feature', feature)
        table['display_supported'] = table.transactions >= 50
        tables.append(table)
        median = table.pivot(index='bin', columns='flat_type', values='rent_median').reindex(index=labels, columns=flat_order)
        count = table.pivot(index='bin', columns='flat_type', values='transactions').reindex(index=labels, columns=flat_order)
        values = median.where(count >= 50)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('#eef2f6')
        image = ax.imshow(np.ma.masked_invalid(values.to_numpy()), aspect='auto', cmap=cmap, vmin=1500, vmax=5000)
        ax.set_xticks(range(len(flat_order)), [value.title() for value in flat_order], rotation=45, ha='right')
        ax.set_yticks(range(len(labels)), labels)
        ax.set(title=title, ylabel=ylabel, xlabel='Flat type')
        ax.tick_params(length=0)
        for row in range(len(labels)):
            for col in range(len(flat_order)):
                value = values.iloc[row, col]
                if pd.notna(value):
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    color = 'white' if luminance < 0.52 else '#17324b'
                    ax.text(col, row, f'{value:,.0f}\nn={count.iloc[row, col]:,.0f}', ha='center', va='center', fontsize=7.7, color=color)
    if image is not None:
        fig.colorbar(image, ax=list(axes[:2]), shrink=0.7, label='Median monthly rent (SGD/month)')
    model = frame.groupby('flat_model').target.agg(rent_median='median', transactions='size').sort_values('transactions')
    model['display_supported'] = model.transactions >= 50
    y = np.arange(len(model))
    display = model.rent_median.where(model.display_supported, 0)
    axes[2].barh(y, display, height=0.6, color='#406f91', alpha=0.88)
    axes[2].set_yticks(y, [name.title() for name in model.index], fontsize=8.5)
    axes[2].set(title='Flat-model medians, ordered by sample size', xlabel='Monthly rent (SGD/month)')
    axes[2].set_xlim(0, max(4500, model.loc[model.display_supported, 'rent_median'].max() * 1.45))
    axes[2].grid(axis='x', alpha=0.15)
    axes[2].set_axisbelow(True)
    for i, row in enumerate(model.itertuples()):
        if row.display_supported:
            text = f'{row.rent_median:,.0f} | n={row.transactions:,}'
            axes[2].text(row.rent_median + 40, i, text, va='center', fontsize=8, color='#27475f')
        else:
            axes[2].text(65, i, f'n={row.transactions:,}: median masked', va='center', fontsize=8, color='#6d7f8f')
    fig.text(0.01, -0.018, 'One transaction = one vote. Cells/bars require n >= 50. Grey cells and masked bars have insufficient support. Raw medians retain month, location, model and dwelling-composition confounding; no causal or adjusted effect is claimed.', ha='left', va='top', fontsize=9, color='#4c6279')
    for extension in ('png', 'pdf'):
        fig.savefig(figures / f'distribution_07_original_dwelling.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    pd.concat(tables, ignore_index=True).to_csv(results / 'distribution_original_dwelling_bins.csv', index=False)
    model.reset_index().to_csv(results / 'distribution_original_dwelling_models.csv', index=False)
    audit = {'scope': 'All labelled training transactions, after fixed experiment configuration lock', 'rows': len(frame), 'first_month': frame.month.min(), 'last_month': frame.month.max(), 'minimum_display_transactions': 50, 'row_weighting': 'One transaction, one vote', 'control_adjustment': 'None', 'predictive_model_fitted': False, 'causal_interpretation': False, 'flat_models': len(model), 'models_with_display_support': int(model.display_supported.sum()), 'flat_model_ordering': 'By transaction count, including masked groups; no ranking by hidden small-sample medians', 'binning_policy': 'Fixed, left-closed intervals; table retains unsupported cells and original sample counts'}
    (results / 'distribution_original_dwelling_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    note = '原始住宅属性分布补图说明\n\n图：analysis/figures/distribution_07_original_dwelling.png（另有PDF）。使用全部150,000条训练交易，只作配置锁定后的描述，不拟合预测模型或选择变量。\n\n左图：面积分箱×房型的月租金中位数及交易数。同一房型内面积变化可以观察到租金分布变化，但房型标签不是实测卧室数，不同面积格也可能来自不同town、年份或flat model。\n\n中图：审批年份减LEASE_COMMENCE_DATE得到lease age，再按固定年龄段与房型交叉。这里是租约起始年龄，并非已验证的装修状态或实际住宅单元状态。年龄与地区、建筑年份、面积、模型和月份相关，不能用原始中位数宣称老房必然便宜或某个年龄有独立溢价。\n\n右图：全部观察到的flat model，展示月租中位数及样本数；少于50条的模型只显示样本数、遮盖中位数，避免把单笔/极小样本当成稳定价格。不同模型拥有不同房型、面积、地区和建造时期，因此这不是模型质量排名，也不是控制其他变量后的模型系数。\n\n所有图一条租约一票；至少50条才显示中位数。完整底表仍保留低样本组合，可核查灰格与遮盖的原因。数据表：results/distribution_original_dwelling_bins.csv与distribution_original_dwelling_models.csv；审计：distribution_original_dwelling_audit.json。\n'
    (ANALYSIS / 'RESULTS_ORIGINAL_DWELLING_CN.txt').write_text(note, encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
