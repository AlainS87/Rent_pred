from experiments.local.bootstrap import Path, CODE_DIR, DATA_ROOT, SUPPORT_ROOT, ARTIFACT_ROOT, asset_path, asset_key, support_path

def main():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    p = asset_path(Path('external_release_audit_01'))
    d = pd.read_csv(p / 'release_aligned_monthly.csv')
    t = pd.to_datetime(d.origin_date)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True, layout='constrained')
    axs[0].plot(t, 100 * d.own_trailing6_monthly_log_growth, label='HDB 样本租金指数：过去6个月平均增速', color='#165AA7', lw=2)
    axs[0].step(t, 100 * np.log1p(d.private_rent_qoq_pct / 100) / 3, where='post', label='月末已公布的私人租金季度增速（月均）', color='#B85A12', lw=1.6)
    axs[0].set_ylabel('月均对数增长率（%）')
    axs[0].legend(loc='upper left', fontsize=8)
    axs[1].step(t, d.vacancy_pct, where='post', color='#18846D', lw=2)
    axs[1].set_ylabel('月末已公布私人住宅\n空置率（%）')
    axs[2].plot(t, d.sora_prior_month_pct, color='#7650A5', lw=2)
    axs[2].set_ylabel('上月末3个月\n复合 SORA（%）')
    for ax in axs:
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
        for dt in ['2022-09-30', '2023-05-31', '2024-09-30']:
            ax.axvline(pd.Timestamp(dt), color='#777777', alpha=0.5, ls=':', lw=1)
    axs[2].xaxis.set_major_locator(md.MonthLocator(interval=6))
    axs[2].xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
    fig.suptitle('按当时发布时间对齐：租金增长、空置率与利率', fontsize=15)
    fig.supxlabel('各点代表月末可用信息，截至原训练期 2024-09；虚线为两个历史预测起点及当前起点。\n租金指数为回顾性估计；图示关联不代表预测提升。', fontsize=9)
    fig.savefig(p / 'release_aligned_signals.png', dpi=160)
if __name__ == '__main__':
    import argparse as _cli_argparse
    _cli_argparse.ArgumentParser(description=__doc__).parse_args()
    main()
