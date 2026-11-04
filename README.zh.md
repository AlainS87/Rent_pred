# CS5228 组屋月租

[English](README.md)

预测新加坡组屋的 `MONTHLY_RENT`。指标是 RMSE。先安装 `requirements.txt`，再在项目根目录运行下面的命令。随机种子是 `5228`。

列的公式在 [Docs/auxiliary_new_features.md](Docs/auxiliary_new_features.md)。命令说明在 [Docs/training_pipeline.md](Docs/training_pipeline.md)。取舍记录在 [Docs/design_tradeoffs.md](Docs/design_tradeoffs.md)。

```mermaid
flowchart LR
  features[Feature files]
  train[Train original columns]
  ablate[Ablate auxiliary columns]
  finalize[Larger budget]
  submit[Test predictions]
  features --> train --> ablate --> finalize --> submit
```

加在 `models/<model>/` 下的模型，要按顺序跑完 `train`、`ablate`、`finalize`、`submit` 才算做完。`--model` 是 `ridge`、`xgb`、`mlp` 或 `catboost`。

## 1. 特征准备

```text
python -m src.features.build_features
```

这个命令读取 `Data/train.csv`、`Data/test.csv`，以及 `Data/auxiliary/` 里的来源表。它按租金表的行序写出 `Data/auxiliary_features/train_features.csv` 和 `test_features.csv`，并同时写出 parquet。辅助表不在磁盘上和租金表拼在一起。训练时再按行序拼接。

辅助表有 94 列。`MONTHLY_RENT` 不在其中。六个组内平均租金只用批准月之前三个月的训练租金。

## 2. 初始模型

```text
python -m src.pipeline.run train --model <model> --features original
```

`original` 是 `train.csv` 里的十列：`RENT_APPROVAL_DATE`、`TOWN`、`BLOCK`、`STREET`、`FLAT_TYPE`、`FLAT_MODEL`、`FLOOR_AREA_SQM`、`FURNISHED`、`LEASE_COMMENCE_DATE` 和 `FEE`。这条命令拟合这十列，并写下第一份 `models/<model>/best/`。之后只有搜索期 RMSE 严格更低时，才会换掉这份快照。

### 2.1 怎么评估

`--val-months 3` 按批准月份把有标签的数据切开。

| 时段 | 月份 | 作用 |
|---|---|---|
| Fit | 到 2024-09 为止 | 拟合预处理器和权重 |
| Search | 2024-10 到 2024-12 | 停止树和 MLP，决定特征是否保留，并决定是否替换 `best/` |
| Confirm | 2025-01 到 2025-03 | 在搜索期停止之后记录一次 |

预处理器只用拟合期。搜索期 RMSE 只有严格更低时才替换 `best/`。确认期 RMSE 写在后面，不作为接受标准。带 `--max-rows` 的运行会在每个时段里抽步长样本，不能当作全表结果。

## 3. 消融

```text
python -m src.pipeline.run ablate --model <model>
```

消融从 `models/<model>/best` 里已经有的列开始。它逐个试探这份快照里还没有的辅助列，不从全部 94 列起步。

Ridge 和 MLP 只在搜索期 RMSE 比该轮集合 A 至少低 1% 时，把这一列记为有希望。XGBoost 和 CatBoost 只要搜索期 RMSE 严格更低即可。这些试探不改 `best/`。

集合 C 是当前列加上每一列有希望的辅助列，只训练一次。集合 C 自己的搜索期 RMSE 更低时，才替换 `best/`。某一轮没有有希望的列，或者集合 C 没有赢，循环就停。

## 4. 加大预算再训练

```text
python -m src.pipeline.run finalize --model <model>
```

这一阶段保留 `best/` 里已经选定的列，写出 `models/<model>/final/`。它不加列，也不删列。搜索期 RMSE 选择停下来的那一轮。

| 模型 | 预算 |
|---|---|
| Ridge | `alpha` 1 |
| MLP | 宽度 512、256、128；最多 200 轮；耐心 15 |
| XGBoost | 深度 8；学习率 0.05；最多 2000 棵树；耐心 50 |
| CatBoost | 深度 8；学习率 0.05；最多 2000 轮；耐心 50 |

## 5. 预测测试集

```text
python -m src.pipeline.run submit --model <model>
```

这一阶段把 `Data/test.csv` 和 `Data/auxiliary_features/test_features.csv` 按行序拼接，写出两个文件：

- `outputs/predictions/<model>_final.csv`，来自已保存的 `final/` 权重
- `outputs/predictions/<model>_refit.csv`，用全部有标签的月份再拟合一次，树的棵数、CatBoost 的轮数和 MLP 的轮数冻结为搜索期已经选定的大小，并关掉早停

表头是 `Id,Predicted`。编号从 `0` 到 `49999`，每个文件 50,000 行。这次全量再拟合没有留出月份上的租金分数。`predict` 在 `final/` 存在时读取那份权重。

`evolve` 和 `use-all` 是额外命令，不代替 train、ablate、finalize、submit 这条顺序。`use-all` 把全部原始列和辅助列拟合到 `models/<model>/use_all_features`，不替换 `best/` 或 `final/`。

## 当前进度

按日期写的记录是 [logs/progress_1029.md](logs/progress_1029.md) 和 [logs/progress_1031.md](logs/progress_1031.md)。搜索期 RMSE 用来停下一次训练。确认期 RMSE 只记录，不参与选择。

[mean_residual_model](mean_residual_model/README.md) 是单独的实验。它不替换 `models/<model>/best` 或 `final/`。

按搜索期停下的测试文件是 `outputs/predictions/mean_residual_multistage.csv`。它用三层 RNN 预测每平方米的市镇月均，再用每个市镇的 CatBoost 和高斯混合去拟合这个每平方米差值。每平方米均值、两个差值、面积，以及原始经纬度，分开送进全岛共用的 128-64 MLP。MLP 的输出就是月租。搜索期 RMSE 是 448.24，停在第 8 轮。确认期 RMSE 是 462.87。

`python -m mean_residual_model.refit.run` 把同一套模型在全部有标签月份上再拟合一次，规模沿用搜索期选定的大小，写到 `outputs/predictions/mean_residual_multistage_refit.csv`。这次没有搜索期或确认期 RMSE。150,000 行上的样本内训练 RMSE 是 458.84。

更早的实验仍然写出 `outputs/predictions/mean_residual_<model>.csv`。在那一组文件里，一套房子的预测租金是市镇月均加上残差。月均是岭回归（`alpha` 1），目标是每个 `TOWN`、每个批准月里 `MONTHLY_RENT` 的平均，拟合到 2024-09 为止的格子。输入是市镇、从 2021-01 起的月份序号、`coe_c_premium_lag1`、`bank_basket_adj_close_lag1` 和 `property_basket_adj_close_lag1`。加权格子 RMSE 在搜索期是 93.36，在确认期是 103.31。

Ridge、XGBoost、MLP 和 CatBoost 各自复制 `models/<model>/best/genome.json`，拟合 `MONTHLY_RENT` 减去这个月均。两者相加后的搜索期 RMSE 是 461.16、454.49、461.78 和 451.26。测试文件是 `outputs/predictions/mean_residual_<model>.csv`。这些权重拟合到 2024-09。2025-01 到 2025-03 没有再拟合进这些文件。

```text
python -m mean_residual_model.run
python -m mean_residual_model.run_by_town
```

第二条命令用同一套基因组，在每个市镇自己的行上分别拟合残差。Ridge、XGBoost 和 MLP 的搜索期 RMSE 高于全岛共用的那一个残差模型。CatBoost 按 26 个市镇分开后，搜索期 RMSE 是 450.50，全岛单模型是 451.26，确认期更高，所以 `mean_residual_catboost.csv` 仍用单模型。26 个市镇的预测在 `outputs/predictions/mean_residual_by_town_<model>.csv`。
