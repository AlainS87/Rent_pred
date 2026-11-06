# CS5228 组屋月租

[English](README.md)

预测新加坡组屋的 `MONTHLY_RENT`。指标是 RMSE，单位是新元。本文件说明代码在哪个目录、用哪条命令训练。

`Docs/merged/` 里有两份报告。[report.tex](Docs/merged/report.tex)（[report.pdf](Docs/merged/report.pdf)）是总报告。[report_reduced.tex](Docs/merged/report_reduced.tex)（[report_reduced.pdf](Docs/merged/report_reduced.pdf)）是概述。总报告保留分节论证、图和附录。概述只保留问题、四套模型和分数。

这个仓库里有三套代码，Python 路径互不通用。

| 目录 | 工作目录 | 内容 |
|---|---|---|
| 仓库根目录 | 项目根 | 特征表、ridge / XGBoost / MLP / CatBoost 流水线，以及多阶段模型（Model 3） |
| [model4](model4/README.md) | `model4/` | 独立的预测项目，包含 Model 4 |
| [catboost_historical](catboost_historical/README.md) | `catboost_historical/` | 历史 CatBoost 线，包含 F1 提交 |

只在上表写明的目录里运行命令。`model4/` 自己有一份 `src/` 和 `models/`。在仓库根目录启动那些模块会导入错误的包。

## 数据和划分

有标签的批准记录从 2021 年 1 月到 2025 年 3 月，`Data/train.csv` 共 150,000 行。未标注的比赛文件是 `Data/test.csv`，50,000 行，表头 `Id,Predicted`，id 从 0 到 49999。辅助来源表在 `Data/auxiliary/`。

Model 3 按批准月切开有标签的月份：

| 名称 | 月份 | 作用 |
|---|---|---|
| Training set | 到 2024 年 9 月 | 拟合参数 |
| Validation set | 2024 年 10–12 月 | 唯一用来停止或选择模型的划分 |
| Test set | 2025 年 1–3 月 | 选择之后再打分，不参与选择 |
| Private test set | 有标签时段之后的未标注文件 | 比赛分数，不参与选择 |

Model 1 和 Model 2 使用别的有标签窗口。报告里仍把那些分数叫作 validation RMSE。它们不是 2024 年 10–12 月这个 validation set。Private-test RMSE 不是 validation RMSE，也不是 test RMSE。

仓库根目录、Model 3 和 Model 4 的默认种子是 `5228`。历史 CatBoost 线使用种子 42。带 `--max-rows` 的运行不能当作全表结果。

## 报告里的模型

| 报告中的名字 | 做法 | 代码位置 | 报告记下的分数 |
|---|---|---|---|
| Model 1 | 清洗后的租约字段上的 CatBoost，前面有镇和户型的均值 | 写在报告里。下面的根目录流水线是维护中的 CatBoost 入口，不是那次六个月基线的重放 | 2024 年 10 月到 2025 年 3 月的 validation RMSE 454.29，拟合到 2024 年 9 月 |
| Model 2，`add_mrt` | 加入辅助特征后的 CatBoost；`add_mrt` 在 development pooled RMSE 上是其中最好的配置 | 比较写在报告里。这个检出没有那套 1,878 列候选特征的构建程序 | 2023 年 11 月到 2025 年 3 月的 validation RMSE 499.53，拟合到 2023 年 10 月。Private-test RMSE 432.50 |
| Model 3 | 每个镇一个 RNN，预测每平方米租金的镇月均值；再分别用 CatBoost 和高斯混合预测单位缺口；最后一个共享 MLP | `mean_residual_model/` | Validation RMSE 448.24，第 8 轮。Test RMSE 462.87。更早的一次 private test 提交是 451.26 |
| Model 4 | 全岛一个 CatBoost，加上按房屋属性调整的租金时间指数 | `model4/experiments/local/` | 与 Model 3 相同月份上的 validation RMSE 444.47、test RMSE 450.89。Private-test RMSE 441.46，是最近一次提交，不是最好 |
| F1 | 带楼栋历史租金的 CatBoost，1600 轮，再加上向前的市场修正 | `catboost_historical/` | Private-test RMSE 432.50394 |

报告对照表里的最好 private-test RMSE 是 432.50，名字是 `add_mrt`。同一份报告后面的 F1 表把 1600 轮、带楼栋历史的提交记为 432.50394。这是公开发布的比赛分数。它们没有用来选择 Model 3 的轮数。Gold Miner (Group 12) 排名 21，共 8 次提交。最近一次提交 441.46 没有替换 432.50。

## 根目录：安装和特征

在项目根目录安装 `requirements.txt` 之后：

```text
python -m src.features.build_features
```

它读取 `Data/train.csv`、`Data/test.csv` 和 `Data/auxiliary/`，按原行序写出 `Data/auxiliary_features/train_features.csv` 和 `test_features.csv`，不排序。辅助表有 94 列。六个组均值只用之前三个批准月的有标签租金。其他辅助列不读 `MONTHLY_RENT`。公式在 [Docs/auxiliary_new_features.md](Docs/auxiliary_new_features.md)。

## 根目录：ridge、XGBoost、MLP、CatBoost

`--model` 是 `ridge`、`xgb`、`mlp` 或 `catboost`。`models/<model>/` 下的模型要按顺序跑完下面四条才算做完。`ablate` 只加入 `best/` 里还没有的辅助列，不从全部列开始。

```text
python -m src.pipeline.run train --model <model> --features original
python -m src.pipeline.run ablate --model <model>
python -m src.pipeline.run finalize --model <model>
python -m src.pipeline.run submit --model <model>
```

`train` 拟合十个原始列，写下第一份 `best/`。`ablate` 每次试一个尚未加入的辅助列。Ridge 和 MLP 只有在 validation RMSE 比该轮当前集合至少低 1% 时才把该列当作有希望。XGBoost 和 CatBoost 只要 validation RMSE 严格更低。这些试探不改 `best/`。当前列加上所有有希望的列，只有自己的 validation RMSE 更低时才替换 `best/`。某一轮没有有希望的列，或者合并后的拟合没有赢，就停止。Test RMSE 会记下，但不决定是否收下这一列。做法也写在报告附录里。

`finalize` 用更大的预算重训 `best/` 里已经留下的列，写出 `final/`，不加列也不减列。`submit` 按行序把 `Data/test.csv` 和辅助测试表接上，用 `final/` 写出 `outputs/predictions/<model>_final.csv`，再用停住的规模在全部有标签月份上重拟合，写出 `outputs/predictions/<model>_refit.csv`。

不替代上面顺序的其他命令：

```text
python -m src.pipeline.run test --model <model>
python -m src.pipeline.run predict --model <model> --out outputs/predictions/<model>.csv
python -m src.pipeline.run evolve --model <model> --features original,blocks --generations 4
python -m src.pipeline.run use-all --model <model>
```

`use-all` 把全部原始列和辅助列拟合进 `models/<model>/use_all_features`，不替换 `best/` 或 `final/`。

## 根目录：镇月均值加残差

这是更早的实验，不替换 `models/<model>/best` 或 `final/`。

```text
python -m mean_residual_model.run
python -m mean_residual_model.run_by_town
```

第一条先用 ridge 预测镇月均值，再让每个算法拟合 `MONTHLY_RENT` 减去这个均值。测试文件是 `outputs/predictions/mean_residual_<model>.csv`。第二条在每个镇里分开拟合同一套结构。细节在 [mean_residual_model/README.md](mean_residual_model/README.md)。

## 根目录：Model 3，多阶段模型

区域均值是每个镇一个三层 RNN，预测每平方米租金。然后每个镇有一个 CatBoost 和一个高斯混合，都预测单位缺口，两者互不看见。一个共享 MLP 接收 RNN 均值、两个缺口、面积和该套房子的其他特征，输出月租。Validation set 停止 CatBoost 的轮数和 MLP 的 epoch。

```text
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py"
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --residuals-only
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --mlp-only
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --frozen-catboost
python "mean_residual_model/multi_stage_model(lstm+catboost+mlp)/run.py" --predict
```

`--residuals-only` 拟合镇内 CatBoost 和混合模型，不拟合 MLP。`--mlp-only` 用已保存的缺口训练 MLP。`--frozen-catboost` 读回每个镇的 CatBoost，再重拟合混合模型和 MLP。`--predict` 用已保存的权重写测试文件，不重新拟合。按 validation set 停下来的测试文件是 `outputs/predictions/mean_residual_multistage.csv`。

后面的检查不覆盖这些权重，并写出单独的文件：

```text
python -m mean_residual_model.refit.run
python -u "mean_residual_model/aligned_preprocess/run.py"
python -u "mean_residual_model/catboost_head/run.py"
```

`refit` 在到 2025 年 3 月的全部有标签月份上再拟合同一套结构，规模冻结为已经停住的大小，写出 `outputs/predictions/mean_residual_multistage_refit.csv`。这次没有 validation RMSE 或 test RMSE。`aligned_preprocess` 用楼栋编号、日历字段和房龄重训镇内 CatBoost、混合模型和 MLP，不替换按 validation set 停住的文件。`catboost_head` 冻住 RNN、镇内 CatBoost 和混合模型，用一棵 CatBoost 换掉 MLP。它的测试文件是 `outputs/predictions/mean_residual_catboost_head.csv`。

栈使用的 RNN 权重用这条命令训练：

```text
python -m mean_residual_model.reginal_mean_model.rnn.train_l3
```

它会给留出月份打分，并写到 `mean_residual_model/reginal_mean_model/rnn/h16_l3/`。`refit` 在 `mean_residual_model/refit/rnn/` 另存一份，不覆盖那个目录。

## model4：Model 4 和预测项目

在 `model4/` 里安装和运行，不要在仓库根目录运行。它自己的说明是 [model4/README.md](model4/README.md)。Python 3.11 或 3.12：

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[models,test]"
```

Model 4 的全局 CatBoost 和租金时间指数是固定划分脚本。每条都是先 `prepare` 再 `run`：

```text
python -m experiments.local.global_monthly_catboost prepare
python -m experiments.local.global_monthly_catboost run
python -m experiments.local.temporal_index_trial prepare
python -m experiments.local.temporal_index_trial run
```

时间指数脚本就是报告里的 Model 4：深度 8，学习率 0.05，最多 1000 轮，种子 5228，在 2024 年 10–12 月上以 50 轮耐心早停。它的 private test 文件是 441.46 那次提交，没有替换 432.50。

同一目录里的其他入口，都可以加 `--help`：

```text
python -m src.features.build_features --help
python -m src.pipeline.run --help
python -m src.backtest.run --help
python -m src.research_round2.run --help
python -m src.research_round3.run --help
python -m src.research_round4.run --help
python -m experiments.local.bootstrap prepare-features
python -m experiments.rolling_local validate --config experiments/rolling_local/example_config.json
python -m experiments.rolling_local run --config experiments/rolling_local/example_config.json --output artifacts/rolling-local-example
python -m analysis.external_signals --help
```

`model4/src/pipeline/run.py` 不是仓库根目录的 `src/pipeline/run.py`。这套代码的原始输入默认在 `model4/data/raw/`，和根目录的 `Data/` 不是同一个位置。

## catboost_historical：F1

在 `catboost_historical/` 里安装和运行。它自己的说明是 [catboost_historical/README.md](catboost_historical/README.md)。它读 `../Data`，不导入根目录的 `src/` 或 `models/`。

```text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m hdb_rent.final validate --data-dir ../Data --iterations 1600 --threads 4 --fold-preset strict --output-dir outputs/f1_validation
python -m hdb_rent.final train --data-dir ../Data --iterations 1600 --threads 4 --model-dir outputs/f1
python -m hdb_rent.final predict --data-dir ../Data --model-dir outputs/f1 --output outputs/f1/submission_F1_1600.csv
python -m hdb_rent.final submit --data-dir ../Data --iterations 1600 --threads 4 --model-dir outputs/f1 --output outputs/f1/submission_F1_1600.csv
```

`validate` 给按时间切开的折打分，不写提交文件。`train` 拟合并保存模型。`predict` 用已保存的模型预测。`submit` 先训练再预测。F1 是 1600 轮、深度 6、学习率 0.055、种子 42，带楼栋历史特征和衰减的市场修正。报告里的 Kaggle 表给这次提交的 private-test RMSE 是 432.50394。

历史变体和作图仍在 `catboost_historical/`：

```text
python -m experiments.historical --data-dir ../Data --mode validate --fold-preset strict --variants F0,F1,F2,F3,F4 --iterations 1600 --threads 4 --output-dir outputs/history_strict
python -m experiments.temporal --data-dir ../Data --mode validate --fold-preset strict --iterations 800,1200,1600,2000 --threads 4 --output-dir outputs/controlled_sweep
python -m experiments.baseline --data-dir ../Data --mode both --iterations 180 --depth 6 --threads 4 --output-dir outputs/baseline
python -m unittest discover -s tests -v
```

## 报告

在 `Docs/merged/` 里把每份源文件编译两次：

```text
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report_reduced.tex
```

`report.pdf` 是总报告。`report_reduced.pdf` 是概述。`main` 上还有一份更早的总报告拷贝：`Docs/report/merged/report.tex`。
