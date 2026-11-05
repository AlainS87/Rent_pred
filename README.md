# Singapore Rental Forecasting

用于新加坡住宅**月租金（SGD）预测**的研究项目，包含特征工程、固定划分实验、滚动回测和外部市场信号分析。

本文件是项目唯一的 README，统一说明代码结构、数据、配置、运行和历史结果。原有模型定义、时间边界、调参规则和评价口径保留；整理代码不代表重新复现了历史分数。

## 目录

- [项目结构](#layout)
- [安装与快速检查](#setup)
- [数据与特征缓存](#data)
- [配置和运行入口](#workflows)
- [固定划分实验及依赖](#local)
- [三起点滚动实验](#rolling)
- [早期研究和历史数据工具](#other-studies)
- [历史结果和评价边界](#results)
- [测试与持续集成](#testing)
- [贡献、来源和许可](#contributing)

<a id="layout"></a>
## 项目结构

| 路径 | 作用 |
|---|---|
| `src/features/` | 核心 83 列特征及其公式 |
| `src/pipeline/` | 单模型预处理、训练、搜索、冻结配置和推理 |
| `src/backtest/` | 第一轮滚动回测、时间隔离、评分、进程和断点控制 |
| `src/research_round2/`、`research_round3/`、`research_round4/` | 后续研究轮次，沿用各自协议 |
| `src/experiments/`、`src/viz/` | 核心流程的可选诊断和绘图 |
| `src/paths.py` | 源码资源、工作目录及两套特征缓存的路径定义 |
| `models/` | Ridge、XGBoost、CatBoost、MLP 实现及 `default_genome.json`，不存放权重 |
| `mean_residual_model/` | 早期 town-month 均值 Ridge + 残差实验 |
| `experiments/local/` | 固定划分研究及其独立的内部依赖 `_support/` |
| `experiments/rolling_local/` | 三起点 Global CatBoost + 租金时间指数 |
| `experiments/early_analysis/` | 早期空间、PCA、历史租金编码和敏感性研究 |
| `experiments/historical_data/` | 公开历史数据及发布日期核查工具 |
| `analysis/` | 描述性外部信号分析，不训练预测模型 |
| `configs/` | 随安装包分发的研究协议和编码参数 |
| `tests/` | 时间契约、断点、评分、进程及结构回归测试 |
| `results/` | 历史聚合指标、表格和图，不是新的运行输出 |
| `data/raw/` | 本地原始输入，Git 忽略 |
| `artifacts/` | 生成特征、权重、预测、缓存和运行日志，Git 忽略 |
| `docs/evaluation.md` | 更详细的评价与信息可用性说明 |
| `scripts/` | 仓库校验、历史结果查看和 wheel 安装验证 |

`src` 目前是实际 Python 包名，并非传统的 `src/<package>/` 布局。内部实验使用 `experiments.local._support...` 完整包名，不向模块搜索路径注入同名 `src`、`models` 或 `mean_residual_model`。这些历史实现不能因为名称相似就直接互换。

<a id="setup"></a>
## 安装与快速检查

需要 Python 3.11 或 3.12。在仓库根目录执行：

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test]"
```

按实际工作安装模型后端：

```sh
python -m pip install catboost==1.2.8
python -m pip install xgboost
# 如需神经模型，选择适合当前 CPU/CUDA 的 PyTorch 2.6.0 发行版。
# 一次安装全部模型后端也可使用：
python -m pip install -e ".[models,test]"
```

基础依赖由 [pyproject.toml](pyproject.toml) 管理；[requirements.txt](requirements.txt) 指向项目自身。
[requirements/core-research.txt](requirements/core-research.txt) 记录核心导入和测试环境；[requirements/local-cpu.txt](requirements/local-cpu.txt) 记录历史三起点 CPU 实验环境。它们是两套独立的直接依赖约束，不是跨平台完整锁文件；使用 `pip -c` 时只选一套，不混用。

只读检查和帮助：

```sh
python scripts/show_results.py
python scripts/validate_repository.py
python -m experiments.rolling_local --help
python -m analysis.external_signals --help
python -m experiments.local.bootstrap --help
```

结果查看和仓库校验只需标准库；分析命令的帮助需要已声明的数值依赖。帮助命令不启动训练。统一用 `python -m <包名>` 启动研究模块；安装后也提供 `rental-local-backtest`、`rental-signal-analysis` 两个命令。

普通 wheel 安装会包含 `configs/` 协议、模型默认 JSON、本地协议和公开来源目录。配置从安装包定位，不依赖当前目录。默认数据和输出属于工作目录，不写入 `site-packages`。早期核心回测编排器还会校验源码、`pyproject.toml` 和历史资产；执行这些完整冻结流程应使用源码 checkout 和 editable 安装，不能仅凭 wheel 或历史分数替代其运行证据。

<a id="data"></a>
## 数据与特征缓存

### 原始输入

原始数据不随源码或 wheel 分发。默认布局：

```text
data/raw/
  train.csv
  query.csv
  auxiliary/
    sg-hdb-block.csv
    sg-mrt-stations.csv
    sg-schools.csv
    sg-shopping-malls.csv
    sg-coe-prices.csv
    sg-stock-prices.csv
```

`train.csv` 包含标签；`query.csv` 是无标签的未来房屋记录，没有固定行数。未来推理流程可能需要后者，本地特征准备命令要求两者齐全。

| 原始列 | 含义 |
|---|---|
| `RENT_APPROVAL_DATE` | 月份，`YYYY-MM` |
| `TOWN`、`BLOCK`、`STREET` | 地址；`BLOCK` 按字符串读取，保留字母和前导字符 |
| `FLAT_TYPE`、`FLAT_MODEL` | 房型、户型类别 |
| `FLOOR_AREA_SQM` | 正数面积，平方米 |
| `FURNISHED` | 装修/家具类别 |
| `LEASE_COMMENCE_DATE` | 租赁起始年份 |
| `FEE` | 数值费用字段 |
| `MONTHLY_RENT` | 月租金 SGD，仅有标签数据包含 |

辅助表分别提供楼栋坐标与属性、地铁、学校、商场、COE 竞拍和股票时间序列。公开的 [data/local_data_manifest.json](data/local_data_manifest.json) 记录文件级来源和哈希，不含逐条租赁记录。分发原始输入前需确认来源及使用许可；历史股票表未记录数据提供商和当时的价格调整版本。

保留历史协议的原始行顺序和重复记录。按行拼接的特征表必须与原始数据的行数、顺序完全一致；不得通过去重或修改标签后宣称复现原实验。

### 两套缓存共用原始数据，分别生成

| 特征版本 | 构建入口 | 默认缓存 | 使用者 |
|---|---|---|---|
| 核心 83 列 | `python -m src.features.build_features` | `artifacts/features/core83/`，CSV 和 Parquet | 核心单模型、早期均值/残差流程 |
| 本地 94 列 | `python -m experiments.local.bootstrap prepare-features` | `artifacts/features/local94/`，CSV | 固定划分实验和三起点滚动实验 |

核心文件名为 `train_features.*`、`test_features.*`，其中 `test` 对应无标签的 `query.csv`；本地文件名为 `train_features.csv`、`query_features.csv`。历史本地代码的 `test_features.csv` 和 `Data/auxiliary_features/` 写法会显式解析到本地 94 列缓存。它们是兼容路径标识，不是另一份原始输入。

**两套缓存不能互相替换。** 保留不同公式、列目录和过去窗口构造。核心回测另有 `artifacts/backtest/cache/` 的无标签缓存，不等同于上述完整特征表。早期分析的独立特征保存在 `artifacts/early_analysis/`。

已有缓存不会自动移动或重新计算。旧核心缓存需放到 `artifacts/features/core83/`；旧本地缓存可放到 `artifacts/features/local94/`，也可用 `RENTAL_LOCAL_FEATURES_DIR` 指向原有目录。不要混合两种版本或覆盖历史证据。

```sh
python -m experiments.local.bootstrap check-inputs
python -m experiments.local.bootstrap prepare-features
```

前者只检查文件及表头，后者显式构建 94 列，拒绝替换已经存在的目标缓存目录。两者均不训练预测模型。

### 工作路径

| 环境变量 | 用途 |
|---|---|
| `RENTAL_PROJECT_DIR` | 工作目录；源码 checkout 默认仓库根目录，wheel 默认启动时的当前目录 |
| `RENTAL_DATA_DIR` | 本地实验、特征构建及分析的原始输入根目录，默认 `<工作目录>/data/raw` |
| `RENTAL_ARTIFACTS_DIR` | 上述流程的产物根目录，默认 `<工作目录>/artifacts` |
| `RENTAL_LOCAL_FEATURES_DIR` | 94 列缓存目录，默认 `<产物根目录>/features/local94` |
| `RENTAL_BASELINE_DIR` | 本地实验前置基线资产，默认 `<产物根目录>/local/baseline` |

核心历史回测编排器仍按其冻结工作目录中的 `data/raw/`、`artifacts/backtest/` 和源码契约运行；不要用环境变量重定向来绕过已固定的输入哈希。三起点滚动命令使用 JSON 中的显式输入路径，路径相对配置文件解析，输出路径相对当前目录解析。

源码校验只跳过 `data/raw/` 和 `artifacts/` 中确认为 Git 忽略的文件；已跟踪或未忽略的输入/产物会报错。其他目录中的模型和运行文件仍会报错。没有 Git 时只识别 `.gitignore` 的明确目录规则，并报告该限制。校验不读取原始记录内容，数据契约与来源核查另行进行。

<a id="workflows"></a>
## 配置和运行入口

| 配置 | 对应入口 | 前置条件及输出 |
|---|---|---|
| [configs/backtest/protocol.json](configs/backtest/protocol.json) | `python -m src.backtest.run --help`；`src.backtest.development`、`src.backtest.supervisor` | 第一轮时间隔离缓存、冻结 manifest、选择和断点；`artifacts/backtest/` |
| [configs/research_round2/protocol.json](configs/research_round2/protocol.json) | `python -m src.research_round2.run --help` | B0/B1/B2 及仅使用过去标签的价格快照；依赖声明的第一轮缓存及证据 |
| [configs/research_round3/protocol.json](configs/research_round3/protocol.json) | `python -m src.research_round3.run --help` | C1/C2/C3 与冻结 B0 对照；需要第二轮基线资产 |
| [configs/research_round4/protocol.json](configs/research_round4/protocol.json) | `python -m src.research_round4.run --help` | B0/C3，四个新起点、两个复用历史起点；需要指定前序证据 |
| [configs/encodings/bin.json](configs/encodings/bin.json)、[target.json](configs/encodings/target.json) | 编码参数定义，使用前查看相应预处理实现 | 不单独启动训练 |
| [experiments/local/protocol.json](experiments/local/protocol.json) | `python -m experiments.local.prepare`、`python -m experiments.local.run --help` | 时间 stacking 协议及其前置模型；`artifacts/local/` |
| [experiments/rolling_local/example_config.json](experiments/rolling_local/example_config.json) | `python -m experiments.rolling_local validate --config ...` / `run --config ... --output ...` | 94 列缓存；新的专用输出目录 |
| `models/<模型>/default_genome.json` | `python -m src.pipeline.run --help` | 模型特征、预处理及超参数；`artifacts/models/` |
| [experiments/historical_data/sources.json](experiments/historical_data/sources.json) | 历史数据获取工具 | 公开 URL、日期目录；下载内容存于 `artifacts/historical_data/` |

这些 JSON 保留各自的特征、时间起点、内层划分、容量和选择规则，不是可以随意混用的全局配置。第四轮保留的正常训练预算为 16 次。配置中的历史 manifest/checkpoint 哈希是证据标识，相关权重和回执未打包；缺少前序资产时不能删掉校验或换个模型继续运行。

核心单模型支持冻结 genome 拟合、时间划分、特征探索、验证和通用推理。输出为可变行数的 `row_id,prediction`。早期 `mean_residual_model` 用 town、月份及三个滞后宏观水平拟合 town-month 平均月租金，再拟合四种残差模型；按 town 的变体分别训练各地区。它需要 `artifacts/models/` 中已选择的配置和资产，产物在 `artifacts/mean_residual/`。

```sh
python -m src.features.build_features --help
python -m src.pipeline.run --help
python -m mean_residual_model.run --help
python -m mean_residual_model.run_by_town --help
```

以上只显示帮助。去掉帮助或使用训练子命令可能读数据、拟合模型、启动子进程及写入产物。

<a id="local"></a>
## 固定划分实验及依赖

`experiments/local/_support/` 保留本地实验实际依赖的 94 列特征、预处理、CatBoost、town RNN 以及条件混合/MLP 函数。它不是核心源码的可互换副本。模型栈的动态实例保留各实验独立的可变状态，不修改全局导入搜索路径。

| 方法 | `experiments/local/` 中的文件 |
|---|---|
| 区域 RNN、CatBoost、条件混合与 MLP 基线 | `baseline_probe.py`、`train_town_rnn.py`、`common.py` |
| 面积/楼龄单特征消融 | `feature_ablation.py` |
| 冻结 MLP 的面积残差模型 | `confirm_frozen_mlp_area.py` |
| 面积平方残差加权 | `weighted_residual.py` |
| 历史邻域租金 | `spatial_history_candidate.py`、`spatial_history_features.py` |
| 全局月租金 CatBoost、类别统计 | `global_monthly_catboost.py`、`global_monthly_ctr8.py` |
| 全局/区域等权混合、完整地址 | `confirm_global_original_blend.py`、`global_full_address.py` |
| 区域 RNN 水平上的全局残差 | `global_rnn_residual.py` |
| 住宅内租金时间指数、空置率修正 | `temporal_index.py`、`temporal_index_trial.py`、`vacancy_index_trial.py` |
| 三种时间 stacking | `prepare.py`、`run.py`、`protocol.json` |
| 误差、时间和来源审计 | `diagnose_*.py`、`analyze_*.py`、`audit_*.py`、`external_release_audit.py` |
| 公开记录获取、已存资产核查 | `fetch_hdb_search_records.py`、`verify_*.py` |

前置依赖顺序大致为：区域基线 → 面积/楼龄和冻结 MLP → 全局直接模型 → 类别统计 → 完整地址 → 时间指数或 RNN 残差 → 可选空置率实验。各入口仍校验自己实际要求的前序模型、冻结输入及预算。它们不是在空目录中可以任意跳步执行的独立训练脚本。

例如使用 `python -m experiments.local.global_monthly_catboost --help` 查看入口。`baseline_probe.py` 从已有完整训练堆栈输入拟合最终 MLP，仅评分有标签时段；没有固定行数的外部交付分支。时间指数与 stacking 是不同实验；失败、未评分和仅有历史记录的方法不能因为源码存在就视为验证通过。

需要 NumPy、pandas、SciPy、scikit-learn、CatBoost、PyTorch、joblib、threadpoolctl；绘图需 matplotlib，获取数据可能需 requests。不自动安装依赖或下载代码。长任务需明确的资源及截止时间控制，不自动恢复历史训练。

<a id="rolling"></a>
## 三起点滚动实验

唯一模型为 `global_catboost_index`：26 个选定输入的 Global CatBoost，加上仅使用历史的住宅内租金指数。

```sh
python -m experiments.rolling_local validate --config experiments/rolling_local/example_config.json
python -m experiments.rolling_local run --config experiments/rolling_local/example_config.json --output artifacts/rolling-local-example
```

`validate` 只验证配置，不读训练数据或拟合模型。`run` 显式开始新计算，输出目录必须不存在或为空；没有自动恢复、下载、GPU 回退或额外模型搜索。wheel 用户应复制示例配置到自己的工作目录，修改其中相对输入路径后使用。

保留的计算规则：

- 指数采用住宅组固定效应和月份效应的 log 月租金 OLS；只有至少在两个历史月份出现的住宅组识别月份效应。用最近六个月斜率并以 0.9 阻尼外推。
- CatBoost：深度 8、学习率 0.05、最多 1000 棵树、种子 5228、8 线程；保留原生类别和数值中位数填补。
- 学习目标为 `y/I(month)`，权重为 `I(month)^2 / mean(I(month)^2)`，内层验证也使用相应权重；转换后的原始租金 RMSE 必须在 `1e-3` 内一致。
- 对起点 T，用截至 T-3 的历史预测最后三个月以选择树数，再截至 T 从头拟合指数、预处理和 CatBoost，关闭 early stopping。一次预测其后 16 个月，不从未来真实租金更新。
- 户型名称数值排名在每次允许历史内拟合，按月租金中位数、均值、标准化名称排序，未知类别用已知排名中位数。
- 宏观字段冻结在最后可用的 lag1 值；这些字段不在本模型选定的 26 列中，冻结属于数据边界。

示例起点为 2023-05、2022-03、2022-10。预算为 180 分钟，其中 10 分钟留给报告。一个受监督 worker 顺序执行；计算截止时停止自己的 worker，仅报告已保存的完整起点，不重启训练。每个起点拟合两次指数和两次 CatBoost，三起点各六次，无隐藏拟合。

每个起点保存容量、预处理/指数统计、权重、保留原行身份的预测和评分。指标包括 RMSE、MAE、平均误差、总体相对偏差和高估比例；报告覆盖 h1–16、h7–16、固定区间、各月、town、房型和已见/未见地址。未完成起点明确排除，重叠预测窗口不是独立市场周期。

所需依赖为 NumPy、pandas、SciPy、scikit-learn、CatBoost、joblib、PyArrow、psutil、threadpoolctl；帮助和配置校验只依赖标准库。静态设施表的历史开放日期/发布时点未全面验证，保留全部原始记录及异常大误差，不承诺不同环境逐位一致。

<a id="other-studies"></a>
## 早期研究和历史数据工具

`experiments/early_analysis/` 是独立研究，其特征及划分不同于后来的固定划分实验。包含空间/宏观特征、开发期条件关联、股票 PCA、Native Tree/Ridge 比较、严格历史租金编码、空间/趋势/协方差敏感性及描述性分布。

```sh
python experiments/early_analysis/run_all.py --refit
```

该命令重建特征、重新拟合锁定研究并运行诊断；不加 `--refit` 则只用已有资产运行诊断。三个开发起点为 2022-04、2022-10、2023-04，各预测六个月；2023-10 起点预测 17 个月作为确认。截止时冻结宏观变量和使用提供的月度滞后变量是两个不同情景。`verify_artifacts.py` 核对用户提供的已有资产，不因打包而产生新分数。依赖 NumPy、pandas、SciPy、scikit-learn、CatBoost、joblib 和 matplotlib。

`experiments/historical_data/` 检查历史租金中位数、私人住宅存量/空置率和公开发布文件的可用性：

- `fetch_public_history.py` 显式获取源码声明的公开端点和方法页。
- `fetch_release_documents.py` 按 `sources.json` 获取带日期 PDF，依赖外部 `pdftotext`。
- `audit_history.py` 检查已有文件的重复、抑制、缺口、发布日期及统计定义变化。

网络访问只发生在显式运行相应获取命令时；导入模块不下载或训练。Python 依赖为 requests 和 pandas。最新修订历史不自动等于当时可用版本，少数发布文件不能证明完整历史版本链。

描述性市场分析入口：

```sh
python -m analysis.external_signals --raw-dir data/raw --output-dir artifacts/external_signals
```

这会写统计表和图，不产生预测模型。租金描述指数连接相邻月 town×房型租金/平方米中位数变化，每个单元两个月均需至少五条记录，单元等权；它仅部分控制组成变化，不是官方或完整质量调整指数。共同趋势、重叠窗口和多重比较使相关性不能直接解释为预测增益或因果关系。

<a id="results"></a>
## 历史结果和评价边界

固定划分：拟合截至 2024-09；验证 2024-10 至 12（8274 条）；确认 2025-01 至 03（8924 条）。确认集在开发中被反复查看。以下为累计改动，不能解释为独立的单因素实验。

| 模型 | 验证 RMSE | 确认 RMSE |
|---|---:|---:|
| Global CatBoost | 447.78 | 455.21 |
| + 类别统计 | 445.38 | 452.87 |
| + 完整地址 | 444.91 | 452.72 |
| + 租金时间指数 | 444.47 | 450.89 |

独立的三起点、16 个月预测研究：

| 起点 | RMSE |
|---|---:|
| 2023-05 | 548.70 |
| 2022-03 | 679.23 |
| 2022-10 | 617.82 |
| 起点等权汇总 | 617.56 |

| 历史结果 | 范围 |
|---|---|
| [固定划分指标](results/fixed_split/metrics.csv) | 12 个本地变体及细节 |
| [第一轮](results/rolling_round1/)、[第二轮](results/rolling_round2/)、[第三轮](results/rolling_round3/)、[第四轮](results/rolling_round4/) | 各轮独立协议及新/复用窗口 |
| [三起点汇总](results/rolling_local/aggregate_results.json) | 全局模型 + 时间指数 |
| [外部信号](results/external_signals/analysis_metadata.json) | 描述性关联，不是预测表现 |
| [早期研究](results/early_analysis/scores.csv) | 空间、历史、PCA 及敏感性 |

按原始逐条月租金评分；面积归一化或残差预测先转换回 SGD。偏差为预测减真实值。起点等权 RMSE 为 `sqrt(mean(origin MSE))`，pooled RMSE 则合并逐条记录；二者不可混称。不同研究的时间、特征、信息假设和预测长度不同，不应直接横向宣称优化。缺失分数保留缺失，失败保留失败，小幅重跑差异不算新的优化增益。

一个月的日历滞后不足以证明长预测期内股票、COE 或官方指标可用；需逐项确认起点可用值、发布日期和历史版本。更详细边界见 [docs/evaluation.md](docs/evaluation.md)。

<a id="testing"></a>
## 测试与持续集成

默认 pytest 排除实际拟合模型的测试，CI 同样明确使用 `not training`：

```sh
python scripts/validate_repository.py
python scripts/show_results.py --format json
python -m pytest -m "not training"
python scripts/check_wheel.py
```

非训练测试覆盖核心协议/选择性标签读取/评分/断点和进程契约、第二轮执行边界、第三轮输入与设备契约、第四轮工作范围、rolling 的纯变换，以及配置资源、特征路径和导入顺序。测试夹具禁止未标记测试调用真实 Ridge、CatBoost、XGBoost 拟合，防止新增测试意外触发训练。预处理和数学变换仍可测试。第二轮价格快照测试会拟合真实 Ridge，归入训练测试。

需要明确运行模型回归测试时，使用 `python -m pytest -m training`；包括拟合 checkpoint 的 fixture 所属测试也标记为训练。`python -m pytest -m ""` 可显式运行全部测试。拟合测试不是默认结构检查的一部分。

`check_wheel.py` 用当前环境构建 wheel，在临时目录单独安装，离开源码 checkout 检查 JSON、导入、工作路径和帮助。它不训练、不读取租赁数据，需要 pip、setuptools 和 wheel；使用 `--no-build-isolation` 复用现有构建工具。

语法、导入、帮助、配置和纯契约测试不代表历史模型端到端数值复现。本次排查中，一个旧测试在合成小样本上先拟合再验证恢复；现已归入训练测试，并补全 CatBoostRegressor 的拟合拦截。没有使用项目租赁数据重新训练或生成新的项目逐条预测；历史权重、选定配置、回执和前序资产不会由测试自动补齐。

<a id="contributing"></a>
## 贡献、来源和许可

修改与基线分开，记录输入 schema、源码版本、时间截止、特征、容量、设备、种子和预算。不使用外层评价标签选择模型。保留失败实验及原始评价标准；改变预处理、时间边界或未来协变量假设必须明确记录。

不要提交原始记录、逐条预测、权重、凭据、环境和日志。公开结果应为聚合指标，并注明评价人群。提交前运行相关非训练检查，如实报告已执行和未验证部分。

完整贡献要求见 [CONTRIBUTING.md](CONTRIBUTING.md)，来源归属见 [NOTICE.md](NOTICE.md)。项目尚未选定统一许可，参见 [LICENSE.md](LICENSE.md)。
