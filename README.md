# 新能源汽车消费特征挖掘系统

这是一个基于 Python 的可复现分析系统，用于挖掘懂车帝新能源汽车评论数据，完成需求主题识别、情感满意度分析、消费者分群、关联规则挖掘、竞品对比和可视化看板。

## 项目结构

```
NEV-Consumer-Mining/
├── run.py                      # 一键启动器：检查依赖与产物，运行流水线或启动看板（--check/--pipeline/--port）
├── _smoke_app.py               # 看板冒烟测试：桩化 streamlit，验证 7 页面 21 张图可渲染
├── requirements.txt            # 核心依赖（pandas/scikit-learn/jieba/mlxtend/plotly/streamlit/joblib/pytest）
├── requirements-optional.txt   # 可选依赖（snownlp/transformers/torch/bertopic）
├── pytest.ini                  # pytest 配置（testpaths=tests）
├── README.md
├── .gitignore
├── app/
│   └── streamlit_app.py        # 可视化看板（7 页面：数据总览/消费画像/需求主题/满意度/消费者聚类/竞品对比/数据质量）
├── config/                     # 配置与资源文件
│   ├── settings.json           # 数据路径、use_precleaned 等运行配置
│   ├── sentiment_lexicon.json  # 情感分析词典
│   ├── topics.json             # LDA 主题定义
│   └── stopwords_zh.txt        # 中文停用词
├── src/                        # 分析流水线源码包（2750 行，13 个模块）
│   ├── __init__.py
│   ├── pipeline.py             # 流水线编排，串联 8 步分析任务（130 行）
│   ├── config.py               # 配置加载与路径解析（79 行）
│   ├── precleaned_adapter.py   # 适配层：将 caiji 已清洗数据对齐到统一 schema（306 行）
│   ├── data_preprocessing.py   # 数据清洗、关联、评分/价格/地区归一化（753 行）
│   ├── feature_engineering.py  # 价格/时间/地区/评分/主题/情感特征构建（126 行）
│   ├── topic_mining.py         # LDA 主题模型训练与关键词提取（129 行）
│   ├── sentiment_analysis.py   # 词典法文本情感分析（159 行）
│   ├── consumer_clustering.py  # KMeans 消费者行为分群（238 行）
│   ├── association_rules.py    # Apriori 关联规则挖掘（131 行）
│   ├── reporting.py            # 统计表/HTML 图表/Markdown 报告生成（467 行）
│   ├── database.py             # SQLite 结果写入（81 行）
│   └── utils.py                # 通用工具函数（149 行）
├── models/                     # 训练好的模型（gitignore）
│   ├── lda_model.joblib
│   ├── lda_vectorizer.joblib
│   ├── kmeans_model.joblib
│   └── cluster_preprocessor.joblib
├── data/
│   ├── raw/                    # 原始数据（空；系统读取 caiji/ 已清洗数据，原始文件保持不变）
│   └── processed/             # 清洗数据/特征/SQLite/算法结果（gitignore，约 1.2 GB）
│       ├── reviews_processed.csv         # 清洗后评论主表
│       ├── review_features.csv           # 构建的特征表
│       ├── topic_probabilities.csv       # LDA 主题概率
│       ├── sentiment_results.csv         # 情感分析结果
│       ├── consumer_clusters.csv         # 分群结果
│       ├── association_rules.csv         # 关联规则结果
│       ├── nev_consumer.db               # SQLite 星型/结果库（约 630 MB）
│       └── preprocessing_quality_report.json
├── outputs/
│   ├── tables/                 # 可导入论文的统计表（CSV/JSON，gitignore）
│   ├── figures/                # 交互式 HTML 图表（gitignore）
│   └── reports/                # Markdown 分析报告（gitignore）
├── tests/                      # pytest 单元测试
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_module_imports.py
│   └── test_utils.py
└── docs/                       # 文档（计划中，尚未创建：系统设计与实现.md、数据字典.md）
```

> 说明：标注 `gitignore` 的产物（模型、数据库、输出表/图表/报告）不纳入版本库，运行流水线后自动生成。源码按「配置 → 适配 → 清洗 → 特征 → 主题/情感/聚类/关联 → 报告 → 入库」的职能拆分在 `src/` 中，`pipeline.py` 统一编排，`app/streamlit_app.py` 负责可视化呈现。

## 数据范围

系统优先读取 `caiji/` 目录**已清洗**的数据集（`use_precleaned = true`）：

| 文件 | 内容 | 规模 |
| --- | --- | --- |
| `caiji/分析主表_精简.csv` | 评论主表（已去重、价格已解析、缺失已标记） | 38,636 条 × 58 列 |
| `caiji/car_series_扩展.csv` | 车系元数据（级别、能源类型、口碑数） | 46 车系 |
| `caiji/car_model_全量.csv` | 车型版本参数（续航、电池、电机、加速） | 336 个版本 |

数据覆盖 49 个车系、21 个品牌，按价格段 × 能源类型 × 车辆级别分层抽样。

适配层 `src/precleaned_adapter.py` 只做列名对齐，不重复清洗；
评分标准化、价格区间校验、地区归一化和参数关联仍复用 `src/data_preprocessing.py`，
因此两条数据入口产出完全一致的 schema。

若关闭 `use_precleaned`，系统回退到读取 `评论数据_12车系.csv`、`car_series.csv`、`car_model.csv` 并自行清洗。

原始文件保持不变。处理结果写入本项目：

- `data/processed/`：清洗数据、特征表、SQLite 数据库和算法结果
- `models/`：主题模型、聚类模型和评估指标
- `outputs/tables/`：可导入论文的统计表
- `outputs/figures/`：交互式 HTML 图表
- `outputs/reports/`：分析摘要报告

## 环境安装

```powershell
cd D:\PythonProject1\设计\NEV-Consumer-Mining
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行完整流水线

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

流水线依次执行：

1. 清洗评论并关联车系、版本和车型参数。
2. 构建价格、时间、地区、评分、主题和情感特征。
3. 训练 LDA 主题模型。
4. 执行词典法文本情感分析。
5. 使用 KMeans 划分消费者群体。
6. 使用 Apriori 挖掘消费特征关联规则。
7. 生成统计表、交互式图表和 Markdown 分析报告。
8. 将关键结果写入 SQLite。

常用调试参数：

```powershell
.\.venv\Scripts\python.exe -m src.pipeline --skip-topics
.\.venv\Scripts\python.exe -m src.pipeline --skip-clustering
.\.venv\Scripts\python.exe -m src.pipeline --skip-association
```

## 启动可视化系统

```powershell
cd D:\PythonProject1\设计\NEV-Consumer-Mining
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

系统包括数据总览、消费画像、需求主题、满意度、消费者聚类、竞品对比和数据质量七个页面。

项目设计说明见 `docs/系统设计与实现.md`，字段说明见 `docs/数据字典.md`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 方法边界

- 数据没有年龄、性别、职业和收入字段，因此系统分析的是购车行为、地区、价格和评论语义特征，属于**行为细分**而非人口统计细分。
- 购车价格缺失已证实为非随机缺失（认证车主缺失率 6%、非认证车主 33%，缺失组评分低 0.151），因此主表保留缺失标记，不静默填补。
- 部分评分维度缺失率较高（动力/舒适/操控约 88%、辅助驾驶/智能座舱约 79%），按接口开放范围决定，只对有效样本计算均值并报告样本量。
- 版本级参数匹配不完整时使用车系中位数补充，并保留 `参数匹配方式` 标记。
