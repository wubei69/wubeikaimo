from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import get_paths, load_settings
from src.utils import setup_logger, write_json


LOGGER = setup_logger()

SCORE_COLUMNS = [
    "综合评分",
    "外观评分",
    "空间评分",
    "动力评分",
    "舒适评分",
    "性价比评分",
    "内饰评分",
    "操控评分",
    "续航评分",
    "辅助驾驶评分",
    "智能座舱评分",
    "驾驶评分",
]

TOPIC_COLUMNS = [
    "主题_续航充电",
    "主题_空间实用",
    "主题_动力操控",
    "主题_智能驾驶",
    "主题_智能座舱",
    "主题_舒适静谧",
    "主题_外观内饰",
    "主题_性价比售后",
]


def _save_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(
        template="plotly_white",
        font={"family": "Microsoft YaHei, Noto Sans CJK SC, sans-serif"},
        margin={"l": 40, "r": 30, "t": 60, "b": 40},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _format_cell(value: Any) -> str:
    """把浮点值压到可读精度。

    原始 mean() 结果直接进表格会出现 4.443128171189058 这种长度，
    报告是给人看、给论文引用的，保留 3 位小数足够。
    """
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown_table(data: pd.DataFrame, columns: list[str], limit: int = 12) -> str:
    subset = data[columns].head(limit).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in subset.itertuples(index=False):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def generate_reports(
    features: pd.DataFrame | None = None,
    topic_results: pd.DataFrame | None = None,
    sentiment_results: pd.DataFrame | None = None,
    cluster_results: pd.DataFrame | None = None,
    rules: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    settings = settings or load_settings()
    paths = get_paths(settings)
    tables_dir = paths["tables_dir"]
    figures_dir = paths["figures_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if features is None:
        features = pd.read_csv(paths["feature_data"], encoding="utf-8-sig", low_memory=False)
    data = features.copy()
    if sentiment_results is not None and "文本情感倾向" not in data.columns:
        data = data.merge(
            sentiment_results[
                ["doc_id", "文本情感分", "文本情感倾向", "评分情感倾向"]
            ],
            on="doc_id",
            how="left",
        )
    if cluster_results is not None and "消费者群体" not in data.columns:
        data = data.merge(
            cluster_results[["doc_id", "消费者群体", "cluster_id"]],
            on="doc_id",
            how="left",
        )
    if topic_results is not None and "主导主题" not in data.columns:
        data = data.merge(
            topic_results[["doc_id", "主导主题", "主导主题概率"]],
            on="doc_id",
            how="left",
        )

    available_scores = [column for column in SCORE_COLUMNS if column in data.columns]
    series_summary = (
        data.groupby(["品牌", "车型"], dropna=False)
        .agg(
            评论数=("doc_id", "count"),
            用户数=("用户", "nunique"),
            购车价格中位数=("购车价格_分析值", "median"),
            购车价格均值=("购车价格_分析值", "mean"),
            核心评分均值=("核心评分均值", "mean"),
            平均评论长度=("评论长度", "mean"),
            正面文本占比=(
                "文本情感倾向",
                lambda values: values.eq("正面").mean() if values.notna().any() else np.nan,
            ),
        )
        .reset_index()
    )
    for score in available_scores:
        series_summary[f"{score}_均值"] = (
            data.groupby(["品牌", "车型"], dropna=False)[score].mean().values
        )
    series_summary = series_summary.sort_values("评论数", ascending=False)
    series_summary.to_csv(
        tables_dir / "series_summary.csv", index=False, encoding="utf-8-sig"
    )

    score_rows = []
    for score in available_scores:
        series = data[score]
        score_rows.append(
            {
                "评分维度": score,
                "有效样本数": int(series.notna().sum()),
                "有效率": float(series.notna().mean()),
                "均值": float(series.mean()),
                "标准差": float(series.std()),
                "中位数": float(series.median()),
            }
        )
    score_summary = pd.DataFrame(score_rows)
    score_summary.to_csv(
        tables_dir / "score_summary.csv", index=False, encoding="utf-8-sig"
    )

    region_summary = (
        data.groupby("购车地区", dropna=False)
        .agg(
            评论数=("doc_id", "count"),
            价格中位数=("购车价格_分析值", "median"),
            核心评分均值=("核心评分均值", "mean"),
        )
        .reset_index()
        .sort_values("评论数", ascending=False)
    )
    region_summary.to_csv(
        tables_dir / "region_summary.csv", index=False, encoding="utf-8-sig"
    )

    price_segment_summary = (
        data.groupby("价格段", dropna=False)
        .agg(
            评论数=("doc_id", "count"),
            价格中位数=("购车价格_分析值", "median"),
            续航中位数=("续航_参数", "median"),
            核心评分均值=("核心评分均值", "mean"),
        )
        .reset_index()
    )
    price_segment_summary.to_csv(
        tables_dir / "price_segment_summary.csv", index=False, encoding="utf-8-sig"
    )

    time_trend = (
        data.dropna(subset=["评论年月"])
        .groupby("评论年月")
        .agg(
            评论数=("doc_id", "count"),
            核心评分均值=("核心评分均值", "mean"),
            正面文本占比=(
                "文本情感倾向",
                lambda values: values.eq("正面").mean() if values.notna().any() else np.nan,
            ),
        )
        .reset_index()
    )
    time_trend.to_csv(
        tables_dir / "time_trend.csv", index=False, encoding="utf-8-sig"
    )

    topic_rows = []
    for column in TOPIC_COLUMNS:
        if column not in data.columns:
            continue
        topic_rows.append(
            {
                "需求主题": column.removeprefix("主题_"),
                "评论提及率": float(data[column].gt(0).mean()),
                "主题强度均值": float(data[column].mean()),
                "正面文本占比": (
                    float(data.loc[data[column].gt(0), "文本情感倾向"].eq("正面").mean())
                    if data.loc[data[column].gt(0), "文本情感倾向"].notna().any()
                    else np.nan
                ),
            }
        )
    topic_summary = pd.DataFrame(topic_rows)
    topic_summary.to_csv(
        tables_dir / "topic_summary.csv", index=False, encoding="utf-8-sig"
    )

    topic_heatmap = data.pivot_table(
        index="车型",
        values=[column for column in TOPIC_COLUMNS if column in data.columns],
        aggfunc="mean",
    )
    topic_heatmap.columns = [
        column.removeprefix("主题_") for column in topic_heatmap.columns
    ]
    topic_heatmap.to_csv(tables_dir / "series_topic_heatmap.csv", encoding="utf-8-sig")

    overview = {
        "评论数": int(len(data)),
        "用户数": int(data["用户"].nunique()),
        "品牌数": int(data["品牌"].nunique()),
        "车系数": int(data["车型"].nunique()),
        "地区数": int(data["购车地区"].nunique()),
        "评论时间最小值": data["评论时间"].min(),
        "评论时间最大值": data["评论时间"].max(),
        "购车价格中位数": float(data["购车价格_分析值"].median()),
        "购车价格均值": float(data["购车价格_分析值"].mean()),
        "价格估算比例": float(data["购车价格是否估算"].mean()),
        "版本参数匹配比例": float(data["参数匹配方式"].eq("版本匹配").mean()),
        "平均评论长度": float(data["评论长度"].mean()),
        "正面文本占比": (
            float(data["文本情感倾向"].eq("正面").mean())
            if "文本情感倾向" in data
            else None
        ),
    }
    write_json(tables_dir / "overview_metrics.json", overview)

    _save_figure(
        px.bar(
            series_summary,
            x="评论数",
            y="车型",
            orientation="h",
            color="品牌",
            title="各车系评论样本量",
        ).update_layout(yaxis={"categoryorder": "total ascending"}),
        figures_dir / "series_sample_size.html",
    )
    _save_figure(
        px.box(
            data,
            x="车型",
            y="购车价格_万元",
            color="品牌",
            title="各车系购车价格分布",
        ).update_layout(xaxis={"categoryorder": "median descending"}),
        figures_dir / "price_distribution.html",
    )
    _save_figure(
        px.histogram(
            data,
            x="购车价格_万元",
            color="价格段",
            nbins=40,
            title="购车价格分布",
        ),
        figures_dir / "price_histogram.html",
    )

    if score_rows:
        top_series = series_summary.head(6)["车型"].tolist()
        fig = go.Figure()
        for series_name in top_series:
            row = series_summary[series_summary["车型"] == series_name]
            if row.empty:
                continue
            values = [row[f"{score}_均值"].iloc[0] for score in available_scores]
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=available_scores + [available_scores[0]],
                    fill="toself",
                    name=series_name,
                )
            )
        fig.update_layout(title="重点车系满意度雷达图", polar={"radialaxis": {"range": [1, 5]}})
        _save_figure(fig, figures_dir / "satisfaction_radar.html")

    topic_plot_data = data.melt(
        id_vars=["车型"],
        value_vars=[column for column in TOPIC_COLUMNS if column in data.columns],
        var_name="主题",
        value_name="主题强度",
    )
    topic_plot_data["主题"] = topic_plot_data["主题"].str.removeprefix("主题_")
    _save_figure(
        px.density_heatmap(
            topic_plot_data,
            x="主题",
            y="车型",
            z="主题强度",
            histfunc="avg",
            title="车系需求主题热力图",
        ),
        figures_dir / "topic_heatmap.html",
    )

    if "文本情感倾向" in data.columns:
        sentiment_counts = (
            data["文本情感倾向"].value_counts().rename_axis("情感").reset_index(name="数量")
        )
        _save_figure(
            px.pie(
                sentiment_counts,
                names="情感",
                values="数量",
                hole=0.45,
                title="评论文本情感分布",
            ),
            figures_dir / "sentiment_distribution.html",
        )

    if cluster_results is not None and not cluster_results.empty:
        _save_figure(
            px.scatter(
                cluster_results,
                x="PCA1",
                y="PCA2",
                color="消费者群体",
                hover_data=["车型", "能源类型", "购车价格_分析值"],
                opacity=0.65,
                title="消费者群体聚类分布（PCA二维投影）",
            ),
            figures_dir / "consumer_clusters.html",
        )

    if not time_trend.empty:
        _save_figure(
            px.line(
                time_trend,
                x="评论年月",
                y="评论数",
                markers=True,
                title="评论样本月度趋势",
            ),
            figures_dir / "monthly_trend.html",
        )

    if rules is not None and not rules.empty:
        rule_plot = rules.head(15).copy()
        rule_plot["规则"] = rule_plot["前件"] + " → " + rule_plot["后件"]
        _save_figure(
            px.bar(
                rule_plot.sort_values("lift"),
                x="lift",
                y="规则",
                orientation="h",
                color="confidence",
                title="高提升度消费特征关联规则",
            ),
            figures_dir / "association_rules.html",
        )

    report_lines = [
        "# 新能源汽车消费特征挖掘分析报告",
        "",
        "## 一、数据概况",
        "",
        f"- 有效评论样本：{overview['评论数']:,} 条。",
        f"- 覆盖品牌：{overview['品牌数']} 个，车系：{overview['车系数']} 个。",
        f"- 购车价格中位数：{overview['购车价格中位数'] / 10000:.2f} 万元。",
        f"- 版本级参数匹配比例：{overview['版本参数匹配比例']:.2%}。",
        f"- 平均评论长度：{overview['平均评论长度']:.1f} 字。",
        "",
        "## 二、主要发现",
        "",
        "### 2.1 样本量较高的车系",
        "",
        _markdown_table(
            series_summary,
            ["品牌", "车型", "评论数", "购车价格中位数", "核心评分均值"],
            limit=12,
        ),
        "",
        "### 2.2 消费需求主题",
        "",
        _markdown_table(
            topic_summary.sort_values("评论提及率", ascending=False),
            ["需求主题", "评论提及率", "主题强度均值", "正面文本占比"],
            limit=8,
        ),
        "",
        "### 2.3 消费者群体",
        "",
    ]
    if cluster_results is not None and "消费者群体" in cluster_results.columns:
        cluster_summary = (
            cluster_results.groupby("消费者群体")
            .agg(
                样本数=("doc_id", "count"),
                价格中位数=("购车价格_分析值", "median"),
                续航中位数=("续航_参数", "median"),
            )
            .reset_index()
            .sort_values("样本数", ascending=False)
        )
        report_lines.append(
            _markdown_table(
                cluster_summary,
                ["消费者群体", "样本数", "价格中位数", "续航中位数"],
                limit=10,
            )
        )
    else:
        report_lines.append("未生成消费者聚类结果。")

    report_lines.extend(
        [
            "",
            "## 三、方法限制",
            "",
            "- 评论数据缺少年龄、性别、职业和家庭收入，消费画像由购车行为、地区和文本偏好构成。",
            "- 高频缺失评分不做批量填充，满意度结论只基于有效样本。",
            "- 价格或车型参数使用估算值时，系统保留相应标记，可开展敏感性分析。",
            "- 关联规则反映共现关系，不代表因果关系。",
            "",
            "## 四、输出文件",
            "",
            "- 统计表：`outputs/tables/`",
            "- 交互式图表：`outputs/figures/`",
            "- SQLite数据库：`data/processed/nev_consumer.db`",
            "- 可视化系统：运行 `streamlit run app/streamlit_app.py`",
        ]
    )
    report_path = paths["reports_dir"] / "analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    LOGGER.info("统计报告与图表生成完成")
    return {
        "series_summary": series_summary,
        "score_summary": score_summary,
        "region_summary": region_summary,
        "price_segment_summary": price_segment_summary,
        "time_trend": time_trend,
        "topic_summary": topic_summary,
        "topic_heatmap": topic_heatmap.reset_index(),
    }

