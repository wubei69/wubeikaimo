from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_paths, load_settings  # noqa: E402


st.set_page_config(
    page_title="新能源汽车消费特征挖掘系统",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #152536;
        --muted: #607080;
        --line: #dbe3e8;
        --panel: #f7f9fa;
        --teal: #087e8b;
        --orange: #e07a3f;
    }
    .stApp { background: #ffffff; color: var(--ink); }
    [data-testid="stSidebar"] {
        background: #f2f6f7;
        border-right: 1px solid var(--line);
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px 16px;
    }
    [data-testid="stMetricValue"] { color: var(--ink); }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    .section-note { color: var(--muted); font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

SETTINGS = load_settings()
PATHS = get_paths(SETTINGS)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    feature_data = load_csv(str(PATHS["feature_data"]))
    extras = {
        "topics": load_csv(str(PATHS["topic_probabilities"])),
        "sentiment": load_csv(str(PATHS["sentiment_results"])),
        "clusters": load_csv(str(PATHS["cluster_results"])),
        "rules": load_csv(str(PATHS["association_rules"])),
        "series_summary": load_csv(str(PATHS["tables_dir"] / "series_summary.csv")),
        "score_summary": load_csv(str(PATHS["tables_dir"] / "score_summary.csv")),
        "topic_summary": load_csv(str(PATHS["tables_dir"] / "topic_summary.csv")),
        "time_trend": load_csv(str(PATHS["tables_dir"] / "time_trend.csv")),
        "region_summary": load_csv(str(PATHS["tables_dir"] / "region_summary.csv")),
        "cluster_summary": load_csv(
            str(PATHS["tables_dir"] / "consumer_cluster_summary.csv")
        ),
    }
    if not feature_data.empty:
        if "文本情感倾向" not in feature_data.columns and not extras["sentiment"].empty:
            feature_data = feature_data.merge(
                extras["sentiment"], on="doc_id", how="left"
            )
        if "消费者群体" not in feature_data.columns and not extras["clusters"].empty:
            feature_data = feature_data.merge(
                extras["clusters"][
                    ["doc_id", "cluster_id", "消费者群体", "PCA1", "PCA2"]
                ],
                on="doc_id",
                how="left",
            )
    return feature_data, extras


def chart_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 30, "r": 20, "t": 55, "b": 30},
        font={"family": "Microsoft YaHei, Noto Sans CJK SC, sans-serif"},
        legend_title_text="",
    )
    return fig


def data_download(label: str, data: pd.DataFrame, file_name: str) -> None:
    if data.empty:
        return
    st.download_button(
        label,
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
        width="content",
    )


data, outputs = read_data()

if data.empty:
    st.title("新能源汽车消费特征挖掘系统")
    st.warning("尚未找到分析结果。请先在项目根目录执行完整流水线。")
    st.code(
        r".\.venv\Scripts\python.exe -m src.pipeline",
        language="powershell",
    )
    st.stop()

st.sidebar.title("新能源汽车消费特征")
page = st.sidebar.radio(
    "分析页面",
    [
        "数据总览",
        "消费画像",
        "需求主题",
        "满意度分析",
        "消费者聚类",
        "竞品对比",
        "数据质量",
    ],
)
st.sidebar.divider()
st.sidebar.metric("评论样本", f"{len(data):,}")
st.sidebar.metric("覆盖车系", f"{data['车型'].nunique()}")
st.sidebar.metric("覆盖地区", f"{data['购车地区'].nunique()}")

score_columns = [
    column
    for column in [
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
    if column in data.columns
]

topic_columns = [column for column in data.columns if column.startswith("主题_")]

if page == "数据总览":
    st.title("数据总览")
    metric_columns = st.columns(5)
    metric_columns[0].metric("有效评论", f"{len(data):,}")
    metric_columns[1].metric("用户数", f"{data['用户'].nunique():,}")
    metric_columns[2].metric(
        "价格中位数",
        f"{data['购车价格_分析值'].median() / 10000:.1f}万",
    )
    metric_columns[3].metric(
        "参数版本匹配率",
        f"{data['参数匹配方式'].eq('版本匹配').mean():.1%}",
    )
    metric_columns[4].metric(
        "正面文本占比",
        (
            f"{data['文本情感倾向'].eq('正面').mean():.1%}"
            if "文本情感倾向" in data
            else "待计算"
        ),
    )

    summary = outputs["series_summary"]
    if not summary.empty:
        left, right = st.columns([1.2, 1])
        with left:
            fig = px.bar(
                summary.sort_values("评论数"),
                x="评论数",
                y="车型",
                orientation="h",
                color="品牌",
                title="各车系评论样本量",
            )
            st.plotly_chart(chart_layout(fig, 450), width="stretch")
        with right:
            fig = px.histogram(
                data,
                x="购车价格_万元",
                color="价格段",
                nbins=35,
                title="购车价格分布",
            )
            st.plotly_chart(chart_layout(fig, 450), width="stretch")

    if not outputs["time_trend"].empty:
        trend_data = outputs["time_trend"].copy()
        trend_data["评论年月"] = trend_data["评论年月"].astype(str)
        fig = px.line(
            trend_data,
            x="评论年月",
            y="评论数",
            markers=True,
            title="评论月度趋势",
        )
        st.plotly_chart(chart_layout(fig, 360), width="stretch")

    st.subheader("车系统计")
    display_columns = [
        column
        for column in [
            "品牌",
            "车型",
            "评论数",
            "用户数",
            "购车价格中位数",
            "核心评分均值",
            "正面文本占比",
        ]
        if column in summary.columns
    ]
    st.dataframe(
        summary[display_columns].style.format(
            {
                "购车价格中位数": "{:,.0f}",
                "核心评分均值": "{:.2f}",
                "正面文本占比": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

elif page == "消费画像":
    st.title("消费画像")
    filter_columns = st.columns(4)
    brands = ["全部"] + sorted(data["品牌"].dropna().unique().tolist())
    selected_brand = filter_columns[0].selectbox("品牌", brands)
    energy_types = ["全部"] + sorted(data["能源类型"].dropna().unique().tolist())
    selected_energy = filter_columns[1].selectbox("能源类型", energy_types)
    price_values = data["购车价格_万元"].dropna()
    price_range = filter_columns[2].slider(
        "价格区间（万元）",
        min_value=float(price_values.min()),
        max_value=float(price_values.max()),
        value=(float(price_values.quantile(0.01)), float(price_values.quantile(0.99))),
    )
    regions = ["全部"] + data["购车地区"].value_counts().head(40).index.tolist()
    selected_region = filter_columns[3].selectbox("地区", regions)

    filtered = data.copy()
    if selected_brand != "全部":
        filtered = filtered[filtered["品牌"] == selected_brand]
    if selected_energy != "全部":
        filtered = filtered[filtered["能源类型"] == selected_energy]
    if selected_region != "全部":
        filtered = filtered[filtered["购车地区"] == selected_region]
    filtered = filtered[
        filtered["购车价格_万元"].between(price_range[0], price_range[1])
    ]

    col1, col2 = st.columns(2)
    with col1:
        price_segment = (
            filtered["价格段"].value_counts().rename_axis("价格段").reset_index(name="评论数")
        )
        fig = px.bar(
            price_segment,
            x="价格段",
            y="评论数",
            color="价格段",
            title="价格区间分布",
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    with col2:
        region_data = filtered["购车地区"].value_counts().head(15).rename_axis("地区").reset_index(name="评论数")
        fig = px.bar(
            region_data.sort_values("评论数"),
            x="评论数",
            y="地区",
            orientation="h",
            title="重点购车地区",
        )
        st.plotly_chart(chart_layout(fig), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        level_data = (
            filtered["车辆级别"].value_counts().rename_axis("车辆级别").reset_index(name="评论数")
        )
        fig = px.pie(level_data, names="车辆级别", values="评论数", hole=0.45, title="车辆级别分布")
        st.plotly_chart(chart_layout(fig), width="stretch")
    with col4:
        energy_data = (
            filtered["能源类型"].value_counts().rename_axis("能源类型").reset_index(name="评论数")
        )
        fig = px.pie(energy_data, names="能源类型", values="评论数", hole=0.45, title="能源类型分布")
        st.plotly_chart(chart_layout(fig), width="stretch")

    trend = (
        filtered.dropna(subset=["购车年份"])
        .groupby(["购车年份", "能源类型"])
        .size()
        .reset_index(name="评论数")
    )
    fig = px.bar(
        trend,
        x="购车年份",
        y="评论数",
        color="能源类型",
        barmode="stack",
        title="购车年份结构",
    )
    st.plotly_chart(chart_layout(fig, 360), width="stretch")
    data_download("下载当前画像数据", filtered, "consumer_profile_filtered.csv")

elif page == "需求主题":
    st.title("需求主题")
    topic_summary = outputs["topic_summary"]
    if topic_summary.empty:
        st.warning("尚未生成主题统计结果。")
    else:
        topic_summary = topic_summary.sort_values("评论提及率", ascending=False)
        col1, col2 = st.columns([1.1, 1])
        with col1:
            fig = px.bar(
                topic_summary.sort_values("评论提及率"),
                x="评论提及率",
                y="需求主题",
                orientation="h",
                color="主题强度均值",
                title="消费者需求提及率",
            )
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(chart_layout(fig, 430), width="stretch")
        with col2:
            fig = px.scatter(
                topic_summary,
                x="主题强度均值",
                y="正面文本占比",
                size="评论提及率",
                color="需求主题",
                text="需求主题",
                title="主题强度与正面情绪",
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(chart_layout(fig, 430), width="stretch")

        heatmap_data = (
            data.groupby("车型")[topic_columns].mean().reset_index()
            if topic_columns
            else pd.DataFrame()
        )
        if not heatmap_data.empty:
            matrix = heatmap_data.set_index("车型")
            matrix.columns = [column.removeprefix("主题_") for column in matrix.columns]
            fig = px.imshow(
                matrix,
                aspect="auto",
                color_continuous_scale=["#edf6f7", "#65b7bf", "#087e8b"],
                title="车系需求主题热力图",
            )
            st.plotly_chart(chart_layout(fig, 560), width="stretch")

        topic_metrics = load_json_file(str(PATHS["tables_dir"] / "topic_model_metrics.json"))
        if topic_metrics:
            metric_columns = st.columns(3)
            metric_columns[0].metric("主题数", topic_metrics.get("topic_count", "-"))
            metric_columns[1].metric("词表大小", f"{topic_metrics.get('vocabulary_size', 0):,}")
            metric_columns[2].metric(
                "LDA困惑度", f"{topic_metrics.get('perplexity', 0):.1f}"
            )
            topic_words = topic_metrics.get("topic_top_words", {})
            word_table = pd.DataFrame(
                [
                    {"业务主题": topic_metrics["topic_names"].get(topic, topic), "关键词": "、".join(words[:15])}
                    for topic, words in topic_words.items()
                ]
            )
            st.dataframe(word_table, width="stretch", hide_index=True)

        keyword_path = PATHS["tables_dir"] / "tfidf_keywords.csv"
        keyword_data = load_csv(str(keyword_path))
        if not keyword_data.empty:
            st.subheader("TF-IDF关键词")
            st.dataframe(keyword_data.head(30), width="stretch", hide_index=True)
        data_download("下载主题统计", topic_summary, "topic_summary.csv")

elif page == "满意度分析":
    st.title("满意度分析")
    score_summary = outputs["score_summary"]
    if score_summary.empty:
        st.warning("尚未生成评分统计结果。")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                score_summary.sort_values("均值"),
                x="均值",
                y="评分维度",
                orientation="h",
                color="有效率",
                range_x=[1, 5],
                title="各评分维度均值",
            )
            st.plotly_chart(chart_layout(fig, 430), width="stretch")
        with col2:
            fig = px.bar(
                score_summary.sort_values("有效率"),
                x="有效率",
                y="评分维度",
                orientation="h",
                color="有效率",
                title="评分字段有效率",
            )
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(chart_layout(fig, 430), width="stretch")

        selected_series = st.multiselect(
            "对比车系",
            sorted(data["车型"].unique().tolist()),
            default=outputs["series_summary"].head(5)["车型"].tolist(),
            max_selections=6,
        )
        if selected_series and score_columns:
            means = (
                data[data["车型"].isin(selected_series)]
                .groupby("车型")[score_columns]
                .mean()
                .reset_index()
            )
            fig = go.Figure()
            for row in means.itertuples(index=False):
                values = [getattr(row, score) for score in score_columns]
                fig.add_trace(
                    go.Scatterpolar(
                        r=values + [values[0]],
                        theta=score_columns + [score_columns[0]],
                        fill="toself",
                        name=row.车型,
                    )
                )
            fig.update_layout(title="车系满意度雷达图", polar={"radialaxis": {"range": [1, 5]}})
            st.plotly_chart(chart_layout(fig, 560), width="stretch")

        if "文本情感倾向" in data:
            sentiment_series = (
                data.groupby(["车型", "文本情感倾向"]).size().reset_index(name="评论数")
            )
            fig = px.bar(
                sentiment_series,
                x="车型",
                y="评论数",
                color="文本情感倾向",
                title="车系文本情感结构",
            )
            # Plotly 没有 barmode="fill" 这个枚举值（合法值仅
            # stack / group / overlay / relative）。这里要的是百分比堆叠，
            # 正确写法是 barmode="relative" + layout.barnorm="percent"，
            # 二者配合才与下方 tickformat=".0%" 对应得上。
            fig.update_layout(barmode="relative", barnorm="percent")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(chart_layout(fig, 430), width="stretch")
        st.dataframe(score_summary, width="stretch", hide_index=True)

elif page == "消费者聚类":
    st.title("消费者聚类")
    cluster_data = outputs["clusters"]
    if cluster_data.empty:
        st.warning("尚未生成消费者聚类结果。")
    else:
        merged = data.merge(
            cluster_data[["doc_id", "消费者群体", "PCA1", "PCA2"]],
            on="doc_id",
            how="left",
            suffixes=("", "_聚类"),
        )
        summary = outputs["cluster_summary"]
        if not summary.empty:
            metric_columns = st.columns(min(5, len(summary)))
            for index, row in summary.head(5).reset_index(drop=True).iterrows():
                metric_columns[index].metric(
                    row["消费者群体"],
                    f"{int(row['样本数']):,}",
                    f"价格中位数 {row['价格中位数'] / 10000:.1f}万",
                )

        fig = px.scatter(
            merged,
            x="PCA1",
            y="PCA2",
            color="消费者群体",
            hover_data=["车型", "能源类型", "购车价格_分析值"],
            opacity=0.6,
            title="消费者群体分布（PCA二维投影）",
        )
        st.plotly_chart(chart_layout(fig, 560), width="stretch")

        if not summary.empty:
            fig = px.bar(
                summary,
                x="消费者群体",
                y="价格中位数",
                color="消费者群体",
                title="各群体价格中位数",
            )
            st.plotly_chart(chart_layout(fig, 380), width="stretch")
            st.dataframe(summary, width="stretch", hide_index=True)

        cluster_metrics = load_json_file(str(PATHS["tables_dir"] / "cluster_metrics.json"))
        if cluster_metrics:
            metric_columns = st.columns(3)
            metric_columns[0].metric("最佳聚类数", cluster_metrics.get("best_k", "-"))
            metric_columns[1].metric(
                "轮廓系数", f"{cluster_metrics.get('selected_silhouette', 0):.3f}"
            )
            explained = cluster_metrics.get("pca_explained_variance", [0, 0])
            metric_columns[2].metric(
                "PCA解释率", f"{sum(explained) * 100:.1f}%"
            )
        data_download("下载聚类结果", merged, "consumer_clusters.csv")

elif page == "竞品对比":
    st.title("竞品对比")
    series_options = sorted(data["车型"].unique().tolist())
    selected = st.multiselect(
        "选择车系",
        series_options,
        default=outputs["series_summary"].head(4)["车型"].tolist(),
        max_selections=6,
    )
    if selected:
        selected_data = data[data["车型"].isin(selected)]
        means = selected_data.groupby("车型")[score_columns].mean().reset_index()
        fig = go.Figure()
        for row in means.itertuples(index=False):
            values = [getattr(row, score) for score in score_columns]
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=score_columns + [score_columns[0]],
                    fill="toself",
                    name=row.车型,
                )
            )
        fig.update_layout(title="满意度对比", polar={"radialaxis": {"range": [1, 5]}})
        st.plotly_chart(chart_layout(fig, 560), width="stretch")

        topic_compare = (
            selected_data.groupby("车型")[topic_columns].mean().reset_index()
        )
        topic_compare = topic_compare.melt(
            id_vars="车型",
            var_name="需求主题",
            value_name="主题强度",
        )
        topic_compare["需求主题"] = topic_compare["需求主题"].str.removeprefix("主题_")
        fig = px.bar(
            topic_compare,
            x="需求主题",
            y="主题强度",
            color="车型",
            barmode="group",
            title="需求主题强度对比",
        )
        st.plotly_chart(chart_layout(fig, 430), width="stretch")

        comparison = (
            selected_data.groupby("车型")
            .agg(
                评论数=("doc_id", "count"),
                用户数=("用户", "nunique"),
                价格中位数=("购车价格_分析值", "median"),
                续航中位数=("续航_参数", "median"),
                核心评分均值=("核心评分均值", "mean"),
                正面文本占比=(
                    "文本情感倾向",
                    lambda values: values.eq("正面").mean() if values.notna().any() else None,
                ),
            )
            .reset_index()
        )
        st.dataframe(comparison, width="stretch", hide_index=True)
        data_download("下载竞品对比", comparison, "series_comparison.csv")
    else:
        st.info("请选择至少一个车系。")

elif page == "数据质量":
    st.title("数据质量")
    missing = (
        data.isna()
        .mean()
        .sort_values(ascending=False)
        .rename_axis("字段")
        .reset_index(name="缺失率")
    )
    missing = missing[missing["缺失率"] > 0].head(25)

    metric_columns = st.columns(4)
    metric_columns[0].metric("原始重复标识行", "3")
    metric_columns[1].metric(
        "价格估算比例", f"{data['购车价格是否估算'].mean():.2%}"
    )
    metric_columns[2].metric(
        "车系参数回退比例", f"{data['参数匹配方式'].eq('车系中位数').mean():.2%}"
    )
    metric_columns[3].metric(
        "未知地区比例", f"{data['地区是否未知'].mean():.2%}"
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            missing.sort_values("缺失率"),
            x="缺失率",
            y="字段",
            orientation="h",
            title="主要字段缺失率",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(chart_layout(fig, 480), width="stretch")
    with col2:
        verification = pd.DataFrame(
            {
                "检查项": ["评论文本非空", "价格有效或估算", "评分维度有值", "参数匹配完成"],
                "通过率": [
                    data["评论内容"].notna().mean(),
                    data["购车价格_分析值"].notna().mean(),
                    data[score_columns].notna().any(axis=1).mean() if score_columns else 0,
                    data["参数匹配方式"].notna().mean(),
                ],
            }
        )
        fig = px.bar(
            verification,
            x="通过率",
            y="检查项",
            orientation="h",
            title="关键质量检查",
        )
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(chart_layout(fig, 480), width="stretch")

    st.dataframe(missing, width="stretch", hide_index=True)
    if not outputs["rules"].empty:
        st.subheader("关联规则")
        st.dataframe(outputs["rules"].head(30), width="stretch", hide_index=True)
    data_download("下载完整特征数据", data, "review_features.csv")

