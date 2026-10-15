from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import get_paths, load_settings
from src.utils import setup_logger, write_json


LOGGER = setup_logger()


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

NUMERIC_COLUMNS = [
    "购车价格_分析值",
    "续航_参数",
    "电池电量_参数",
    "电机功率_参数",
    "百公里加速_参数",
    "词数",
    "核心评分均值",
    *TOPIC_COLUMNS,
]

CATEGORICAL_COLUMNS = ["能源类型", "车辆级别"]


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        sparse_threshold=0.0,
    )


def _select_k(
    matrix: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int,
) -> tuple[int, list[dict[str, float]]]:
    metrics: list[dict[str, float]] = []
    minimum_cluster_size = max(20, int(len(matrix) * 0.05))
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=15)
        labels = model.fit_predict(matrix)
        cluster_sizes = np.bincount(labels, minlength=k)
        sample_size = min(5000, len(matrix))
        silhouette = silhouette_score(
            matrix,
            labels,
            sample_size=sample_size,
            random_state=random_state,
        )
        metrics.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette),
                "minimum_cluster_size": int(cluster_sizes.min()),
                "minimum_cluster_ratio": float(cluster_sizes.min() / len(matrix)),
            }
        )
    valid_metrics = [
        item for item in metrics if item["minimum_cluster_size"] >= minimum_cluster_size
    ]
    best = max(valid_metrics or metrics, key=lambda item: item["silhouette"])
    return int(best["k"]), metrics


def _cluster_names(
    data: pd.DataFrame,
    cluster_column: str,
) -> dict[int, str]:
    summary = (
        data.groupby(cluster_column)
        .agg(
            价格中位数=("购车价格_分析值", "median"),
            续航中位数=("续航_参数", "median"),
            插混占比=("能源类型", lambda values: values.fillna("").str.contains("插电|增程").mean()),
            纯电占比=("是否纯电", "mean"),
            智能主题均值=("主题_智能座舱", "mean"),
            智驾主题均值=("主题_智能驾驶", "mean"),
            舒适主题均值=("主题_舒适静谧", "mean"),
            样本数=("doc_id", "count"),
        )
        .sort_values("价格中位数")
    )
    global_price = data["购车价格_分析值"].median()
    global_range = data["续航_参数"].median()
    smart_threshold = data[["主题_智能座舱", "主题_智能驾驶"]].mean().mean()
    used: dict[str, int] = {}
    names: dict[int, str] = {}
    for cluster_id, row in summary.iterrows():
        # 纯电占比放在价格之后判断：同为高端价位，纯电主导和增程/插混主导
        # 是两类完全不同的用车场景，混在一个名字里解释不了任何东西
        if row["价格中位数"] < global_price * 0.85:
            base = "入门通勤型"
        elif row["价格中位数"] > global_price * 1.35:
            if row["纯电占比"] >= 0.7:
                base = "旗舰纯电长续航型"
            elif row["续航中位数"] >= global_range:
                base = "高端混动长续航型"
            else:
                base = "高端家用型"
        elif row["插混占比"] >= 0.5:
            base = "插混家用型"
        elif max(row["智能主题均值"], row["智驾主题均值"]) >= smart_threshold:
            base = "智能体验型"
        elif row["续航中位数"] >= global_range * 1.08:
            base = "长续航刚需型"
        else:
            base = "均衡家用型"
        used[base] = used.get(base, 0) + 1
        names[int(cluster_id)] = base if used[base] == 1 else f"{base}-{used[base]}"
    return names


def run_consumer_clustering(
    features: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = settings or load_settings()
    paths = get_paths(settings)
    if features is None:
        features = pd.read_csv(paths["feature_data"], encoding="utf-8-sig", low_memory=False)

    required = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"聚类所需字段缺失: {', '.join(missing)}")

    data = features.copy()
    preprocessor = _build_preprocessor()
    matrix = preprocessor.fit_transform(data[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])
    matrix = np.asarray(matrix, dtype=float)
    random_state = int(settings["random_state"])
    best_k, selection_metrics = _select_k(
        matrix,
        int(settings["cluster_min_k"]),
        int(settings["cluster_max_k"]),
        random_state,
    )
    model = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
    labels = model.fit_predict(matrix)
    data["cluster_id"] = labels

    names = _cluster_names(data, "cluster_id")
    data["消费者群体"] = data["cluster_id"].map(names)
    pca = PCA(n_components=2, random_state=random_state)
    coordinates = pca.fit_transform(matrix)
    result = data[
        [
            "doc_id",
            "品牌",
            "车型",
            "能源类型",
            "车辆级别",
            "购车地区",
            "购车价格_分析值",
            "续航_参数",
            "cluster_id",
            "消费者群体",
        ]
    ].copy()
    result["PCA1"] = coordinates[:, 0]
    result["PCA2"] = coordinates[:, 1]
    result.to_csv(paths["cluster_results"], index=False, encoding="utf-8-sig")

    summary = (
        data.groupby(["cluster_id", "消费者群体"])
        .agg(
            样本数=("doc_id", "count"),
            价格中位数=("购车价格_分析值", "median"),
            续航中位数=("续航_参数", "median"),
            电池容量中位数=("电池电量_参数", "median"),
            核心评分均值=("核心评分均值", "mean"),
            纯电占比=("是否纯电", "mean"),
            最关注主题=("主题标签", lambda values: values.value_counts().index[0]),
        )
        .reset_index()
    )
    summary.to_csv(
        paths["tables_dir"] / "consumer_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paths["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, paths["artifacts_dir"] / "cluster_preprocessor.joblib")
    joblib.dump(model, paths["artifacts_dir"] / "kmeans_model.joblib")
    metrics = {
        "best_k": best_k,
        "selected_silhouette": next(
            item["silhouette"] for item in selection_metrics if item["k"] == best_k
        ),
        "selection": selection_metrics,
        "cluster_names": names,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }
    write_json(paths["tables_dir"] / "cluster_metrics.json", metrics)
    LOGGER.info("消费者聚类完成: k=%d", best_k)
    return result, metrics

