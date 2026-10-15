from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.association_rules import run_association_rules
from src.config import ensure_project_dirs, get_paths, load_settings
from src.consumer_clustering import run_consumer_clustering
from src.data_preprocessing import preprocess_data
from src.database import build_database
from src.feature_engineering import engineer_features
from src.reporting import generate_reports
from src.sentiment_analysis import run_sentiment_analysis
from src.topic_mining import run_topic_mining
from src.utils import json_default, setup_logger


LOGGER = setup_logger()


def _load_if_available(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    return None


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_settings()
    ensure_project_dirs(settings)
    paths = get_paths(settings)

    LOGGER.info("阶段 1/7：数据清洗")
    processed_reviews = preprocess_data(settings)

    LOGGER.info("阶段 2/7：特征工程")
    features = engineer_features(processed_reviews, settings)

    LOGGER.info("阶段 3/7：文本情感分析")
    sentiment_results = run_sentiment_analysis(features, settings)
    features = features.merge(
        sentiment_results[
            ["doc_id", "文本情感分", "文本情感倾向", "评分情感倾向"]
        ],
        on="doc_id",
        how="left",
    )
    features.to_csv(paths["feature_data"], index=False, encoding="utf-8-sig")

    if args.skip_topics:
        LOGGER.info("阶段 4/7：跳过LDA主题模型")
        topic_results = _load_if_available(paths["topic_probabilities"])
    else:
        LOGGER.info("阶段 4/7：LDA主题模型")
        topic_results, _ = run_topic_mining(features, settings)
    if topic_results is not None:
        features = features.merge(
            topic_results[["doc_id", "主导主题", "主导主题概率"]],
            on="doc_id",
            how="left",
        )
        features.to_csv(paths["feature_data"], index=False, encoding="utf-8-sig")

    if args.skip_clustering:
        LOGGER.info("阶段 5/7：跳过消费者聚类")
        cluster_results = _load_if_available(paths["cluster_results"])
    else:
        LOGGER.info("阶段 5/7：消费者聚类")
        cluster_results, _ = run_consumer_clustering(features, settings)

    if args.skip_association:
        LOGGER.info("阶段 6/7：跳过关联规则")
        rules = _load_if_available(paths["association_rules"])
    else:
        LOGGER.info("阶段 6/7：关联规则")
        rules = run_association_rules(features, settings)

    LOGGER.info("阶段 7/7：报告、图表和数据库")
    report_tables = generate_reports(
        features,
        topic_results,
        sentiment_results,
        cluster_results,
        rules,
        settings,
    )
    database_path = build_database(
        processed_reviews,
        features,
        topic_results,
        sentiment_results,
        cluster_results,
        rules,
        report_tables,
        settings,
    )

    manifest = {
        "rows": int(len(features)),
        "series_count": int(features["车型"].nunique()),
        "brand_count": int(features["品牌"].nunique()),
        "sentiment_documents": int(len(sentiment_results)),
        "cluster_documents": 0 if cluster_results is None else int(len(cluster_results)),
        "association_rules": 0 if rules is None else int(len(rules)),
        "database": str(database_path),
        "report": str(paths["reports_dir"] / "analysis_report.md"),
    }
    manifest_path = paths["tables_dir"] / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    LOGGER.info("流水线执行完成")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="新能源汽车消费特征挖掘流水线")
    parser.add_argument("--skip-topics", action="store_true", help="跳过LDA主题模型")
    parser.add_argument("--skip-clustering", action="store_true", help="跳过消费者聚类")
    parser.add_argument("--skip-association", action="store_true", help="跳过关联规则")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())

