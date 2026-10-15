from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_paths, load_settings
from src.utils import setup_logger


LOGGER = setup_logger()


def _write_table(
    connection: sqlite3.Connection,
    table_name: str,
    data: pd.DataFrame | None,
) -> None:
    if data is None:
        return
    data.to_sql(table_name, connection, if_exists="replace", index=False)


def build_database(
    processed_reviews: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    topic_results: pd.DataFrame | None = None,
    sentiment_results: pd.DataFrame | None = None,
    cluster_results: pd.DataFrame | None = None,
    rules: pd.DataFrame | None = None,
    report_tables: dict[str, pd.DataFrame] | None = None,
    settings: dict[str, Any] | None = None,
) -> Path:
    settings = settings or load_settings()
    paths = get_paths(settings)
    database_path = paths["database"]
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    try:
        _write_table(connection, "reviews_processed", processed_reviews)
        _write_table(connection, "review_features", features)
        _write_table(connection, "topic_probabilities", topic_results)
        _write_table(connection, "sentiment_results", sentiment_results)
        _write_table(connection, "consumer_clusters", cluster_results)
        _write_table(connection, "association_rules", rules)
        for table_name, table_data in (report_tables or {}).items():
            _write_table(connection, table_name, table_data)

        indexes = {
            "review_features": ["doc_id", "车系ID", "车型", "购车地区", "价格段"],
            "topic_probabilities": ["doc_id", "主导主题"],
            "sentiment_results": ["doc_id", "文本情感倾向"],
            "consumer_clusters": ["doc_id", "cluster_id", "消费者群体"],
        }
        for table_name, columns in indexes.items():
            table_exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if not table_exists:
                continue
            existing = {
                row[1]
                for row in connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            }
            for column in columns:
                if column in existing:
                    connection.execute(
                        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{column}" '
                        f'ON "{table_name}" ("{column}")'
                    )
        connection.commit()
    finally:
        connection.close()
    LOGGER.info("SQLite数据库生成: %s", database_path)
    return database_path

