from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_settings() -> dict[str, Any]:
    return load_json(CONFIG_DIR / "settings.json")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_project_dirs(settings: dict[str, Any] | None = None) -> None:
    settings = settings or load_settings()
    directory_keys = [
        "artifacts_dir",
        "tables_dir",
        "figures_dir",
        "reports_dir",
    ]
    file_keys = [
        "processed_reviews",
        "feature_data",
        "database",
        "topic_probabilities",
        "sentiment_results",
        "cluster_results",
        "association_rules",
    ]
    for key in directory_keys:
        resolve_path(settings[key]).mkdir(parents=True, exist_ok=True)
    for key in file_keys:
        resolve_path(settings[key]).parent.mkdir(parents=True, exist_ok=True)


def get_paths(settings: dict[str, Any] | None = None) -> dict[str, Path]:
    settings = settings or load_settings()
    file_like_keys = {
        "raw_reviews",
        "series",
        "models",
        "precleaned_reviews",
        "precleaned_series",
        "precleaned_models",
        "processed_reviews",
        "feature_data",
        "database",
        "topic_probabilities",
        "sentiment_results",
        "cluster_results",
        "association_rules",
        "artifacts_dir",
        "tables_dir",
        "figures_dir",
        "reports_dir",
    }
    # 用 in 过滤而不是直接取：settings.json 可能不含预清洗数据配置，
    # 此时不能让整条流水线因缺键崩溃
    return {
        key: resolve_path(settings[key])
        for key in file_like_keys
        if key in settings and settings[key]
    }

