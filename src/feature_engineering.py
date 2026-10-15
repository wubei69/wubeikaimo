from __future__ import annotations

import math
from collections import Counter
from typing import Any

import jieba
import numpy as np
import pandas as pd

from src.config import CONFIG_DIR, get_paths, load_json, load_settings
from src.utils import price_band, read_text_lines, setup_logger


LOGGER = setup_logger()


def _load_topics() -> dict[str, list[str]]:
    return load_json(CONFIG_DIR / "topics.json")


def _load_stopwords() -> set[str]:
    return set(read_text_lines(CONFIG_DIR / "stopwords_zh.txt"))


def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    tokens = []
    for token in jieba.lcut(str(text)):
        token = token.strip().lower()
        if len(token) <= 1 or token in stopwords:
            continue
        if not any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in token):
            continue
        tokens.append(token)
    return tokens


def _topic_metrics(
    tokens: list[str],
    token_text: str,
    topics: dict[str, list[str]],
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    token_counts = Counter(token_text.split())
    token_count = max(1, len(tokens))
    scores: dict[str, float] = {}
    hits: dict[str, int] = {}
    present: list[str] = []
    for topic, terms in topics.items():
        hit_count = sum(token_counts[term.lower()] for term in terms)
        hits[topic] = hit_count
        score = math.log1p(hit_count) / math.log1p(token_count)
        scores[f"主题_{topic}"] = round(min(1.0, score), 6)
        if hit_count > 0:
            present.append(topic)
    return scores, hits, present


def engineer_features(
    reviews: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    paths = get_paths(settings)
    if reviews is None:
        reviews = pd.read_csv(paths["processed_reviews"], encoding="utf-8-sig", low_memory=False)

    data = reviews.copy()
    data["doc_id"] = data["doc_id"].astype(int)
    data["评论长度"] = data["评论内容"].astype(str).str.len()
    data["购车价格_万元"] = (data["购车价格_分析值"] / 10000).round(2)
    data["价格段"] = data["购车价格_分析值"].map(
        lambda value: price_band(
            value,
            settings["price_segments"],
            settings["price_segment_labels"],
        )
    )
    data["购车年份"] = data["购车时间"].dt.year.astype("Int64")
    data["评论年份"] = data["评论时间"].dt.year.astype("Int64")
    data["购买到评论天数_近似"] = (
        data["评论时间"] - data["购车时间"]
    ).dt.total_seconds() / 86400
    data.loc[data["购买到评论天数_近似"] < -31, "购买到评论天数_近似"] = np.nan
    data["购买到评论天数_近似"] = data["购买到评论天数_近似"].clip(lower=0)
    data["是否纯电"] = data["能源类型"].fillna("").str.contains("纯电动")
    data["高满意度"] = data["综合评分"].ge(4.5)

    jieba.initialize()
    for topic, terms in _load_topics().items():
        for term in terms:
            jieba.add_word(term)
    stopwords = _load_stopwords()
    topics = _load_topics()

    token_lists: list[list[str]] = []
    token_texts: list[str] = []
    top_keywords: list[str] = []
    for text in data["评论内容"].astype(str):
        tokens = _tokenize(text, stopwords)
        token_lists.append(tokens)
        token_texts.append(" ".join(tokens))
        top_keywords.append("、".join(token for token, _ in Counter(tokens).most_common(8)))

    topic_rows = []
    hit_rows = []
    topic_labels = []
    for tokens, token_text in zip(token_lists, token_texts, strict=True):
        scores, hits, present = _topic_metrics(tokens, token_text, topics)
        topic_rows.append(scores)
        hit_rows.append({f"主题命中_{key}": value for key, value in hits.items()})
        topic_labels.append("、".join(present) if present else "其他")

    data["分词结果"] = token_texts
    data["关键词"] = top_keywords
    data["词数"] = [len(tokens) for tokens in token_lists]
    data = pd.concat([data, pd.DataFrame(topic_rows, index=data.index)], axis=1)
    data = pd.concat([data, pd.DataFrame(hit_rows, index=data.index)], axis=1)
    data["主题标签"] = topic_labels
    data["主题数量"] = data["主题标签"].map(
        lambda value: 0 if value == "其他" else len(value.split("、"))
    )

    data.to_csv(paths["feature_data"], index=False, encoding="utf-8-sig")
    LOGGER.info("特征工程完成: %d 行, %d 列", len(data), len(data.columns))
    return data

