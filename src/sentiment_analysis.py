from __future__ import annotations

import math
from collections import Counter
from typing import Any

import jieba
import numpy as np
import pandas as pd

from src.config import CONFIG_DIR, get_paths, load_json, load_settings
from src.utils import setup_logger, write_json


LOGGER = setup_logger()


def _sentiment_tokens(text: str) -> list[str]:
    tokens = []
    for token in jieba.lcut(str(text)):
        token = token.strip().lower()
        if not token:
            continue
        if any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in token):
            tokens.append(token)
    return tokens


def _score_document(
    tokens: list[str],
    positive: set[str],
    negative: set[str],
    negations: set[str],
    intensifiers: dict[str, float],
) -> tuple[float, int, int, list[str], list[str]]:
    positive_score = 0.0
    negative_score = 0.0
    positive_hits: list[str] = []
    negative_hits: list[str] = []

    for index, token in enumerate(tokens):
        direction = 1
        source_hits = None
        if token in positive:
            source_hits = positive_hits
        elif token in negative:
            direction = -1
            source_hits = negative_hits
        if source_hits is None:
            continue

        weight = 1.0
        negated = False
        for offset in range(1, 4):
            previous_index = index - offset
            if previous_index < 0:
                break
            previous = tokens[previous_index]
            if previous in intensifiers:
                weight *= intensifiers[previous]
            if previous in negations:
                negated = not negated

        contribution = direction * weight * (-1 if negated else 1)
        if contribution >= 0:
            positive_score += contribution
        else:
            negative_score += abs(contribution)
        source_hits.append(token)

    total = positive_score + negative_score
    polarity = 0.0 if total == 0 else (positive_score - negative_score) / math.sqrt(total + 1)
    polarity = max(-1.0, min(1.0, polarity))
    return polarity, len(positive_hits), len(negative_hits), positive_hits, negative_hits


def _rating_label(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "无评分"
    if score >= 4:
        return "正面"
    if score < 3:
        return "负面"
    return "中性"


def run_sentiment_analysis(
    features: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    paths = get_paths(settings)
    if features is None:
        features = pd.read_csv(paths["feature_data"], encoding="utf-8-sig", low_memory=False)

    lexicon = load_json(CONFIG_DIR / "sentiment_lexicon.json")
    positive = {item.lower() for item in lexicon["positive"]}
    negative = {item.lower() for item in lexicon["negative"]}
    negations = {item.lower() for item in lexicon["negations"]}
    intensifiers = {key.lower(): float(value) for key, value in lexicon["intensifiers"].items()}
    for term in positive | negative | negations | set(intensifiers):
        jieba.add_word(term)

    rows = []
    positive_counter: Counter[str] = Counter()
    negative_counter: Counter[str] = Counter()
    for row in features[["doc_id", "评论内容", "综合评分"]].itertuples(index=False):
        tokens = _sentiment_tokens(row.评论内容)
        polarity, pos_count, neg_count, pos_hits, neg_hits = _score_document(
            tokens,
            positive,
            negative,
            negations,
            intensifiers,
        )
        if polarity >= 0.12:
            text_label = "正面"
        elif polarity <= -0.12:
            text_label = "负面"
        else:
            text_label = "中性"
        positive_counter.update(pos_hits)
        negative_counter.update(neg_hits)
        rows.append(
            {
                "doc_id": int(row.doc_id),
                "文本情感分": round(polarity, 6),
                "文本正词数": pos_count,
                "文本负词数": neg_count,
                "文本情感倾向": text_label,
                "评分情感倾向": _rating_label(row.综合评分),
            }
        )

    result = pd.DataFrame(rows)
    comparable = result[
        result["评分情感倾向"].isin(["正面", "负面", "中性"])
        & result["文本情感倾向"].isin(["正面", "负面", "中性"])
    ]
    agreement = (
        comparable["文本情感倾向"].eq(comparable["评分情感倾向"]).mean()
        if len(comparable)
        else np.nan
    )
    metrics = {
        "documents": int(len(result)),
        "text_distribution": result["文本情感倾向"].value_counts().to_dict(),
        "rating_distribution": result["评分情感倾向"].value_counts().to_dict(),
        "agreement_rate": None if pd.isna(agreement) else float(agreement),
        "top_positive_terms": positive_counter.most_common(20),
        "top_negative_terms": negative_counter.most_common(20),
    }
    result.to_csv(paths["sentiment_results"], index=False, encoding="utf-8-sig")
    write_json(paths["tables_dir"] / "sentiment_metrics.json", metrics)
    LOGGER.info("文本情感分析完成: 评分-文本一致率 %.2f%%", float(agreement) * 100)
    return result

