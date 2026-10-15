from __future__ import annotations

from collections import Counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import CONFIG_DIR, get_paths, load_json, load_settings
from src.utils import setup_logger, write_json


LOGGER = setup_logger()


def _topic_label(topic_words: list[str], topics: dict[str, list[str]]) -> str:
    best_name = "综合讨论"
    best_overlap = 0
    word_set = set(topic_words)
    for name, terms in topics.items():
        overlap = len(word_set.intersection(term.lower() for term in terms))
        if overlap > best_overlap:
            best_name = name
            best_overlap = overlap
    return best_name


def run_topic_mining(
    features: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = settings or load_settings()
    paths = get_paths(settings)
    if features is None:
        features = pd.read_csv(paths["feature_data"], encoding="utf-8-sig", low_memory=False)

    features["doc_id"] = features["doc_id"].astype(int)
    corpus = features["分词结果"].fillna("").astype(str)
    topic_count = int(settings["topic_count"])
    random_state = int(settings["random_state"])

    count_vectorizer = CountVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        max_features=6000,
        min_df=5,
        max_df=0.85,
    )
    count_matrix = count_vectorizer.fit_transform(corpus)
    feature_names = np.array(count_vectorizer.get_feature_names_out())

    model = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=random_state,
        learning_method="batch",
        max_iter=20,
        evaluate_every=5,
        n_jobs=-1,
    )
    document_topic = model.fit_transform(count_matrix)
    perplexity = float(model.perplexity(count_matrix))

    topics = load_json(CONFIG_DIR / "topics.json")
    topic_words: dict[str, list[str]] = {}
    topic_names: dict[str, str] = {}
    used_names: Counter[str] = Counter()
    for topic_index, component in enumerate(model.components_):
        indices = component.argsort()[-25:][::-1]
        words = feature_names[indices].tolist()
        key = f"主题{topic_index + 1}"
        topic_words[key] = words
        base_name = _topic_label(words, topics)
        used_names[base_name] += 1
        suffix = "" if used_names[base_name] == 1 else f"-{used_names[base_name]}"
        topic_names[key] = f"{base_name}{suffix}"

    topic_df = pd.DataFrame(
        document_topic,
        columns=[topic_names[f"主题{index + 1}"] for index in range(topic_count)],
    )
    probability_columns = topic_df.columns.tolist()
    topic_df.insert(0, "doc_id", features["doc_id"].to_numpy())
    topic_df["主导主题"] = topic_df[probability_columns].idxmax(axis=1)
    topic_df["主导主题概率"] = topic_df[probability_columns].max(axis=1).round(6)
    for column in probability_columns:
        topic_df[column] = topic_df[column].round(6)
    topic_df.to_csv(paths["topic_probabilities"], index=False, encoding="utf-8-sig")

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        max_features=6000,
        min_df=5,
        max_df=0.85,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    tfidf_terms = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    keyword_table = pd.DataFrame(
        {
            "关键词": tfidf_terms,
            "TF-IDF均值": tfidf_scores,
        }
    ).sort_values("TF-IDF均值", ascending=False)
    keyword_table.to_csv(
        paths["tables_dir"] / "tfidf_keywords.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paths["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths["artifacts_dir"] / "lda_model.joblib")
    joblib.dump(count_vectorizer, paths["artifacts_dir"] / "lda_vectorizer.joblib")
    metrics = {
        "perplexity": perplexity,
        "topic_count": topic_count,
        "vocabulary_size": int(len(feature_names)),
        "topic_names": topic_names,
        "topic_top_words": topic_words,
    }
    write_json(paths["tables_dir"] / "topic_model_metrics.json", metrics)
    LOGGER.info("LDA主题模型完成: %d个主题, 困惑度 %.2f", topic_count, perplexity)
    return topic_df, metrics

