from __future__ import annotations

from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.config import get_paths, load_settings
from src.utils import setup_logger, write_json


LOGGER = setup_logger()


# 缺失值占位符。这些不是真实取值，放进事务会挖出伪规则
UNKNOWN_TOKENS = {"未知", "nan", "None", "null", "", "未填写"}

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


def _frozenset_text(value: frozenset[Any]) -> str:
    return " + ".join(sorted(str(item) for item in value))


def run_association_rules(
    features: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    paths = get_paths(settings)
    if features is None:
        features = pd.read_csv(paths["feature_data"], encoding="utf-8-sig", low_memory=False)

    data = features.copy()
    transactions = []
    top_regions = set(data["购车地区"].value_counts().head(12).index)
    for row in data.itertuples(index=False):
        items = set()
        # 「未知」是缺失值的占位符，不是真实取值。
        # 若把它当成一个普通类别放进事务，Apriori 会挖出一堆
        # 「能源类型=未知 → 车辆级别=未知」的伪规则（lift 高达 12.8），
        # 本质只是「这两条记录同一个字段没填」，对消费特征毫无解释力。
        for label, value in (
            ("价格段", row.价格段),
            ("能源类型", row.能源类型),
            ("车辆级别", row.车辆级别),
        ):
            if pd.isna(value):
                continue
            text = str(value).strip()
            if not text or text in UNKNOWN_TOKENS:
                continue
            items.add(f"{label}={text}")

        region = row.购车地区 if row.购车地区 in top_regions else "其他地区"
        if pd.notna(region) and str(region).strip() not in UNKNOWN_TOKENS:
            items.add(f"地区={region}")
        if pd.notna(row.综合评分):
            if row.综合评分 >= 4.5:
                items.add("满意度=高")
            elif row.综合评分 < 3.5:
                items.add("满意度=低")
            else:
                items.add("满意度=中")
        for topic_column in TOPIC_COLUMNS:
            topic_name = topic_column.removeprefix("主题_")
            if getattr(row, topic_column) > 0:
                items.add(f"偏好={topic_name}")
        transactions.append(items)

    item_names = sorted(set().union(*transactions))
    encoded = pd.DataFrame(False, index=range(len(transactions)), columns=item_names, dtype=bool)
    for row_index, transaction in enumerate(transactions):
        encoded.loc[row_index, list(transaction)] = True
    frequent_items = apriori(
        encoded,
        min_support=0.03,
        use_colnames=True,
        max_len=3,
        low_memory=True,
    )
    if frequent_items.empty:
        result = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
    else:
        result = association_rules(
            frequent_items,
            metric="confidence",
            min_threshold=0.6,
            num_itemsets=len(frequent_items),
        )
        result = result[
            (result["lift"] >= 1.1)
            & (result["consequents"].map(len) == 1)
            & (result["antecedents"].map(len) <= 2)
        ].sort_values(
            ["lift", "confidence", "support"], ascending=False
        )
        result["前件"] = result["antecedents"].map(_frozenset_text)
        result["后件"] = result["consequents"].map(_frozenset_text)
        result = result[
            ["前件", "后件", "support", "confidence", "lift", "antecedent support"]
        ]

    result.to_csv(paths["association_rules"], index=False, encoding="utf-8-sig")
    write_json(
        paths["tables_dir"] / "association_metrics.json",
        {
            "transactions": int(len(encoded)),
            "items": int(encoded.shape[1]),
            "rules": int(len(result)),
            "min_support": 0.03,
            "min_confidence": 0.6,
            "min_lift": 1.1,
            "max_antecedent_items": 2,
            "single_consequent": True,
        },
    )
    LOGGER.info("关联规则完成: %d 条规则", len(result))
    return result

