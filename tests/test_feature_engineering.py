from __future__ import annotations

from src.feature_engineering import _topic_metrics


def test_topic_metrics_detects_present_topic() -> None:
    tokens = ["续航", "充电", "很快", "空间", "宽敞"]
    scores, hits, present = _topic_metrics(
        tokens,
        " ".join(tokens),
        {
            "续航充电": ["续航", "充电"],
            "空间实用": ["空间", "宽敞"],
        },
    )
    assert scores["主题_续航充电"] > 0
    assert hits["空间实用"] == 2
    assert present == ["续航充电", "空间实用"]

