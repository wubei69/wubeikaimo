def test_core_modules_import() -> None:
    from src import (  # noqa: F401
        association_rules,
        consumer_clustering,
        data_preprocessing,
        database,
        feature_engineering,
        reporting,
        sentiment_analysis,
        topic_mining,
    )

