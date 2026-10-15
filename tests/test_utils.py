from __future__ import annotations

import numpy as np

from src.utils import clean_comment, normalize_region, normalize_version, parse_price


def test_parse_price_variants() -> None:
    assert parse_price("6.58万") == 65800
    assert parse_price("￥120,000") == 120000
    assert parse_price(159800) == 159800
    assert np.isnan(parse_price("价格待定"))


def test_clean_comment_keeps_punctuation() -> None:
    text = "<p>空间不错，但是  续航一般。</p> https://example.com"
    result = clean_comment(text)
    assert "空间不错，但是" in result
    assert "。" in result
    assert "http" not in result


def test_normalize_region_and_version() -> None:
    assert normalize_region("重庆市") == "重庆"
    assert normalize_region("未知") == "未知"
    assert normalize_version("410km 探索+版") == "410公里探索版"

