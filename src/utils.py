from __future__ import annotations

import html
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<.*?>", re.DOTALL)
SPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")
BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
REGION_SUFFIX_PATTERN = re.compile(
    r"(特别行政区|维吾尔自治区|壮族自治区|回族自治区|自治区|省|市|地区|自治州)$"
)
VERSION_NOISE_PATTERN = re.compile(r"[\s\-—_（）()【】\[\]+/·,，。]")


def setup_logger(name: str = "nev-mining") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def clean_comment(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = SPACE_PATTERN.sub(" ", text)
    text = BLANK_LINE_PATTERN.sub("\n\n", text)
    return text.strip()


def parse_price(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not numbers:
        return np.nan
    number = float(numbers[0])
    if "万" in text:
        number *= 10000
    return number


def parse_first_number(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def normalize_region(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "未知"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "未知", "未填写"}:
        return "未知"
    text = REGION_SUFFIX_PATTERN.sub("", text)
    return text or "未知"


def normalize_version(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("km", "公里").replace("KM", "公里")
    return VERSION_NOISE_PATTERN.sub("", text)


def price_band(value: float, segments: list[float], labels: list[str]) -> str:
    if value is None or pd.isna(value):
        return "价格未知"
    for index, upper in enumerate(segments):
        if value < upper:
            return labels[index]
    return labels[-1]


def normalize_topic_scores(values: Iterable[float]) -> list[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        result.append(max(0.0, min(1.0, number)))
    return result


def top_counter_text(values: Iterable[str], limit: int = 10) -> str:
    counter = Counter(value for value in values if value)
    return "、".join(f"{name}({count})" for name, count in counter.most_common(limit))


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default)


def read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator

