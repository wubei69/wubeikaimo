"""把 caiji 目录已清洗数据接入分析流水线。

背景
----
caiji 目录已经完成了去重、价格解析、缺失值标记和文本清洗（见
``数据清洗_评论.py`` / ``处理缺失值.py`` / ``构建分析表.py``）。
本模块只做「读盘 + 列名对齐」，不重复清洗，随后交给
``data_preprocessing.preprocess_data`` 复用既有的评分、价格、
地区、车系与参数关联逻辑。

为什么要单独一层
----------------
旧流水线读的是 12 车系原始数据，新数据是 49 车系、38636 条评论，
字段命名不同（例如价格已解析为 ``购车价格_元``，文本分成
``正文_清洗`` / ``正文_纯文本`` 两版）。硬改清洗模块会牵动全部测试，
用适配层可以把差异收在一个文件里。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_paths, load_settings
from src.utils import setup_logger


LOGGER = setup_logger()


# 接口对同一能源类型有多种写法，不归一化会被聚类当成两个类别
ENERGY_ALIASES = {
    "增程式电动": "增程式",
    "增程电动": "增程式",
    "增程": "增程式",
    "纯电": "纯电动",
    "插电混动": "插电式混合动力",
    "插电式混动": "插电式混合动力",
    "插混": "插电式混合动力",
    "油电混合": "混合动力",
}


# 用于从评论语义推断能源类型（仅在接口无参数时启用）
PHEV_PATTERN = r"插混|插电|混动|DM-i|dm-i|油耗|亏电|馈电|发动机介入"
EREV_PATTERN = r"增程|增程器|发电机|油发电|烧油发电"
# 相对优势倍数：低于此值不推断，宁可留未知也不猜
EREV_DOMINANCE = 1.2


def _normalize_energy(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: ENERGY_ALIASES.get(str(value).strip(), value)
        if pd.notna(value)
        else value
    )


# 新数据列名 -> 流水线内部列名
# 注意：购车价格_元 要覆盖到「购车价格」上，必须先把原始字符串列挪走，
# 否则 rename 后会得到两个同名列，后续 result["购车价格"] 返回 DataFrame
REVIEW_RENAME = {
    "购买版本": "车型版本",
    "车型年份": "年款",
}


def _read(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        LOGGER.warning("%s 不存在: %s", label, path)
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    LOGGER.info("读取 %s: %d 行 x %d 列", label, len(frame), len(frame.columns))
    return frame


def _patch_series_from_models(
    series: pd.DataFrame,
    models: pd.DataFrame | None,
) -> pd.DataFrame:
    """用车型参数表补齐车系表的级别与能源类型。

    car_series_扩展.csv 只覆盖 46 个车系，主表有 49 个。
    级别和能源类型是聚类与关联规则的分组变量，缺了会把样本
    全部归入「未知」，因此从车型参数表按车系取众数补上。
    """
    if models is None or series is None or series.empty:
        return series

    needed = {"级别", "能源类型"}
    if not needed.issubset(models.columns):
        return series

    result = series.copy()
    for column in needed:
        if column not in result.columns:
            result[column] = pd.NA

    lookup = models.dropna(subset=["所属车系ID"]).copy()
    lookup["所属车系ID"] = pd.to_numeric(lookup["所属车系ID"], errors="coerce")
    lookup = lookup.dropna(subset=["所属车系ID"])
    if lookup.empty:
        return result

    # 同一车系下各版本参数一致，取出现次数最多的值即可
    modes = lookup.groupby("所属车系ID")[list(needed)].agg(
        lambda values: values.dropna().iloc[0] if values.dropna().any() else pd.NA
    )
    result["车系ID"] = pd.to_numeric(result["车系ID"], errors="coerce")
    for column in needed:
        mapping = modes[column].to_dict()
        fill_values = result["车系ID"].map(mapping)
        result[column] = result[column].fillna(fill_values)

    filled = int(result["级别"].notna().sum())
    LOGGER.info("车系表补齐后: %d 个车系含级别信息", filled)
    return result


def _infer_energy_from_reviews(
    reviews: pd.DataFrame,
    series: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """对车系表缺能源类型的车系，用评论语义推断。

    宋PLUS DM、银河L7、哪吒L 三个车系的车型参数接口已返回空（停售），
    车系表里也就没有级别与能源类型。能源类型是聚类和关联规则的分组变量，
    整块缺失会把 3011 条评论（7.8%）全部归入「未知」。

    这里不凭车系名猜测，而是看评论里车主自己怎么说：
    插混车主会谈油耗、亏电、发动机介入；增程车主会谈增程器、烧油发电。
    两者互斥且区分度高，取相对优势明显的一方作为推断结果。
    """
    result = series.copy()
    target_ids = (
        result.loc[result["能源类型"].isna(), "车系ID"]
        .dropna()
        .astype("int64")
        .tolist()
    )
    if not target_ids:
        return result, []

    text = reviews.get("评论内容", pd.Series(dtype=str)).fillna("").astype(str)
    review_ids = pd.to_numeric(reviews["车系ID"], errors="coerce")
    evidence: list[dict[str, Any]] = []

    for series_id in target_ids:
        mask = review_ids.eq(series_id)
        subset = text[mask]
        if subset.empty:
            continue
        phev_hits = int(subset.str.contains(PHEV_PATTERN, regex=True).sum())
        erev_hits = int(subset.str.contains(EREV_PATTERN, regex=True).sum())
        total = int(len(subset))

        # 要求相对优势明显，否则宁可留未知
        if erev_hits > phev_hits * EREV_DOMINANCE:
            inferred = "增程式"
        elif phev_hits > erev_hits * EREV_DOMINANCE:
            inferred = "插电式混合动力"
        else:
            inferred = None

        if inferred is None:
            continue
        result.loc[result["车系ID"] == series_id, "能源类型"] = inferred
        row = {
            "车系ID": int(series_id),
            "车系名": reviews.loc[mask, "车型"].iloc[0],
            "评论数": total,
            "插混线索数": phev_hits,
            "增程线索数": erev_hits,
            "推断能源类型": inferred,
            "推断依据": "评论语义（接口无车型参数，车系已停售）",
        }
        evidence.append(row)
        LOGGER.warning(
            "车系 %s 接口无参数，按评论语义推断能源类型=%s（插混%d / 增程%d）",
            row["车系名"], inferred, phev_hits, erev_hits,
        )

    return result, evidence


def _ensure_series_coverage(
    series: pd.DataFrame | None,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """车系表缺车系时，用评论表自身的品牌/车型回填，保证可左连接。"""
    if series is None:
        series = pd.DataFrame(columns=["车系ID", "品牌", "车系名", "级别", "能源类型"])

    result = series.copy()
    for column in ["车系ID", "品牌", "车系名", "级别", "能源类型"]:
        if column not in result.columns:
            result[column] = pd.NA

    known = set(pd.to_numeric(result["车系ID"], errors="coerce").dropna().astype("int64"))
    review_ids = pd.to_numeric(reviews["车系ID"], errors="coerce").dropna().astype("int64")
    missing_ids = sorted(set(review_ids) - known)
    if not missing_ids:
        return result

    extra = (
        reviews.loc[reviews["车系ID"].isin(missing_ids), ["车系ID", "品牌", "车型"]]
        .drop_duplicates(subset=["车系ID"])
        .rename(columns={"车型": "车系名"})
    )
    extra["级别"] = pd.NA
    extra["能源类型"] = pd.NA
    LOGGER.warning("车系表缺少 %d 个车系，已从评论表回填品牌/车系名", len(extra))
    return pd.concat([result, extra], ignore_index=True)


def load_precleaned_frames(
    settings: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """返回 (评论表, 车系表, 车型参数表)，列名已对齐流水线内部约定。"""
    settings = settings or load_settings()
    paths = get_paths(settings)

    reviews_path = paths.get("precleaned_reviews")
    series_path = paths.get("precleaned_series")
    models_path = paths.get("precleaned_models")

    if reviews_path is None:
        raise ValueError("settings 缺少 precleaned_reviews 配置")

    reviews = _read(reviews_path, "评论主表")
    if reviews is None or reviews.empty:
        return None, None, None

    series = _read(series_path, "车系表") if series_path else None
    models = _read(models_path, "车型参数表") if models_path else None

    # --- 评论表对齐 ---
    reviews = reviews.rename(columns=REVIEW_RENAME)
    if "购车价格_元" in reviews.columns:
        if "购车价格" in reviews.columns:
            reviews = reviews.rename(columns={"购车价格": "购车价格_原始文本"})
        reviews = reviews.rename(columns={"购车价格_元": "购车价格"})

    # 文本优先用去掉【】模板标签的版本，LDA 与情感分析都不该吃模板词
    if "正文_纯文本" in reviews.columns:
        plain = reviews["正文_纯文本"].fillna("").astype(str).str.strip()
        cleaned = reviews.get("正文_清洗", reviews["评论内容"]).fillna("").astype(str)
        reviews["评论内容_含标签"] = cleaned
        reviews["评论内容"] = plain.where(plain.str.len() > 0, cleaned)
    elif "正文_清洗" in reviews.columns:
        reviews["评论内容"] = reviews["正文_清洗"].fillna("").astype(str)

    if "购车价格" not in reviews.columns and "购车价格_元" in reviews.columns:
        reviews["购车价格"] = reviews["购车价格_元"]

    # 旧流水线的车型版本字段用于参数匹配
    if "车型版本" not in reviews.columns and "购买版本" in reviews.columns:
        reviews["车型版本"] = reviews["购买版本"]

    # --- 车系表对齐 ---
    for frame in (models, series):
        if frame is not None and "能源类型" in frame.columns:
            frame["能源类型"] = _normalize_energy(frame["能源类型"])

    if series is not None:
        series = _patch_series_from_models(series, models)
    series = _ensure_series_coverage(series, reviews)
    series, energy_evidence = _infer_energy_from_reviews(reviews, series)
    if energy_evidence:
        _write_energy_evidence(energy_evidence, settings)

    LOGGER.info(
        "适配完成: 评论 %d 行, 车系 %d 个, 车型参数 %s 行",
        len(reviews),
        len(series),
        "0" if models is None else len(models),
    )
    return reviews, series, models


def _write_energy_evidence(
    evidence: list[dict[str, Any]],
    settings: dict[str, Any],
) -> None:
    """把推断依据落盘，论文「数据预处理」章节可直接引用。"""
    try:
        paths = get_paths(settings)
        target = Path(paths["tables_dir"]) / "energy_type_inference.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("能源类型推断依据已写入: %s", target)
    except Exception as error:  # noqa: BLE001 - 落盘失败不应中断分析
        LOGGER.warning("能源类型推断依据写入失败: %s", error)


def has_precleaned(settings: dict[str, Any] | None = None) -> bool:
    settings = settings or load_settings()
    paths = get_paths(settings)
    path = paths.get("precleaned_reviews")
    return bool(settings.get("use_precleaned")) and path is not None and path.exists()
