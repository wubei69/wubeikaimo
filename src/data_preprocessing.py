from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, get_paths, load_settings
from src.utils import (
    clean_comment,
    normalize_region,
    normalize_version,
    parse_first_number,
    parse_price,
    setup_logger,
    write_json,
)


LOGGER = setup_logger()

SCORE_COLUMNS = [
    "综合评分",
    "外观评分",
    "空间评分",
    "动力评分",
    "舒适评分",
    "性价比评分",
    "内饰评分",
    "操控评分",
    "续航评分",
    "辅助驾驶评分",
    "智能座舱评分",
    "动力控制评分",
    "油耗评分",
    "驾驶评分",
]

CORE_SCORE_COLUMNS = [
    "综合评分",
    "外观评分",
    "空间评分",
    "性价比评分",
    "内饰评分",
    "续航评分",
]

MODEL_NUMERIC_COLUMNS = {
    "官方指导价": "指导价_参数",
    "百公里加速": "百公里加速_参数",
    "续航": "续航_参数",
    "电池电量": "电池电量_参数",
    "电机功率": "电机功率_参数",
    "电机扭矩": "电机扭矩_参数",
    "最高时速": "最高时速_参数",
}

MODEL_PRICE_COLUMNS = {"官方指导价"}

PARAMETER_COLUMNS = list(MODEL_NUMERIC_COLUMNS.values()) + [
    "电池种类_参数",
    "驱动方式_参数",
    "前悬架形式_参数",
    "后悬架形式_参数",
    "电机类型_参数",
]

NUMERIC_PARAMETER_COLUMNS = list(MODEL_NUMERIC_COLUMNS.values())
CATEGORICAL_PARAMETER_COLUMNS = [
    column for column in PARAMETER_COLUMNS if column not in NUMERIC_PARAMETER_COLUMNS
]

REQUIRED_REVIEW_COLUMNS = ["车系ID", "评论内容"]
REQUIRED_SERIES_COLUMNS = ["车系ID", "品牌", "车系名", "级别", "能源类型"]
REQUIRED_MODEL_COLUMNS = [
    "所属车系ID",
    "车型版本",
    "官方指导价",
    "百公里加速",
    "续航",
    "电池电量",
    "电机功率",
    "电机扭矩",
    "最高时速",
]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_SELF_CHECK_MESSAGE = f"""
未找到可清洗的评论数据，程序已停止，未生成空结果或随机数据。

请检查以下位置和文件：
1. 配置的数据文件是否存在且非空。
2. 若使用默认数据目录，请将懂车帝评论 CSV 放入：{RAW_DATA_DIR}
3. 评论文件至少包含字段：车系ID、评论内容。
4. 车型参数文件至少包含：所属车系ID、车型版本。
5. 车系文件至少包含：车系ID、品牌、车系名、级别、能源类型。

如果数据来自采集脚本，请先运行采集脚本并确认 CSV 表头、编码和记录数。
""".strip()


class DataInputError(RuntimeError):
    """Raised when source data is missing, empty or structurally invalid."""


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_input_files(role: str) -> list[Path]:
    if not RAW_DATA_DIR.exists():
        return []
    keywords = {
        "reviews": ("评论", "review", "comment"),
        "series": ("车系", "series"),
        "models": ("车型", "model"),
    }[role]
    return sorted(
        path
        for path in RAW_DATA_DIR.glob("*.csv")
        if any(keyword.lower() in path.name.lower() for keyword in keywords)
    )


def _resolve_input_path(configured_path: Path, role: str) -> Path:
    if configured_path.exists() and configured_path.stat().st_size > 0:
        return configured_path

    candidates = _candidate_input_files(role)
    if candidates:
        LOGGER.warning("配置路径不可用，改用 data/raw 候选文件: %s", candidates[0])
        return candidates[0]

    raise DataInputError(DATA_SELF_CHECK_MESSAGE)


def _read_csv(path: Path, role: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DataInputError(
            f"{role} 文件为空：{path}\n\n{DATA_SELF_CHECK_MESSAGE}"
        ) from exc
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataInputError(f"无法读取 {role} 文件：{path}\n原因：{exc}") from exc

    if data.empty:
        raise DataInputError(f"{role} 文件只有表头或没有记录：{path}\n\n{DATA_SELF_CHECK_MESSAGE}")
    return data


def _require_columns(data: pd.DataFrame, columns: list[str], role: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise DataInputError(
            f"{role} 缺少必需字段：{'、'.join(missing)}\n\n{DATA_SELF_CHECK_MESSAGE}"
        )


def _ensure_column(
    data: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in data.columns:
        data[column] = default


def _deduplicate_reviews(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    result = data.copy()
    result["原始行号"] = np.arange(2, len(result) + 2, dtype=int)

    exact_duplicate_count = int(result.duplicated(keep="first").sum())
    result = result.drop_duplicates(keep="first").copy()

    return result, {
        "exact_duplicate_rows": exact_duplicate_count,
        "business_duplicate_rows": 0,
    }


def _deduplicate_clean_reviews(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = data.copy()
    key_columns = ["车系ID", "评论内容"]
    if "用户" in result.columns:
        key_columns.append("用户")
    elif "评论时间" in result.columns:
        key_columns.append("评论时间")

    existing_keys = [column for column in key_columns if column in result.columns]
    duplicate_count = int(result.duplicated(subset=existing_keys, keep="first").sum())
    result = result.drop_duplicates(subset=existing_keys, keep="first").copy()
    return result, duplicate_count


def _clean_text(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = data.copy()
    raw_text = result["评论内容"].fillna("").astype(str)
    result["评论内容_原始"] = raw_text
    result["评论内容"] = raw_text.map(clean_comment)
    result["文本清洗变化"] = result["评论内容_原始"] != result["评论内容"]
    result["文本清洗损失字符数"] = (
        result["评论内容_原始"].str.len() - result["评论内容"].str.len()
    ).clip(lower=0)
    result["文本是否有效"] = result["评论内容"].map(
        lambda value: len(value) > 2
        and any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in value)
    )

    invalid_count = int((~result["文本是否有效"]).sum())
    result = result.loc[result["文本是否有效"]].reset_index(drop=True)
    return result, invalid_count


def _parse_base_fields(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    _ensure_column(result, "车型年份", np.nan)
    _ensure_column(result, "购买版本", "")
    _ensure_column(result, "购车价格", np.nan)
    _ensure_column(result, "购车地区", "未知")
    _ensure_column(result, "购车时间", pd.NaT)
    _ensure_column(result, "评论时间", pd.NaT)
    _ensure_column(result, "点赞", np.nan)
    _ensure_column(result, "评论数", np.nan)
    _ensure_column(result, "用户", np.nan)

    result["用户是否缺失"] = result["用户"].isna() | result["用户"].astype("string").str.strip().eq(
        ""
    )
    result["车系ID"] = pd.to_numeric(result["车系ID"], errors="coerce").astype("Int64")
    result["车系ID是否缺失"] = result["车系ID"].isna()
    result["车型年份"] = pd.to_numeric(result["车型年份"], errors="coerce")
    result["车型年份是否缺失"] = result["车型年份"].isna()
    result["点赞"] = pd.to_numeric(result["点赞"], errors="coerce")
    result["评论数"] = pd.to_numeric(result["评论数"], errors="coerce")
    result["点赞是否缺失"] = result["点赞"].isna()
    result["评论数是否缺失"] = result["评论数"].isna()
    # 预清洗数据里该列已改名为「车型版本」。若硬读「购买版本」会拿到
    # _ensure_column 补出来的空列，导致车型版本_规范 全为空、
    # 参数匹配全部退化成车系中位数（精确匹配率 0%）。
    version_source = _pick_version_series(result)
    result["车型版本_规范"] = version_source.map(normalize_version)
    return result


def _pick_version_series(data: pd.DataFrame) -> pd.Series:
    """在两个数据入口下都能取到真实的购买版本。

    不能用 notna() 判断：_ensure_column 补出来的列可能是空字符串，
    notna() 对 "" 返回 True，会误选中一个全空的「购买版本」，
    把真正有值的「车型版本」挡在后面。
    """
    for column in ("购买版本", "车型版本"):
        if column not in data.columns:
            continue
        series = data[column]
        as_text = series.fillna("").astype(str).str.strip()
        if (as_text.ne("") & as_text.str.lower().ne("nan")).any():
            return as_text
    return pd.Series("", index=data.index, dtype=object)


def _clean_scores(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    zero_count = pd.Series(0, index=result.index, dtype="int64")
    out_of_range_count = pd.Series(0, index=result.index, dtype="int64")

    for column in SCORE_COLUMNS:
        _ensure_column(result, column, np.nan)
        numeric = pd.to_numeric(result[column], errors="coerce")
        zero_mask = numeric.eq(0)
        out_of_range_mask = numeric.lt(0) | numeric.gt(5)
        zero_count = zero_count.add(zero_mask.astype("int64"))
        out_of_range_count = out_of_range_count.add(out_of_range_mask.astype("int64"))
        result[column] = numeric.mask(zero_mask | out_of_range_mask)

    result["评分为零转缺失数"] = zero_count
    result["评分越界转缺失数"] = out_of_range_count
    result["评分有效维度数"] = result[SCORE_COLUMNS].notna().sum(axis=1).astype("int64")
    result["评分缺失维度数"] = result[SCORE_COLUMNS].isna().sum(axis=1).astype("int64")
    result["评分全部缺失"] = result["评分有效维度数"].eq(0)
    result["核心评分均值"] = result[CORE_SCORE_COLUMNS].mean(axis=1, skipna=True)
    return result


def _parse_price(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["购车价格_原始值"] = result["购车价格"].map(parse_price)
    result["购车价格_有效值"] = result["购车价格_原始值"].where(
        result["购车价格_原始值"].between(10000, 1_000_000)
    )

    version_median = pd.Series(np.nan, index=result.index, dtype="float64")
    version_group_mask = (
        result["车系ID"].notna() & result["车型版本_规范"].fillna("").ne("")
    )
    if version_group_mask.any():
        version_median.loc[version_group_mask] = result.loc[version_group_mask].groupby(
            ["车系ID", "车型版本_规范"],
            dropna=True,
        )["购车价格_有效值"].transform("median")

    series_median = pd.Series(np.nan, index=result.index, dtype="float64")
    series_group_mask = result["车系ID"].notna()
    if series_group_mask.any():
        series_median.loc[series_group_mask] = result.loc[series_group_mask].groupby(
            "车系ID",
            dropna=True,
        )["购车价格_有效值"].transform("median")
    global_median = result["购车价格_有效值"].median()

    analyzed = result["购车价格_有效值"].copy()
    source = pd.Series("原始有效", index=result.index, dtype="string")

    version_fill = analyzed.isna() & version_median.notna()
    analyzed = analyzed.fillna(version_median)
    source = source.mask(version_fill, "购买版本中位数")

    series_fill = analyzed.isna() & series_median.notna()
    analyzed = analyzed.fillna(series_median)
    source = source.mask(series_fill, "车系中位数")

    global_fill = analyzed.isna() & pd.notna(global_median)
    analyzed = analyzed.fillna(global_median)
    source = source.mask(global_fill, "总体中位数")

    source = source.mask(analyzed.isna(), "无法补充")
    result["购车价格_分析值"] = analyzed
    result["购车价格是否估算"] = (
        result["购车价格_有效值"].isna() & result["购车价格_分析值"].notna()
    )
    result["购车价格_补充来源"] = source
    result["购车价格是否缺失"] = result["购车价格_分析值"].isna()
    result["价格是否可用于分析"] = result["购车价格_分析值"].notna()

    if pd.isna(global_median):
        LOGGER.warning("没有任何有效购车价格，价格分析值保持缺失，未填 0")
    return result


def _parse_region_and_time(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["购车地区_原始"] = result["购车地区"]
    result["购车地区"] = result["购车地区"].map(normalize_region)
    result["地区是否未知"] = result["购车地区"].eq("未知")
    result["地区识别状态"] = np.where(result["地区是否未知"], "未知", "已识别")

    result["购车时间_原始"] = result["购车时间"]
    result["评论时间_原始"] = result["评论时间"]
    result["购车时间"] = pd.to_datetime(result["购车时间"], errors="coerce")
    result["评论时间"] = pd.to_datetime(result["评论时间"], errors="coerce")
    result["购车时间是否缺失"] = result["购车时间"].isna()
    result["评论时间是否缺失"] = result["评论时间"].isna()
    result["购车年月"] = (
        result["购车时间"].dt.to_period("M").astype("string")
    )
    result["评论年月"] = (
        result["评论时间"].dt.to_period("M").astype("string")
    )
    return result


def _prepare_series(series: pd.DataFrame) -> pd.DataFrame:
    _require_columns(series, REQUIRED_SERIES_COLUMNS, "车系表")
    result = series.copy()
    result["车系ID"] = pd.to_numeric(result["车系ID"], errors="coerce").astype("Int64")
    result = result.rename(
        columns={
            "品牌": "品牌_车系表",
            "车系名": "车型_车系表",
            "级别": "车辆级别",
            "能源类型": "能源类型",
        }
    )
    return result.drop_duplicates(subset=["车系ID"], keep="first")


def _prepare_models(models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = models.copy()
    result["车系ID"] = pd.to_numeric(result["所属车系ID"], errors="coerce").astype("Int64")
    result["车型版本_规范"] = result["车型版本"].map(normalize_version)

    for source, target in MODEL_NUMERIC_COLUMNS.items():
        parser = parse_price if source in MODEL_PRICE_COLUMNS else parse_first_number
        result[target] = result[source].map(parser)

    result = result.rename(
        columns={
            "车型版本": "参数车型版本",
            "电池种类": "电池种类_参数",
            "驱动方式": "驱动方式_参数",
            "前悬架形式": "前悬架形式_参数",
            "后悬架形式": "后悬架形式_参数",
            "电机类型": "电机类型_参数",
        }
    )

    selected = [
        "车系ID",
        "车型版本_规范",
        "参数车型版本",
        *PARAMETER_COLUMNS,
    ]
    exact = result[selected].drop_duplicates(
        subset=["车系ID", "车型版本_规范"],
        keep="first",
    )

    series_median = (
        result.groupby("车系ID", dropna=False)[NUMERIC_PARAMETER_COLUMNS]
        .median()
        .reset_index()
        .rename(
            columns={
                column: f"{column}_车系中位数"
                for column in NUMERIC_PARAMETER_COLUMNS
            }
        )
    )
    return exact, series_median


def _merge_parameter_data(
    reviews: pd.DataFrame,
    exact: pd.DataFrame,
    series_median: pd.DataFrame,
) -> pd.DataFrame:
    result = reviews.merge(
        exact,
        on=["车系ID", "车型版本_规范"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        series_median,
        on="车系ID",
        how="left",
        validate="many_to_one",
    )

    version_match = result["参数车型版本"].notna()
    series_median_columns = [f"{column}_车系中位数" for column in NUMERIC_PARAMETER_COLUMNS]
    series_median_available = result[series_median_columns].notna().any(axis=1)
    result["参数匹配方式"] = np.select(
        [version_match, series_median_available],
        ["版本匹配", "车系中位数"],
        default="无法匹配",
    )

    original_missing = result[PARAMETER_COLUMNS].isna()
    estimated_flags: dict[str, pd.Series] = {}

    for column in NUMERIC_PARAMETER_COLUMNS:
        median_column = f"{column}_车系中位数"
        missing = result[column].isna()
        filled = result[column].fillna(result[median_column])
        estimated_flags[column] = missing & filled.notna()
        result[column] = filled
        result.drop(columns=[median_column], inplace=True)

    for column in CATEGORICAL_PARAMETER_COLUMNS:
        result[column] = result[column].fillna("未知")

    estimated_matrix = pd.DataFrame(estimated_flags, index=result.index)
    result["参数原始缺失数"] = original_missing.sum(axis=1).astype("int64")
    result["参数估算字段数"] = estimated_matrix.sum(axis=1).astype("int64")
    result["参数是否估算"] = result["参数估算字段数"].gt(0)
    result["参数估算字段"] = estimated_matrix.apply(
        lambda row: "、".join(row.index[row].tolist()),
        axis=1,
    )

    remaining_missing_matrix = pd.DataFrame(index=result.index)
    for column in NUMERIC_PARAMETER_COLUMNS:
        remaining_missing_matrix[column] = result[column].isna()
    for column in CATEGORICAL_PARAMETER_COLUMNS:
        remaining_missing_matrix[column] = result[column].eq("未知")

    result["参数仍缺失字段数"] = remaining_missing_matrix.sum(axis=1).astype("int64")
    result["参数仍缺失字段"] = remaining_missing_matrix.apply(
        lambda row: "、".join(row.index[row].tolist()),
        axis=1,
    )
    result["参数是否完整"] = result["参数仍缺失字段数"].eq(0)
    result["参数是否可用于分析"] = (
        result["参数匹配方式"].ne("无法匹配")
        & result["参数仍缺失字段数"].lt(len(PARAMETER_COLUMNS))
    )
    return result


def _attach_series_identity(
    reviews: pd.DataFrame,
    series: pd.DataFrame,
) -> pd.DataFrame:
    result = reviews.copy()
    _ensure_column(result, "品牌", np.nan)
    _ensure_column(result, "车型", np.nan)
    result["品牌_原始"] = result["品牌"]
    result["车型_原始"] = result["车型"]

    series_data = _prepare_series(series)
    result = result.merge(
        series_data,
        on="车系ID",
        how="left",
        validate="many_to_one",
    )
    result["车系信息是否匹配"] = result["车型_车系表"].notna()
    result["品牌"] = result["品牌"].fillna(result["品牌_车系表"]).fillna("未知")
    result["车型"] = result["车型"].fillna(result["车型_车系表"]).fillna("未知")
    result["车辆级别"] = result["车辆级别"].fillna("未知")
    result["能源类型"] = result["能源类型"].fillna("未知")
    result["车系信息是否缺失"] = (
        result["品牌"].eq("未知")
        | result["车型"].eq("未知")
        | result["车辆级别"].eq("未知")
        | result["能源类型"].eq("未知")
    )
    result.drop(columns=["品牌_车系表", "车型_车系表"], inplace=True)
    return result


def _build_quality_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    conditions = {
        "价格估算": result["购车价格是否估算"],
        "价格不可分析": result["购车价格_分析值"].isna(),
        "地区未知": result["地区是否未知"],
        "参数估算": result["参数是否估算"],
        "参数仍缺失": result["参数仍缺失字段数"].gt(0),
        "参数无法匹配": result["参数匹配方式"].eq("无法匹配"),
        "购车时间缺失": result["购车时间是否缺失"],
        "评论时间缺失": result["评论时间是否缺失"],
        "评分全部缺失": result["评分全部缺失"],
        "用户标识缺失": result["用户是否缺失"],
        "车系信息缺失": result["车系信息是否缺失"],
        "文本清洗发生变化": result["文本清洗变化"],
    }

    condition_frame = pd.DataFrame(conditions, index=result.index).fillna(False)
    tag_frame = condition_frame.apply(
        lambda column: np.where(column, column.name + "；", ""),
    )
    result["质量标记"] = tag_frame.sum(axis=1).str.rstrip("；")
    result["质量标记数量"] = condition_frame.sum(axis=1).astype("int64")
    result["质量标记"] = result["质量标记"].mask(
        result["质量标记数量"].eq(0),
        "无明显质量问题",
    )

    low_quality = (
        ~result["文本是否有效"]
        | result["购车价格_分析值"].isna()
        | result["参数匹配方式"].eq("无法匹配")
        | result["评分全部缺失"]
    )
    medium_quality = result["质量标记数量"].gt(0) & ~low_quality
    result["数据质量等级"] = np.select(
        [low_quality, medium_quality],
        ["低", "中"],
        default="高",
    )

    result["可用于文本分析"] = result["文本是否有效"]
    result["可用于价格分析"] = result["购车价格_分析值"].notna()
    result["可用于参数分析"] = (
        ~result["参数匹配方式"].eq("无法匹配")
        & result["参数仍缺失字段数"].lt(len(PARAMETER_COLUMNS))
    )
    result["可用于主分析"] = (
        result["可用于文本分析"]
        & result["可用于价格分析"]
        & result["可用于参数分析"]
    )
    return result


def _value_counts(data: pd.Series) -> dict[str, int]:
    counts = data.fillna("缺失").astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _build_quality_report(
    reviews: pd.DataFrame,
    input_files: dict[str, Path | None],
    counts: dict[str, int],
) -> dict[str, Any]:
    score_valid_rates = {
        column: round(float(reviews[column].notna().mean()), 6)
        for column in SCORE_COLUMNS
    }
    missing_rates = {
        column: round(float(reviews[column].isna().mean()), 6)
        for column in [
            "购车价格_有效值",
            "购车价格_分析值",
            "购车时间",
            "评论时间",
            "品牌",
            "车型",
            "能源类型",
        ]
    }
    return {
        **counts,
        "final_rows": int(len(reviews)),
        "input_files": {
            role: (
                {"path": str(path), "sha256": _file_hash(path)}
                if path is not None
                else {"path": "内存输入", "sha256": "内存输入"}
            )
            for role, path in input_files.items()
        },
        "price_source_counts": _value_counts(reviews["购车价格_补充来源"]),
        "parameter_match_counts": _value_counts(reviews["参数匹配方式"]),
        "quality_level_counts": _value_counts(reviews["数据质量等级"]),
        "score_valid_rates": score_valid_rates,
        "missing_rates": missing_rates,
        "analysis_availability": {
            "text": int(reviews["可用于文本分析"].sum()),
            "price": int(reviews["可用于价格分析"].sum()),
            "parameter": int(reviews["可用于参数分析"].sum()),
            "main": int(reviews["可用于主分析"].sum()),
        },
    }


def preprocess_data(
    settings: dict[str, Any] | None = None,
    reviews: pd.DataFrame | None = None,
    series: pd.DataFrame | None = None,
    models: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clean reviews and preserve every missing, estimated and unmatched field."""

    settings = settings or load_settings()
    paths = get_paths(settings)

    # 优先使用 caiji 目录已清洗数据：跳过原始清洗，但仍复用下面的
    # 评分/价格/地区/车系/参数关联逻辑，保证输出 schema 完全一致
    if reviews is None and series is None and models is None:
        try:
            from src.precleaned_adapter import has_precleaned, load_precleaned_frames

            if has_precleaned(settings):
                LOGGER.info("检测到预清洗数据，使用 caiji 数据集")
                pre_reviews, pre_series, pre_models = load_precleaned_frames(settings)
                if pre_reviews is not None:
                    reviews, series, models = pre_reviews, pre_series, pre_models
        except Exception as error:  # noqa: BLE001 - 适配失败应回退到原始模式
            LOGGER.warning("预清洗数据加载失败，回退到原始数据: %s", error)

    if reviews is None:
        reviews_path = _resolve_input_path(paths["raw_reviews"], "reviews")
        reviews = _read_csv(reviews_path, "评论数据")
    else:
        reviews_path = None

    if series is None:
        series_path = _resolve_input_path(paths["series"], "series")
        series = _read_csv(series_path, "车系表")
    else:
        series_path = None

    if models is None:
        models_path = _resolve_input_path(paths["models"], "models")
        models = _read_csv(models_path, "车型表")
    else:
        models_path = None

    _require_columns(reviews, REQUIRED_REVIEW_COLUMNS, "评论数据")
    _require_columns(series, REQUIRED_SERIES_COLUMNS, "车系表")
    _require_columns(models, REQUIRED_MODEL_COLUMNS, "车型表")

    original_count = int(len(reviews))
    result = reviews.copy()
    result["评论内容_原始"] = result["评论内容"].fillna("").astype(str)
    result, duplicate_counts = _deduplicate_reviews(result)
    result, dropped_invalid_text = _clean_text(result)
    result, clean_duplicate_count = _deduplicate_clean_reviews(result)
    duplicate_counts["business_duplicate_rows"] = clean_duplicate_count

    if result.empty:
        raise DataInputError(
            "评论记录经去重和文本清洗后为空。\n\n" + DATA_SELF_CHECK_MESSAGE
        )

    result = _parse_base_fields(result)
    result = _clean_scores(result)
    result = _parse_price(result)
    result = _parse_region_and_time(result)
    result = _attach_series_identity(result, series)
    exact_params, series_params = _prepare_models(models)
    result = _merge_parameter_data(result, exact_params, series_params)
    result = _build_quality_columns(result)

    result["源文件"] = str(reviews_path) if reviews_path is not None else "内存输入"
    result["源文件哈希"] = (
        _file_hash(reviews_path) if reviews_path is not None else "内存输入"
    )
    result = result.sort_values(
        ["评论时间", "原始行号"],
        na_position="last",
    ).reset_index(drop=True)
    result["doc_id"] = np.arange(1, len(result) + 1, dtype=int)

    counts = {
        "original_rows": original_count,
        **duplicate_counts,
        "dropped_invalid_text_rows": dropped_invalid_text,
    }
    report = _build_quality_report(
        result,
        {
            "reviews": reviews_path,
            "series": series_path,
            "models": models_path,
        },
        counts,
    )

    paths["processed_reviews"].parent.mkdir(parents=True, exist_ok=True)
    report_path = paths["processed_reviews"].with_name("preprocessing_quality_report.json")
    result.to_csv(paths["processed_reviews"], index=False, encoding="utf-8-sig")
    write_json(report_path, report)

    estimated_prices = int(result["购车价格是否估算"].sum())
    unmatched_parameters = int(result["参数匹配方式"].eq("无法匹配").sum())
    LOGGER.info(
        "评论清洗完成: 原始%d -> 去重后%d -> 有效%d",
        original_count,
        original_count - duplicate_counts["exact_duplicate_rows"] - duplicate_counts[
            "business_duplicate_rows"
        ],
        len(result),
    )
    LOGGER.info("价格估算记录: %d；参数无法匹配记录: %d", estimated_prices, unmatched_parameters)
    if estimated_prices:
        LOGGER.warning("存在价格估算，使用均值前必须按购车价格是否估算分组核对")
    if unmatched_parameters:
        LOGGER.warning("存在无法匹配车型参数的记录，相关参数保持缺失或未知")
    LOGGER.info("预处理质量报告: %s", report_path)
    return result
