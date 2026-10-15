from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import load_settings
from src.data_preprocessing import DataInputError, preprocess_data


def _test_settings(tmp_path: Path) -> dict:
    settings = load_settings()
    settings.update(
        {
            "raw_reviews": tmp_path / "raw" / "reviews.csv",
            "series": tmp_path / "raw" / "series.csv",
            "models": tmp_path / "raw" / "models.csv",
            "processed_reviews": tmp_path / "processed" / "reviews_processed.csv",
            "feature_data": tmp_path / "processed" / "features.csv",
            "database": tmp_path / "processed" / "analysis.db",
            "topic_probabilities": tmp_path / "processed" / "topics.csv",
            "sentiment_results": tmp_path / "processed" / "sentiment.csv",
            "cluster_results": tmp_path / "processed" / "clusters.csv",
            "association_rules": tmp_path / "processed" / "rules.csv",
            "artifacts_dir": tmp_path / "models",
            "tables_dir": tmp_path / "outputs" / "tables",
            "figures_dir": tmp_path / "outputs" / "figures",
            "reports_dir": tmp_path / "outputs" / "reports",
        }
    )
    return settings


def _write_source_files(tmp_path: Path) -> dict:
    settings = _test_settings(tmp_path)
    settings["raw_reviews"].parent.mkdir(parents=True, exist_ok=True)

    reviews = pd.DataFrame(
        [
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "500km 标准版",
                "购车价格": "10万",
                "购车地区": "盐城市",
                "购车时间": "2024-01",
                "用户": "u1",
                "评论内容": "<p>空间不错，续航一般。</p> https://example.com",
                "综合评分": 0,
                "空间评分": 4.5,
                "评论时间": "2024-02-01",
                "是否有图": "否",
            },
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "500km 标准版",
                "购车价格": "10万",
                "购车地区": "盐城市",
                "购车时间": "2024-01",
                "用户": "u1",
                "评论内容": "<p>空间不错，续航一般。</p> https://example.com",
                "综合评分": 0,
                "空间评分": 4.5,
                "评论时间": "2024-02-01",
                "是否有图": "否",
            },
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "500km 标准版",
                "购车价格": "10万",
                "购车地区": "盐城",
                "购车时间": "2024-01",
                "用户": "u1",
                "评论内容": "空间不错，续航一般。",
                "综合评分": 3.5,
                "空间评分": 4.5,
                "评论时间": "2024-02-02",
                "是否有图": "是",
            },
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "500km 标准版",
                "购车价格": "11万",
                "购车地区": "郑州",
                "购车时间": "2024-02",
                "用户": "u2",
                "评论内容": "配置丰富，动力很好。",
                "综合评分": 4.5,
                "空间评分": 4.0,
                "评论时间": "2024-03-01",
                "是否有图": "否",
            },
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "500km 标准版",
                "购车价格": None,
                "购车地区": "郑州",
                "购车时间": "2024-02",
                "用户": "u3",
                "评论内容": "续航表现稳定。",
                "综合评分": 4.0,
                "空间评分": 4.5,
                "评论时间": "2024-03-02",
                "是否有图": "否",
            },
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "600km 长续航版",
                "购车价格": "20万",
                "购车地区": "北京",
                "购车时间": "2024-03",
                "用户": "u4",
                "评论内容": "长续航版本更合适。",
                "综合评分": 5.0,
                "空间评分": 4.5,
                "评论时间": "2024-04-01",
                "是否有图": "否",
            },
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "700km 未上市版",
                "购车价格": None,
                "购车地区": "上海",
                "购车时间": "2024-03",
                "用户": "u5",
                "评论内容": "期待后续版本。",
                "综合评分": 4.0,
                "空间评分": 4.0,
                "评论时间": "2024-04-02",
                "是否有图": "否",
            },
            {
                "车系ID": 103,
                "品牌": "丙品牌",
                "车型": "丙车型",
                "车型年份": 2024,
                "购买版本": "未知版本",
                "购车价格": None,
                "购车地区": "未知",
                "购车时间": "2024-04",
                "用户": "u6",
                "评论内容": "内饰设计有特点。",
                "综合评分": 3.0,
                "空间评分": 3.5,
                "评论时间": "2024-05-01",
                "是否有图": "否",
            },
            {
                "车系ID": 101,
                "品牌": "甲品牌",
                "车型": "甲车型",
                "车型年份": 2024,
                "购买版本": "500km 标准版",
                "购车价格": "12万",
                "购车地区": "杭州",
                "购车时间": "2024-04",
                "用户": "u7",
                "评论内容": "",
                "综合评分": 4.0,
                "空间评分": 4.0,
                "评论时间": "2024-05-02",
                "是否有图": "否",
            },
        ]
    )
    series = pd.DataFrame(
        [
            {"车系ID": 101, "品牌": "甲品牌", "车系名": "甲车型", "级别": "中型车", "能源类型": "纯电动"},
            {"车系ID": 103, "品牌": "丙品牌", "车系名": "丙车型", "级别": "小型车", "能源类型": "纯电动"},
        ]
    )
    models = pd.DataFrame(
        [
            {
                "所属车系ID": 101,
                "车型版本": "500km 标准版",
                "官方指导价": "10.5万",
                "百公里加速": "7.5秒",
                "续航": "500km",
                "电池电量": "60kWh",
                "电机功率": "150kW",
                "电机扭矩": "310N·m",
                "最高时速": "180km/h",
                "电池种类": "磷酸铁锂",
                "驱动方式": "后置后驱",
                "前悬架形式": "麦弗逊",
                "后悬架形式": "多连杆",
                "电机类型": "永磁同步",
            },
            {
                "所属车系ID": 101,
                "车型版本": "600km 长续航版",
                "官方指导价": "20万",
                "百公里加速": "6.2秒",
                "续航": "600km",
                "电池电量": "80kWh",
                "电机功率": "200kW",
                "电机扭矩": "400N·m",
                "最高时速": "200km/h",
                "电池种类": "三元锂",
                "驱动方式": "双电机四驱",
                "前悬架形式": "双叉臂",
                "后悬架形式": "多连杆",
                "电机类型": "永磁同步",
            },
        ]
    )

    reviews.to_csv(settings["raw_reviews"], index=False, encoding="utf-8-sig")
    series.to_csv(settings["series"], index=False, encoding="utf-8-sig")
    models.to_csv(settings["models"], index=False, encoding="utf-8-sig")
    return settings


def test_preprocess_tracks_duplicates_missing_and_estimates(tmp_path: Path) -> None:
    settings = _write_source_files(tmp_path)
    result = preprocess_data(settings)

    assert len(result) == 6
    assert result["doc_id"].is_unique
    assert "空间不错，续航一般。" in result.loc[0, "评论内容"]
    assert "http" not in result.loc[0, "评论内容"]

    zero_score_row = result.loc[result["doc_id"].eq(1)].iloc[0]
    assert pd.isna(zero_score_row["综合评分"])
    assert zero_score_row["评分为零转缺失数"] >= 1

    source_counts = result["购车价格_补充来源"].value_counts().to_dict()
    assert source_counts["原始有效"] == 3
    assert source_counts["购买版本中位数"] == 1
    assert source_counts["车系中位数"] == 1
    assert source_counts["总体中位数"] == 1
    assert int(result["购车价格是否估算"].sum()) == 3
    assert result["购车价格_分析值"].notna().all()

    estimated_version_price = result.loc[
        result["购车价格_补充来源"].eq("购买版本中位数"),
        "购车价格_分析值",
    ].iloc[0]
    assert estimated_version_price == pytest.approx(105000)

    parameter_methods = set(result["参数匹配方式"])
    assert {"版本匹配", "车系中位数", "无法匹配"}.issubset(parameter_methods)
    assert result["参数估算字段"].str.len().gt(0).any()
    assert result["质量标记"].str.len().gt(0).all()
    assert set(result["数据质量等级"]).issubset({"高", "中", "低"})

    report_path = settings["processed_reviews"].with_name("preprocessing_quality_report.json")
    assert report_path.exists()


def test_preprocess_rejects_header_only_input(tmp_path: Path) -> None:
    settings = _write_source_files(tmp_path)
    pd.DataFrame(columns=["车系ID", "评论内容"]).to_csv(
        settings["raw_reviews"],
        index=False,
        encoding="utf-8-sig",
    )

    with pytest.raises(DataInputError, match="程序已停止"):
        preprocess_data(settings)
