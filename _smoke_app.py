"""看板冒烟测试：用假的 streamlit 逐页执行 app/streamlit_app.py。

目的：Plotly 的枚举参数错误（如 barmode="fill"）只在页面真正渲染时才抛异常，
肉眼看代码很难发现。这里把 streamlit 打桩后把 7 个页面全部跑一遍，
任何一页的图表参数不合法都会当场暴露。
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app" / "streamlit_app.py"

PAGES = [
    "数据总览",
    "消费画像",
    "需求主题",
    "满意度分析",
    "消费者聚类",
    "竞品对比",
    "数据质量",
]


class StopExecution(Exception):
    """对应 st.stop()。"""


def build_fake_streamlit(page: str) -> types.ModuleType:
    st = types.ModuleType("streamlit")

    figures: list = []

    def cache_data(func=None, **_kwargs):
        # 用作装饰器：@st.cache_data 或 @st.cache_data(...)
        if func is None:
            return lambda f: f
        return func

    def columns(spec, **_kwargs):
        # 既支持 st.columns(2)，也支持 st.columns([2, 1]) / st.columns((1, 3))
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [MagicMock() for _ in range(n)]

    def multiselect(_label, options, default=None, **_kwargs):
        # 返回默认值，保证下游 .isin(...) 拿到的是真实取值
        return default if default is not None else list(options)

    def selectbox(_label, options, index=0, **_kwargs):
        return list(options)[index or 0]

    def radio(_label, options, index=0, **_kwargs):
        return page  # 固定为待测页面

    def slider(_label, min_value=0, max_value=1, value=None, **_kwargs):
        return value if value is not None else min_value

    def plotly_chart(fig, **_kwargs):
        # 强制序列化，触发 Plotly 的全部属性校验
        fig.to_json()
        figures.append(fig)

    def stop():
        raise StopExecution()

    st.cache_data = cache_data
    st.columns = columns
    st.multiselect = multiselect
    st.selectbox = selectbox
    st.slider = slider
    st.plotly_chart = plotly_chart
    st.stop = stop
    st.set_page_config = lambda **_k: None
    st.title = st.subheader = st.markdown = st.caption = st.write = lambda *a, **k: None
    st.info = st.warning = st.error = st.success = lambda *a, **k: None
    st.code = lambda *a, **k: None
    st.dataframe = lambda *a, **k: None
    st.metric = lambda *a, **k: None
    st.divider = lambda *a, **k: None
    st.download_button = lambda *a, **k: None
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: False
    st.expander = lambda *a, **k: MagicMock()
    st.spinner = lambda *a, **k: MagicMock()
    st.tabs = lambda labels, **_k: [MagicMock() for _ in labels]

    sidebar = types.SimpleNamespace(
        title=lambda *a, **k: None,
        radio=radio,
        selectbox=selectbox,
        multiselect=multiselect,
        slider=slider,
        metric=lambda *a, **k: None,
        divider=lambda *a, **k: None,
        markdown=lambda *a, **k: None,
        button=lambda *a, **k: False,
        checkbox=lambda *a, **k: False,
    )
    st.sidebar = sidebar
    return st, figures


def main() -> int:
    failed = 0
    for page in PAGES:
        fake_st, figures = build_fake_streamlit(page)
        sys.modules["streamlit"] = fake_st
        try:
            runpy.run_path(str(APP), run_name="__smoke__")
        except StopExecution:
            print(f"  [跳过] {page}  (命中 st.stop，通常是数据未就绪分支)")
            continue
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [失败] {page}: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        print(f"  [通过] {page}  ({len(figures)} 张图表)")
    print()
    print("全部页面通过" if not failed else f"{failed} 个页面存在错误")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
