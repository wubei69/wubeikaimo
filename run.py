from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
APP = ROOT / "app" / "streamlit_app.py"

# 流水线产出的关键文件；缺失则说明还没跑过分析
ARTIFACTS = [
    "data/processed/reviews_processed.csv",
    "data/processed/topic_probabilities.csv",
    "data/processed/sentiment_results.csv",
    "data/processed/consumer_clusters.csv",
    "data/processed/association_rules.csv",
    "outputs/tables/cluster_metrics.json",
    "outputs/reports/analysis_report.md",
]


def resolve_python() -> str:
    """优先用项目自带的 .venv，避免依赖系统 python 的第三方包。"""
    if VENV_PY.exists():
        return str(VENV_PY)
    return sys.executable


def check_artifacts() -> tuple[int, int]:
    ok, missing = 0, []
    for rel in ARTIFACTS:
        path = ROOT / rel
        if path.exists() and path.stat().st_size > 0:
            ok += 1
        else:
            missing.append(rel)
    if missing:
        print("  [缺失] 以下产物还没有，需要先跑流水线：")
        for rel in missing:
            print(f"         - {rel}")
    return ok, len(missing)


def check_environment(python: str) -> bool:
    """确认看板依赖的包都在，否则 streamlit 会在启动时才报 ModuleNotFound。"""
    script = (
        "import importlib.util;"
        "mods=['streamlit','pandas','plotly','numpy','sklearn'];"
        "miss=[m for m in mods if importlib.util.find_spec(m) is None];"
        "print('MISSING:'+','.join(miss)) if miss else print('OK')"
    )
    result = subprocess.run([python, "-c", script], capture_output=True, text=True)
    out = (result.stdout or "").strip()
    if out.startswith("MISSING:"):
        print(f"  [缺失] 依赖包未安装: {out.split(':', 1)[1]}")
        return False
    if out != "OK":
        print(f"  [警告] 环境检查未通过: {out or result.stderr.strip()[:200]}")
        return False
    return True


def run_pipeline(python: str, extra_args: list[str]) -> bool:
    cmd = [python, "-m", "src.pipeline", *extra_args]
    print(f"\n>>> 运行流水线: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[失败] 流水线退出码 {result.returncode}，已中止，不启动看板。")
        return False
    print("\n[完成] 流水线执行成功。")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="新能源汽车消费特征挖掘系统 运行入口")
    parser.add_argument("--pipeline", action="store_true", help="先重跑分析流水线")
    parser.add_argument("--skip-topics", action="store_true", help="跳过 LDA 主题挖掘")
    parser.add_argument("--skip-clustering", action="store_true", help="跳过消费者聚类")
    parser.add_argument("--skip-association", action="store_true", help="跳过关联规则")
    parser.add_argument("--check", action="store_true", help="只体检不启动")
    parser.add_argument("--port", type=int, default=8501, help="看板端口，默认 8501")
    args, _unknown = parser.parse_known_args()

    print("=" * 56)
    print("新能源汽车消费特征挖掘系统")
    print("=" * 56)

    python = resolve_python()
    print(f"\n[1/3] Python 解释器\n      {python}")
    if python == str(VENV_PY):
        print("      使用项目自带 .venv")
    else:
        print("      [提示] 未找到 .venv，改用当前 python；")
        print("             若报缺包，请先执行：python -m venv .venv")

    print("\n[2/3] 依赖检查")
    env_ok = check_environment(python)

    print("\n[3/3] 分析结果检查")
    ok_count, miss_count = check_artifacts()
    print(f"      已有 {ok_count}/{len(ARTIFACTS)} 项产物")

    if args.check:
        print("\n体检结束（未启动看板）。")
        return 0 if (env_ok and miss_count == 0) else 1

    if not env_ok:
        print("\n[中止] 依赖缺失，请先安装：")
        print(f"       {python} -m pip install -r requirements.txt")
        return 1

    if args.pipeline:
        extra = []
        if args.skip_topics:
            extra.append("--skip-topics")
        if args.skip_clustering:
            extra.append("--skip-clustering")
        if args.skip_association:
            extra.append("--skip-association")
        if not run_pipeline(python, extra):
            return 1
    elif miss_count:
        print("\n[提示] 产物不完整，看板部分页面会显示「尚未生成」。")
        print("       如需完整结果，加 --pipeline 重跑：python run.py --pipeline")

    print(f"\n>>> 启动看板 http://localhost:{args.port}")
    print("    关闭方法：在此窗口按 Ctrl+C\n")
    cmd = [
        python, "-m", "streamlit", "run", str(APP),
        "--server.port", str(args.port),
        "--server.headless", "true",
    ]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
