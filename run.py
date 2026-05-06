"""
文件名: run.py
用途: 项目统一启动脚本，加载 .env 环境变量后启动 Gradio 聊天界面。
对外暴露: main (启动入口函数)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _bootstrap() -> None:
    """加载 .env 并把项目根目录加入 sys.path，方便 `python run.py` 直接运行。"""
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    load_dotenv(project_root / ".env", override=False)


def main() -> None:
    """灵程 Agent 启动入口：检查环境变量并拉起 Gradio 应用。"""
    _bootstrap()

    from src.lingcheng_logging import setup_lingcheng_logging

    _console = os.getenv("LINGCHENG_LOG_CONSOLE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    setup_lingcheng_logging(
        Path(__file__).resolve().parent,
        console=_console,
    )

    if not os.getenv("DASHSCOPE_API_KEY"):
        print(
            "[警告] 未检测到 DASHSCOPE_API_KEY 环境变量。\n"
            "请将 .env.example 复制为 .env 并填写阿里百炼的 API Key 后再运行。\n"
            "继续启动可能会在调用 LLM 时报错。"
        )

    from src.ui.gradio_app import launch

    launch()


if __name__ == "__main__":
    main()
