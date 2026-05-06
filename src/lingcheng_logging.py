"""
文件名: lingcheng_logging.py
用途: 配置 Lingcheng 应用日志：INFO、带时间戳；默认同时写入项目根目录 lingcheng.log（UTF-8）
      与标准错误流（终端可见）。
对外暴露:
  - setup_lingcheng_logging: 初始化日志（幂等）
  - get_logger: 获取 lingcheng.<子模块名> 子 logger
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_LOG_NAME = "lingcheng.log"

_CONFIGURED = False


def setup_lingcheng_logging(
    project_root: Optional[Path] = None,
    log_filename: str = _DEFAULT_LOG_NAME,
    *,
    console: bool = True,
) -> logging.Logger:
    """配置 lingcheng 命名空间日志：文件 + 可选控制台；不记录 API Key 等敏感字段由调用方保证。"""
    global _CONFIGURED
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    log_path = project_root.resolve() / log_filename

    logg = logging.getLogger("lingcheng")
    logg.setLevel(logging.INFO)

    if _CONFIGURED:
        return logg

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logg.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logg.addHandler(sh)

    logg.propagate = False
    _CONFIGURED = True

    logg.info("lingcheng logging initialized path=%s console=%s", log_path, console)
    for h in logg.handlers:
        try:
            h.flush()
        except OSError:
            pass
    return logg


def get_logger(name: str) -> logging.Logger:
    """返回子 logger，name 例如 'agent.router'、'ui.gradio'、'tool.12306'。"""
    return logging.getLogger(f"lingcheng.{name}")
