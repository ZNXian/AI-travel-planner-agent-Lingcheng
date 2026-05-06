"""
文件名: lingcheng_logging.py
用途: 配置 Lingcheng 应用日志：级别由 LINGCHENG_LOG_LEVEL 控制（默认 INFO），带时间戳；
      默认同时写入项目根目录 lingcheng.log（UTF-8）与标准错误流（终端可见）。
对外暴露:
  - setup_lingcheng_logging: 初始化日志（幂等）
  - get_logger: 获取 lingcheng.<子模块名> 子 logger
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_DEFAULT_LOG_NAME = "lingcheng.log"

_CONFIGURED = False


def _resolve_log_level() -> int:
    """从环境变量 LINGCHENG_LOG_LEVEL 解析级别，非法或未设置时默认为 INFO。"""
    name = (os.getenv("LINGCHENG_LOG_LEVEL") or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def setup_lingcheng_logging(
    project_root: Optional[Path] = None,
    log_filename: str = _DEFAULT_LOG_NAME,
    *,
    console: bool = True,
) -> logging.Logger:
    """配置 lingcheng 命名空间日志：文件 + 可选控制台；不记录 API Key 等敏感字段由调用方保证。

    日志级别由环境变量 LINGCHENG_LOG_LEVEL 控制（如 DEBUG、INFO），默认 INFO。
    """
    global _CONFIGURED
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    log_path = project_root.resolve() / log_filename

    level = _resolve_log_level()
    logg = logging.getLogger("lingcheng")
    logg.setLevel(level)

    if _CONFIGURED:
        for h in logg.handlers:
            h.setLevel(level)
        return logg

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logg.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logg.addHandler(sh)

    logg.propagate = False
    _CONFIGURED = True

    logg.info(
        "lingcheng logging initialized path=%s console=%s level=%s",
        log_path,
        console,
        logging.getLevelName(level),
    )
    for h in logg.handlers:
        try:
            h.flush()
        except OSError:
            pass
    return logg


def get_logger(name: str) -> logging.Logger:
    """返回子 logger，name 例如 'agent.router'、'ui.gradio'、'tool.12306'。"""
    return logging.getLogger(f"lingcheng.{name}")
