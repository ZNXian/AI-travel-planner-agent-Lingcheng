"""
文件名: src/ui/__init__.py
用途: UI 子包，导出 Gradio 启动入口。
对外暴露: launch
"""

from src.ui.gradio_app import launch

__all__ = ["launch"]
