"""
文件名: llm.py
用途: 统一封装阿里百炼 qwen-max（OpenAI 兼容模式）ChatOpenAI 客户端，避免每个节点重复初始化。
对外暴露:
  - get_llm: 返回一个共享的 ChatOpenAI 实例
  - call_llm_json: 让 LLM 严格输出 JSON 并解析（带异常兜底）
  - call_llm_text: 让 LLM 返回自然语言文本
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.lingcheng_logging import get_logger


_LLM_SINGLETON: Optional[ChatOpenAI] = None
_LOG = get_logger("llm")


def _validate_env() -> None:
    """检查必要环境变量是否齐全，缺失时抛出友好错误（不打印 Key 值本身）。"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "缺少环境变量 DASHSCOPE_API_KEY，请将 .env.example 复制为 .env 并填写后重试。"
        )


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """获取 qwen-max 的 ChatOpenAI 客户端（单例），temperature 可临时调整。"""
    global _LLM_SINGLETON
    _validate_env()
    model = os.getenv("QWEN_MODEL", "qwen-max")
    if _LLM_SINGLETON is None or _LLM_SINGLETON.temperature != temperature:
        _LOG.info(
            "llm_client_init model=%s temperature=%s base_url_configured=%s",
            model,
            temperature,
            bool(os.getenv("DASHSCOPE_BASE_URL")),
        )
        _LLM_SINGLETON = ChatOpenAI(
            model=model,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=temperature,
            timeout=60,
        )
    return _LLM_SINGLETON


def _extract_json_block(text: str) -> str:
    """从 LLM 自由文本中抽取最外层 JSON 对象或数组字符串。

    优先解析 markdown 代码块；否则从首个 `[` 或 `{` 起用 JSONDecoder.raw_decode
    截取**恰好一段**合法 JSON（避免旧版贪婪 `\\{[\\s\\S]*\\}` 把 `[{...},{...}]` 截成单对象片段）。
    """
    if not text:
        return ""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if fence:
        return fence.group(1).strip()
    dec = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch not in "[{":
            continue
        try:
            _, end = dec.raw_decode(stripped, i)
            return stripped[i:end]
        except json.JSONDecodeError:
            continue
    return stripped


def call_llm_json(
    messages: List[BaseMessage],
    system_hint: str = "你必须仅输出严格合法的 JSON，不要带任何解释或前后缀。",
    temperature: float = 0.2,
    fallback: Optional[Any] = None,
) -> Any:
    """让 LLM 输出 JSON 并解析。失败时返回 fallback（默认 None），不会抛异常打断主流程。"""
    model = os.getenv("QWEN_MODEL", "qwen-max")
    _LOG.info(
        "llm_invoke_json start model=%s temperature=%s message_count=%s",
        model,
        temperature,
        len(messages) + 1,
    )
    t0 = time.perf_counter()
    text = ""
    try:
        llm = get_llm(temperature=temperature)
        full_messages: List[BaseMessage] = [SystemMessage(content=system_hint)] + list(messages)
        _LOG.debug(
            "llm_invoke_full_messages=%s",
            full_messages
        )
        resp = llm.invoke(full_messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        parsed = json.loads(_extract_json_block(text))
        _LOG.info(
            "llm_invoke_json ok elapsed_ms=%.1f parsed_type=%s",
            (time.perf_counter() - t0) * 1000,
            type(parsed).__name__,
        )
        _LOG.debug(
            "llm_invoke_json parsed=%s",
            parsed
        )
        return parsed
    except Exception as exc:
        _LOG.info(
            "llm_invoke_json fail elapsed_ms=%.1f err=%s",
            (time.perf_counter() - t0) * 1000,
            type(exc).__name__,
        )
        raw = text or ""
        n = len(raw)
        head = raw[:200]
        tail = raw[-20:] if n >= 20 else raw
        _LOG.debug(
            "llm_invoke_json raw_preview err=%s raw_chars=%s raw_first200=%r raw_last20=%r",
            type(exc).__name__,
            n,
            head,
            tail,
        )
        return fallback


def call_llm_text(
    messages: List[BaseMessage],
    temperature: float = 0.7,
    fallback: str = "",
) -> str:
    """让 LLM 返回自然语言文本，失败时返回 fallback。"""
    model = os.getenv("QWEN_MODEL", "qwen-max")
    _LOG.info(
        "llm_invoke_text start model=%s temperature=%s message_count=%s",
        model,
        temperature,
        len(messages),
    )
    t0 = time.perf_counter()
    try:
        llm = get_llm(temperature=temperature)
        resp = llm.invoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        _LOG.info(
            "llm_invoke_text ok elapsed_ms=%.1f response_chars=%s",
            (time.perf_counter() - t0) * 1000,
            len(text) if isinstance(text, str) else 0,
        )
        return text
    except Exception as exc:
        _LOG.info(
            "llm_invoke_text fail elapsed_ms=%.1f err=%s",
            (time.perf_counter() - t0) * 1000,
            type(exc).__name__,
        )
        return fallback or f"[LLM 调用失败：{type(exc).__name__}]"


__all__ = ["get_llm", "call_llm_json", "call_llm_text"]


def _safe_dict_preview(data: Dict[str, Any]) -> Dict[str, Any]:
    """复制并屏蔽敏感键，便于日志安全展示（当前未直接调用，预留给调试时使用）。"""
    masked: Dict[str, Any] = {}
    sensitive = {"api_key", "key", "token", "password", "secret"}
    for key, value in (data or {}).items():
        if key.lower() in sensitive:
            masked[key] = "***"
        else:
            masked[key] = value
    return masked
