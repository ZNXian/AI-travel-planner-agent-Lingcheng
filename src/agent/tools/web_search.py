"""
文件名: web_search.py
用途: 百炼联网搜索 MCP 的封装。机票查询仍为 Mock；通用搜索在配置
      DASHSCOPE_API_KEY 时走 `_real_web_search_via_dashscope`（enable_search），
      失败或无 Key 时降级 Mock。
对外暴露:
  - search_flights(origin, destination, date): 机票查询
  - web_search(query): 通用联网搜索
  - _real_web_search_via_dashscope(query): 百炼联网（供高级场景直接调用）

================ 真实接入指引（百炼联网搜索 MCP） ================

阿里百炼提供"联网搜索"内置工具，可通过 OpenAI 兼容模式的 chat.completions 调用：

    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": "北京到上海 2025-07-01 机票价格"}],
        extra_body={
            "enable_search": True,            # 开启百炼内置联网搜索
            # 或者通过 MCP 工具方式（控制台启用后可用）：
            # "tools": [{"type": "mcp", "mcp_server": "search"}],
        },
    )

返回值里的 message.content 即包含搜索结果的自然语言摘要。如果需要结构化字段，
可以在 prompt 里要求模型按 JSON 模板输出后再 json.loads。

注意：
  - 每次调用都会消耗 token，注意预算控制。
  - 若控制台尚未开通联网搜索权限，调用会返回错误，本文件已统一降级到 Mock。

==============================================================================================
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.lingcheng_logging import get_logger


_LOG = get_logger("tool.websearch")


_MOCK_FLIGHTS_TEMPLATE: List[Dict[str, Any]] = [
    {
        "flight_no": "CA1501",
        "airline": "中国国航",
        "depart_time": "07:30",
        "arrive_time": "09:50",
        "aircraft": "A330",
        "price_economy": 980.0,
        "price_business": 3680.0,
        "punctuality": "92%",
        "note": "始发航班 / 准点率高",
    },
    {
        "flight_no": "MU5101",
        "airline": "中国东航",
        "depart_time": "10:00",
        "arrive_time": "12:20",
        "aircraft": "B787",
        "price_economy": 1080.0,
        "price_business": 3980.0,
        "punctuality": "88%",
        "note": "宽体机",
    },
    {
        "flight_no": "HU7605",
        "airline": "海南航空",
        "depart_time": "14:35",
        "arrive_time": "16:55",
        "aircraft": "A320",
        "price_economy": 760.0,
        "price_business": 2680.0,
        "punctuality": "90%",
        "note": "性价比之选",
    },
    {
        "flight_no": "MF8102",
        "airline": "厦门航空",
        "depart_time": "20:10",
        "arrive_time": "22:30",
        "aircraft": "B737",
        "price_economy": 680.0,
        "price_business": 2280.0,
        "punctuality": "85%",
        "note": "晚班特价",
    },
]


def _adjust_price_by_date(price: float, date: Optional[str]) -> float:
    """节假日小幅上浮，模拟真实票价波动。"""
    if not date:
        return price
    try:
        weekday = datetime.strptime(date, "%Y-%m-%d").weekday()
    except ValueError:
        return price
    if weekday >= 5:
        return round(price * 1.15, 1)
    return price


# ---------- 真实接入占位 ----------
# def _real_search_flights_via_dashscope(
#     origin: str, destination: str, date: Optional[str]
# ) -> List[Dict[str, Any]]:
#     """通过百炼联网搜索查询机票，要求模型按 JSON 输出。"""
#     import json
#     import os
#     from openai import OpenAI
#
#     client = OpenAI(
#         api_key=os.getenv("DASHSCOPE_API_KEY"),
#         base_url=os.getenv("DASHSCOPE_BASE_URL"),
#     )
#     prompt = (
#         f"请联网搜索 {origin} 到 {destination} 在 {date} 的航班，"
#         "并仅以 JSON 数组返回，每项含 flight_no/airline/depart_time/arrive_time/price_economy。"
#     )
#     resp = client.chat.completions.create(
#         model=os.getenv("QWEN_MODEL", "qwen-max"),
#         messages=[{"role": "user", "content": prompt}],
#         extra_body={"enable_search": True},
#     )
#     return json.loads(resp.choices[0].message.content)


def search_flights(
    origin: str, destination: str, date: Optional[str] = None
) -> Dict[str, Any]:
    """机票查询，返回 {"ok", "flights", "source", "message"}。失败时降级到 Mock。"""
    t0 = time.perf_counter()
    _LOG.info(
        "websearch_mcp search_flights start origin=%s destination=%s date=%s",
        origin,
        destination,
        date,
    )
    if not origin or not destination:
        _LOG.info(
            "websearch_mcp search_flights skip elapsed_ms=%.1f reason=empty_city",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "ok": False,
            "flights": [],
            "source": "mock",
            "message": "起讫城市不能为空。",
        }

    try:
        travel_date = date or datetime.now().strftime("%Y-%m-%d")
        flights: List[Dict[str, Any]] = []
        for tpl in _MOCK_FLIGHTS_TEMPLATE:
            copy = dict(tpl)
            copy["from"] = origin
            copy["to"] = destination
            copy["date"] = travel_date
            copy["price_economy"] = _adjust_price_by_date(copy["price_economy"], travel_date)
            copy["price_business"] = _adjust_price_by_date(copy["price_business"], travel_date)
            flights.append(copy)
        out = {
            "ok": True,
            "flights": flights,
            "source": "mock",
            "message": "已使用 Mock 数据返回 4 个候选航班（接入百炼联网搜索 MCP 后将被替换）。",
        }
        _LOG.info(
            "websearch_mcp search_flights ok elapsed_ms=%.1f source=%s flight_count=%s",
            (time.perf_counter() - t0) * 1000,
            out["source"],
            len(flights),
        )
        return out
    except Exception as exc:
        _LOG.info(
            "websearch_mcp search_flights error elapsed_ms=%.1f err=%s",
            (time.perf_counter() - t0) * 1000,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "flights": [],
            "source": "mock",
            "message": f"机票查询失败：{type(exc).__name__}",
        }


def _dashscope_message_text(content: Any) -> str:
    """从 chat.completions 的 message.content 抽取纯文本（兼容 str 或多段 content）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content).strip()


def _real_web_search_via_dashscope(query: str) -> str:
    """通过百炼 OpenAI 兼容接口开启 enable_search，返回联网摘要纯文本。"""
    from openai import OpenAI

    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")

    base_url = (os.getenv("DASHSCOPE_BASE_URL") or "").strip() or (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = (os.getenv("QWEN_MODEL") or "qwen-max").strip() or "qwen-max"
    client = OpenAI(api_key=api_key, base_url=base_url)
    _LOG.info("websearch_mcp dashscope_search invoke model=%s", model)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query.strip()}],
        extra_body={"enable_search": True},
    )
    msg = resp.choices[0].message
    text = _dashscope_message_text(getattr(msg, "content", None))
    if not text:
        raise RuntimeError("empty search response")
    return text


def web_search(query: str) -> Dict[str, Any]:
    """通用联网搜索，返回 {"ok", "answer", "source", "message"}。

    若已配置 DASHSCOPE_API_KEY，则调用百炼内置联网（enable_search）；
    否则或未开通权限导致失败时，降级为 Mock 占位文案。
    """
    t0 = time.perf_counter()
    q = (query or "").strip()
    _LOG.info(
        "websearch_mcp web_search start query_chars=%s",
        len(q),
    )
    if not q:
        _LOG.info(
            "websearch_mcp web_search skip elapsed_ms=%.1f reason=empty_query",
            (time.perf_counter() - t0) * 1000,
        )
        return {"ok": False, "answer": "", "source": "mock", "message": "查询内容为空。"}

    if not (os.getenv("DASHSCOPE_API_KEY") or "").strip():
        out = {
            "ok": True,
            "answer": (
                f"[Mock] 关于「{q[:200]}」的搜索结果：未配置 DASHSCOPE_API_KEY，"
                "无法启用百炼联网搜索。"
            ),
            "source": "mock",
            "message": "Mock：缺少 API Key。",
        }
        _LOG.info(
            "websearch_mcp web_search ok elapsed_ms=%.1f source=%s answer_chars=%s",
            (time.perf_counter() - t0) * 1000,
            out["source"],
            len(out.get("answer") or ""),
        )
        return out

    try:
        answer = _real_web_search_via_dashscope(q)
        out: Dict[str, Any] = {
            "ok": True,
            "answer": answer,
            "source": "dashscope_search",
            "message": "百炼联网搜索完成。",
        }
        _LOG.info(
            "websearch_mcp web_search ok elapsed_ms=%.1f source=%s answer_chars=%s",
            (time.perf_counter() - t0) * 1000,
            out["source"],
            len(out.get("answer") or ""),
        )
        return out
    except Exception as exc:
        _LOG.info(
            "websearch_mcp web_search dashscope_fail elapsed_ms=%.1f err=%s",
            (time.perf_counter() - t0) * 1000,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "answer": "",
            "source": "mock",
            "message": f"百炼联网搜索失败：{type(exc).__name__}",
        }
