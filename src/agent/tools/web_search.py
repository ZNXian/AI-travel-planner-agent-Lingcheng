"""
文件名: web_search.py
用途: 百炼联网搜索 MCP 的封装。当前为 Mock 实现（机票 + 通用搜索），
      文件内附"真实 MCP 接入"步骤，切换到真实实现时只需替换两个 _real_* 函数。
对外暴露:
  - search_flights(origin, destination, date): 机票查询
  - web_search(query): 通用联网搜索

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


# ---------- 真实接入占位 ----------
# def _real_web_search_via_dashscope(query: str) -> str:
#     """通过百炼联网搜索做通用问答，返回纯文本摘要。"""
#     import os
#     from openai import OpenAI
#
#     client = OpenAI(
#         api_key=os.getenv("DASHSCOPE_API_KEY"),
#         base_url=os.getenv("DASHSCOPE_BASE_URL"),
#     )
#     resp = client.chat.completions.create(
#         model=os.getenv("QWEN_MODEL", "qwen-max"),
#         messages=[{"role": "user", "content": query}],
#         extra_body={"enable_search": True},
#     )
#     return resp.choices[0].message.content


def web_search(query: str) -> Dict[str, Any]:
    """通用联网搜索，返回 {"ok", "answer", "source", "message"}。当前为 Mock 占位。"""
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

    try:
        out = {
            "ok": True,
            "answer": (
                f"[Mock] 关于 '{query}' 的搜索结果暂未对接真实联网搜索 MCP。"
                "切换到真实实现后会返回最新的联网搜索摘要。"
            ),
            "source": "mock",
            "message": "Mock 联网搜索响应。",
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
            "websearch_mcp web_search error elapsed_ms=%.1f err=%s",
            (time.perf_counter() - t0) * 1000,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "answer": "",
            "source": "mock",
            "message": f"联网搜索失败：{type(exc).__name__}",
        }
