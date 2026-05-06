"""
文件名: transport.py
用途: 交通查询节点。根据偏好里的 transport_mode（高铁/飞机）调用对应的工具，
      把结果缓存到 state.last_search_cache['transport']。失败时降级到 Mock。
对外暴露:
  - transport_node(state) -> dict: LangGraph 节点函数
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.agent.state import AgentState
from src.agent.tools.mcp_12306 import search_trains
from src.agent.tools.web_search import search_flights


def _resolve_depart_date(prefs: Dict[str, Any]) -> str:
    """获取出发日期，缺省取明天，避免日期为空导致查询失败。"""
    raw = (prefs.get("depart_date") or "").strip()
    if raw:
        return raw
    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


def _resolve_origin(prefs: Dict[str, Any]) -> str:
    """获取出发地，缺省按 '北京'。"""
    return (prefs.get("origin") or "北京").strip() or "北京"


def transport_node(state: AgentState) -> Dict[str, Any]:
    """交通查询节点：按 transport_mode 调用 12306 或机票工具，结果写入缓存。"""
    preferences: Dict[str, Any] = dict(state.get("preferences") or {})
    thinking_steps: List[str] = list(state.get("thinking_steps") or [])
    cache: Dict[str, Any] = dict(state.get("last_search_cache") or {})

    destination = state.get("confirmed_destination") or preferences.get("destination")
    if not destination:
        thinking_steps.append("[交通] 暂无确认目的地，跳过交通查询。")
        return {"thinking_steps": thinking_steps, "last_search_cache": cache}

    origin = _resolve_origin(preferences)
    date = _resolve_depart_date(preferences)
    mode = preferences.get("transport_mode") or "高铁"

    transport_payload: Dict[str, Any] = {
        "mode": mode,
        "origin": origin,
        "destination": destination,
        "date": date,
        "items": [],
        "source": "mock",
        "message": "",
    }

    try:
        if mode == "飞机":
            result = search_flights(origin, destination, date)
            transport_payload["items"] = result.get("flights", [])
            transport_payload["source"] = result.get("source", "mock")
            transport_payload["message"] = result.get("message", "")
            count = len(transport_payload["items"])
            thinking_steps.append(
                f"[交通] 联网搜索 MCP 查询 {origin}→{destination} ({date}) 机票，返回 {count} 个候选航班。"
            )
        else:
            result = search_trains(origin, destination, date)
            transport_payload["items"] = result.get("trains", [])
            transport_payload["source"] = result.get("source", "mock")
            transport_payload["message"] = result.get("message", "")
            count = len(transport_payload["items"])
            thinking_steps.append(
                f"[交通] 12306 MCP 查询 {origin}→{destination} ({date}) 高铁，返回 {count} 个车次。"
            )
    except Exception as exc:
        transport_payload["message"] = f"交通查询失败：{type(exc).__name__}"
        thinking_steps.append(f"[交通] 查询失败，已降级到空结果：{type(exc).__name__}")

    cache["transport"] = transport_payload

    return {
        "thinking_steps": thinking_steps,
        "last_search_cache": cache,
    }
