"""
文件名: lodging.py
用途: 酒店推荐节点。若配置了 FLYAI_API_KEY 则通过 flyai CLI 查询真实酒店；
      否则或失败时回退 Mock 数据。结果写入 state.last_search_cache['lodging']。
对外暴露:
  - lodging_node(state) -> dict: LangGraph 节点函数
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from src.agent.state import AgentState
from src.agent.tools.flyai_api import search_hotels_flyai
from src.agent.tools.mock_data import get_hotels


def lodging_node(state: AgentState) -> Dict[str, Any]:
    """酒店推荐节点：优先 FlyAI，失败或无 Key 时用 Mock。"""
    preferences: Dict[str, Any] = dict(state.get("preferences") or {})
    thinking_steps: List[str] = list(state.get("thinking_steps") or [])
    cache: Dict[str, Any] = dict(state.get("last_search_cache") or {})

    destination = state.get("confirmed_destination") or preferences.get("destination")
    if not destination:
        thinking_steps.append("[酒店] 暂无确认目的地，跳过酒店推荐。")
        return {"thinking_steps": thinking_steps, "last_search_cache": cache}

    budget_level = preferences.get("budget_level")
    depart_date = preferences.get("depart_date")
    days = preferences.get("days")

    hotels: List[Dict[str, Any]] = []
    source = "mock"
    fly_msg = ""

    if os.getenv("FLYAI_API_KEY"):
        try:
            fr = search_hotels_flyai(
                destination,
                budget_level=budget_level,
                depart_date=depart_date if isinstance(depart_date, str) else None,
                days=days,
            )
            fly_msg = fr.get("message") or ""
            if fr.get("ok") and fr.get("items"):
                hotels = list(fr["items"])[:3]
                source = fr.get("source", "flyai")
        except Exception as exc:
            fly_msg = type(exc).__name__

    if not hotels:
        try:
            hotels = get_hotels(destination, budget_level)
            if os.getenv("FLYAI_API_KEY") and source != "flyai":
                source = "mock_fallback"
            else:
                source = "mock"
        except Exception as exc:
            hotels = []
            thinking_steps.append(f"[酒店] Mock 数据读取异常：{type(exc).__name__}")

    payload: Dict[str, Any] = {
        "destination": destination,
        "budget_level": budget_level,
        "items": hotels,
        "source": source,
        "flyai_message": fly_msg if os.getenv("FLYAI_API_KEY") else "",
    }
    cache["lodging"] = payload

    if hotels:
        thinking_steps.append(
            f"[酒店] 已为 {destination}（预算: {budget_level or '不限'}）"
            f"筛选 {len(hotels)} 家酒店，来源：{source}。"
        )
    else:
        thinking_steps.append(
            f"[酒店] {destination} 暂无酒店结果（来源: {source}）。"
            "请检查 Node/npx 与 FLYAI_API_KEY，或稍后重试。"
        )

    return {
        "thinking_steps": thinking_steps,
        "last_search_cache": cache,
    }
