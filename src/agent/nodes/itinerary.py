"""
文件名: itinerary.py
用途: 行程生成节点。汇总 preferences、交通、酒店、景点信息后调用 qwen-max
      生成 Markdown 格式的每日行程，写入 state.final_itinerary。
对外暴露:
  - itinerary_node(state) -> dict: LangGraph 节点函数
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm import call_llm_text
from src.agent.state import AgentState
from src.agent.tools.flyai_api import search_pois_flyai
from src.agent.tools.mock_data import get_attractions


_ITINERARY_SYSTEM = (
    "你是灵程旅行助手，擅长规划国内旅行。请基于给定的偏好、交通、酒店、景点信息，"
    "生成一份 Markdown 格式的每日行程：\n"
    "- 如提供了「用户已选定景点」，行程必须围绕这些景点安排，不得随意替换或删除；"
    "可结合「其他可选景点」做交通衔接或补足空档，但不要喧宾夺主。\n"
    "- 每天分为 上午 / 下午 / 晚上 三个时段，每个时段给出 1-2 项活动。\n"
    "- 至少包含一餐当地美食推荐（午餐或晚餐）。\n"
    "- 节奏匹配用户的 travel_style：特种兵尽量塞满，悠闲则留出休息时间。\n"
    "- 在开头先用一段 100 字以内的概览说明此次行程的设计思路。\n"
    "- 结尾给出 1-2 条贴心提示（如门票预约、避开高峰）。\n"
    "请使用中文回复。"
)


def _format_transport_brief(transport: Dict[str, Any]) -> str:
    """将交通缓存转成简短文本送给 LLM。"""
    if not transport:
        return "未查询到交通信息。"
    items = transport.get("items") or []
    mode = transport.get("mode", "高铁")
    head = f"{transport.get('origin','?')} → {transport.get('destination','?')} ({transport.get('date','?')})，方式：{mode}"
    if not items:
        return head + "（暂无候选）"
    preview = items[:2]
    return head + "，候选示例：" + json.dumps(preview, ensure_ascii=False)


def _format_lodging_brief(lodging: Dict[str, Any]) -> str:
    """将酒店缓存转成简短文本送给 LLM。"""
    if not lodging:
        return "未查询到酒店信息。"
    items = lodging.get("items") or []
    if not items:
        return f"目的地 {lodging.get('destination','?')} 暂无候选酒店。"
    preview = items[:2]
    return f"预算 {lodging.get('budget_level') or '不限'} 候选酒店：" + json.dumps(
        preview, ensure_ascii=False
    )


def itinerary_node(state: AgentState) -> Dict[str, Any]:
    """行程生成节点：调 LLM 生成 Markdown 每日行程并写入 final_itinerary。"""
    preferences: Dict[str, Any] = dict(state.get("preferences") or {})
    thinking_steps: List[str] = list(state.get("thinking_steps") or [])
    cache: Dict[str, Any] = dict(state.get("last_search_cache") or {})
    destination = state.get("confirmed_destination") or preferences.get("destination")

    if not destination:
        thinking_steps.append("[行程] 暂无确认目的地，无法生成行程。")
        return {"thinking_steps": thinking_steps, "final_itinerary": None}

    days = preferences.get("days") or 3
    travel_style = preferences.get("travel_style") or "普通"
    budget_level = preferences.get("budget_level") or "普通"
    transport_brief = _format_transport_brief(cache.get("transport") or {})
    lodging_brief = _format_lodging_brief(cache.get("lodging") or {})

    attractions: List[Dict[str, Any]] = []
    attractions_note = "Mock"
    if os.getenv("FLYAI_API_KEY"):
        try:
            pr = search_pois_flyai(destination)
            if pr.get("ok") and pr.get("items"):
                attractions = list(pr["items"])[:10]
                attractions_note = "FlyAI search-poi"
        except Exception:
            attractions = []
    if not attractions:
        try:
            attractions = get_attractions(destination)
            attractions_note = "Mock" if attractions else "无"
        except Exception:
            attractions = []
            attractions_note = "无"

    attractions_brief = (
        json.dumps(attractions[:8], ensure_ascii=False)
        if attractions
        else "（暂无景点数据）"
    )

    selected: List[Dict[str, Any]] = list(state.get("selected_attractions") or [])
    if selected:
        selected_names = "、".join(
            (s.get("name") or "?") for s in selected if isinstance(s, dict)
        )
        selected_block = (
            "用户已选定景点（行程必须围绕它们安排）：\n"
            + json.dumps(selected, ensure_ascii=False)
            + "\n"
        )
        selected_summary = f"已选景点：{selected_names}"
    else:
        selected_block = "用户尚未单独挑选景点，可在可选景点中合理选取。\n"
        selected_summary = "无显式选定景点，按可选景点自由组合"

    user_prompt = (
        f"目的地：{destination}\n"
        f"天数：{days} 天\n"
        f"节奏：{travel_style}\n"
        f"预算：{budget_level}\n"
        f"交通信息：{transport_brief}\n"
        f"住宿信息：{lodging_brief}\n"
        f"{selected_block}"
        f"其他可选景点（{attractions_note}）：{attractions_brief}\n\n"
        "请按要求生成 Markdown 行程。"
    )

    text = call_llm_text(
        messages=[
            SystemMessage(content=_ITINERARY_SYSTEM),
            HumanMessage(content=user_prompt),
        ],
        temperature=0.7,
        fallback=(
            f"## {destination} {days} 天行程（兜底版）\n\n"
            "由于 LLM 暂不可用，这里给出一个简单兜底：\n"
            "- Day 1：抵达 → 入住酒店 → 周边漫步 → 当地小吃晚餐\n"
            "- Day 2：核心景点上午场 → 下午文化体验 → 晚上夜景\n"
            "- Day 3：自由活动 → 返程"
        ),
    )

    thinking_steps.append(
        f"[行程] 已基于 {days} 天 / {travel_style} 节奏 / {budget_level} 预算生成行程；{selected_summary}。"
    )

    return {
        "thinking_steps": thinking_steps,
        "final_itinerary": text.strip(),
    }
