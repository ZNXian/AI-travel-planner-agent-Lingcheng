"""
文件名: preference.py
用途: 偏好收集节点。从用户最新消息中抽取偏好字段（含增量调整意图），
      若仍有缺失则把追问问题写入 state.pending_question 由 response 节点最终输出。
对外暴露:
  - preference_node(state) -> dict: LangGraph 节点函数
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agent.llm import call_llm_json, call_llm_text
from src.agent.state import REQUIRED_PREFERENCE_FIELDS, AgentState, merge_preferences


_EXTRACT_SYSTEM = (
    "你是一个旅行偏好抽取助手。请从用户消息中抽取下述字段（没有提到的留空字符串）：\n"
    "- destination: 目的地城市名（仅城市名，比如 '北京'）\n"
    "- origin: 出发地城市名\n"
    "- days: 旅行天数（整数，如 3）\n"
    "- budget_level: 预算档次，仅取 '经济'/'普通'/'豪华' 之一\n"
    "- travel_style: 旅行节奏，仅取 '特种兵'/'普通'/'悠闲' 之一\n"
    "- transport_mode: 出行方式，仅取 '高铁'/'飞机' 之一（提到火车/动车也归为高铁，提到航班/机票归为飞机）\n"
    "- depart_date: 出发日期，YYYY-MM-DD 格式\n"
    "请仅输出 JSON 对象，键名必须严格使用上述英文。"
)


_FOLLOWUP_SYSTEM = (
    "你是灵程旅行助手。基于已知偏好与缺失字段，用一句话自然友好地向用户追问 1-2 个最关键的问题，"
    "不要罗列字段名，要像朋友聊天一样。中文回复。"
)


_TRANSPORT_KEYWORDS = {
    "高铁": "高铁",
    "动车": "高铁",
    "火车": "高铁",
    "飞机": "飞机",
    "航班": "飞机",
    "机票": "飞机",
}

_BUDGET_KEYWORDS = {
    "经济": "经济",
    "穷游": "经济",
    "便宜": "经济",
    "普通": "普通",
    "舒适": "普通",
    "豪华": "豪华",
    "高端": "豪华",
    "奢华": "豪华",
}

_STYLE_KEYWORDS = {
    "特种兵": "特种兵",
    "暴走": "特种兵",
    "悠闲": "悠闲",
    "慢": "悠闲",
}


def _last_human_text(messages: List[BaseMessage]) -> str:
    """取出最近一条 HumanMessage 的文本，没有则返回空串。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _heuristic_extract(text: str) -> Dict[str, Any]:
    """关键字兜底抽取，避免 LLM 不可用时完全卡住。"""
    if not text:
        return {}
    extracted: Dict[str, Any] = {}
    for key, value in _TRANSPORT_KEYWORDS.items():
        if key in text:
            extracted["transport_mode"] = value
            break
    for key, value in _BUDGET_KEYWORDS.items():
        if key in text:
            extracted["budget_level"] = value
            break
    for key, value in _STYLE_KEYWORDS.items():
        if key in text:
            extracted["travel_style"] = value
            break
    days_match = re.search(r"(\d+)\s*天", text)
    if days_match:
        try:
            extracted["days"] = int(days_match.group(1))
        except ValueError:
            pass
    return extracted


def _missing_required(prefs: Dict[str, Any]) -> List[str]:
    """返回必填字段中尚未填写的字段名列表。"""
    return [field for field in REQUIRED_PREFERENCE_FIELDS if not prefs.get(field)]


def _invalidate_caches(state: AgentState, changed: Set[str]) -> Dict[str, Any]:
    """根据偏好变化清空对应缓存与下游产物，返回更新后的 state 片段。"""
    cache: Dict[str, Any] = dict(state.get("last_search_cache") or {})
    confirmed = state.get("confirmed_destination")
    final_itinerary = state.get("final_itinerary")
    attraction_candidates = state.get("attraction_candidates")
    selected_attractions = state.get("selected_attractions")

    if not changed:
        return {
            "last_search_cache": cache,
            "confirmed_destination": confirmed,
            "final_itinerary": final_itinerary,
            "attraction_candidates": attraction_candidates,
            "selected_attractions": selected_attractions,
        }

    if "destination" in changed:
        confirmed = None
        cache.pop("transport", None)
        cache.pop("lodging", None)
        final_itinerary = None
        attraction_candidates = None
        selected_attractions = None
    if "transport_mode" in changed or "origin" in changed or "depart_date" in changed:
        cache.pop("transport", None)
        final_itinerary = None
    if "budget_level" in changed:
        cache.pop("lodging", None)
        final_itinerary = None
    if "days" in changed or "travel_style" in changed:
        final_itinerary = None
        selected_attractions = None

    return {
        "last_search_cache": cache,
        "confirmed_destination": confirmed,
        "final_itinerary": final_itinerary,
        "attraction_candidates": attraction_candidates,
        "selected_attractions": selected_attractions,
    }


def preference_node(state: AgentState) -> Dict[str, Any]:
    """偏好收集节点：抽取最新消息里的偏好，必要时把追问问题写入 pending_question。"""
    messages: List[BaseMessage] = list(state.get("messages") or [])
    preferences: Dict[str, Any] = dict(state.get("preferences") or {})
    thinking_steps: List[str] = list(state.get("thinking_steps") or [])

    user_text = _last_human_text(messages)
    extracted: Dict[str, Any] = {}

    if user_text:
        llm_extracted = call_llm_json(
            messages=[
                SystemMessage(content=_EXTRACT_SYSTEM),
                HumanMessage(content=user_text),
            ],
            fallback={},
        )
        if isinstance(llm_extracted, dict):
            extracted = {k: v for k, v in llm_extracted.items() if v not in (None, "")}
        for key, value in _heuristic_extract(user_text).items():
            extracted.setdefault(key, value)

    changed = merge_preferences(preferences, extracted)
    invalidated = _invalidate_caches(state, changed)

    summary_known = "、".join(f"{k}={v}" for k, v in preferences.items()) or "（暂无）"
    if changed:
        thinking_steps.append(
            f"[偏好收集] 检测到偏好更新：{', '.join(sorted(changed))}；当前偏好：{summary_known}"
        )
    else:
        thinking_steps.append(f"[偏好收集] 当前已知偏好：{summary_known}")

    missing = _missing_required(preferences)
    pending_question: Any = None
    pending_action: Any = state.get("pending_action")

    if missing:
        question_prompt = (
            f"已知偏好：{summary_known}\n"
            f"还缺少：{', '.join(missing)}\n"
            f"请用一句中文友好地追问其中最重要的 1-2 个。"
        )
        pending_question = call_llm_text(
            messages=[
                SystemMessage(content=_FOLLOWUP_SYSTEM),
                HumanMessage(content=question_prompt),
            ],
            fallback=f"为了帮你规划，我想先了解一下：{ '、'.join(missing[:2]) }，你倾向哪种？",
        ).strip()
        thinking_steps.append(f"[偏好收集] 仍缺少 {', '.join(missing)}，准备向用户追问。")
        pending_action = "ask_more"
    else:
        thinking_steps.append("[偏好收集] 必要偏好已齐全，进入下一步。")
        pending_action = None

    return {
        "preferences": preferences,
        "thinking_steps": thinking_steps,
        "last_search_cache": invalidated["last_search_cache"],
        "confirmed_destination": invalidated["confirmed_destination"],
        "final_itinerary": invalidated["final_itinerary"],
        "attraction_candidates": invalidated["attraction_candidates"],
        "selected_attractions": invalidated["selected_attractions"],
        "pending_action": pending_action,
        "pending_question": pending_question,
    }
