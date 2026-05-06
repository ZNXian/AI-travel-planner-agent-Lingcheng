"""
文件名: attraction_sampler.py
用途: 景点采样与选择节点。在用户确认目的地后，先 LLM 生成 6-8 个候选景点；
      再次进入时识别用户回复（具体选择 / 换一批 / 换个城市），更新状态。
对外暴露:
  - attraction_sampler_node(state) -> dict: LangGraph 节点函数
  - PENDING_FLAG: 通知路由"已展示候选、等待用户选择"的 pending_action 值
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agent.llm import call_llm_json
from src.agent.state import AgentState
from src.agent.tools.flyai_api import search_pois_flyai
from src.agent.tools.mock_data import get_attractions
from src.agent.tools.web_search import web_search


PENDING_FLAG = "await_attraction_selection"

_SWITCH_CITY_KEYWORDS = (
    "换个城市",
    "换城市",
    "换目的地",
    "换地方",
    "换个目的地",
    "其他城市",
)

_RESHUFFLE_KEYWORDS = (
    "换一批",
    "再来一批",
    "重新推荐",
    "再换一组",
    "都不感兴趣",
    "都不喜欢",
    "都不想去",
    "换批",
    "另外的景点",
    "其他景点",
)

_CANDIDATE_SYSTEM = (
    "你是一个国内旅行景点推荐助手。若用户消息中附带「联网检索摘要」，请优先从中挑选"
    "与用户出发日期及行程窗口相近或重叠的新开景点、快闪展、临时市集、限定活动、当季特展等，"
    "与经典地标混搭，输出 6 到 8 个候选。"
    "若用户未提供出行时间或没有联网检索摘要，请仅推荐该城市稳定开放、四季皆宜的代表性经典景点，"
    "避免编造时间敏感的临展或限定活动。"
    "仅输出 JSON 数组，每个元素包含："
    "name(景点名), highlight(一句话亮点), duration(建议游玩时长，如 '半天'), "
    "ticket_range(门票区间，如 '免费' 或 '¥55-100'，不确定时填 '见现场'), "
    "emoji(单个 emoji，与景点风格相关)。不要带任何额外文字或解释。"
)

_WEB_SNIPPET_MAX = 6000

_MATCH_SYSTEM = (
    "你是一个意图解析助手。下面给出 JSON 数组形式的候选景点（含 name 字段），以及"
    "用户最新回复。请判断用户想去哪些候选景点，仅返回一个 JSON 数组，元素为候选 name 字符串。"
    "若用户没有明确选择，返回 []。不要返回不在候选中的名称。"
)


def _last_human_text(messages: List[BaseMessage]) -> str:
    """取出最近一条 HumanMessage 的文本，没有则返回空串。"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _has_time_hint(prefs: Dict[str, Any]) -> bool:
    """只要 depart_date 或 depart_date_text 任一非空，就视为「有出行时间」。"""
    return bool(
        (prefs.get("depart_date") or "").strip()
        or (prefs.get("depart_date_text") or "").strip()
    )


def _build_attraction_search_query(city: str, preferences: Dict[str, Any]) -> str:
    """组装面向百炼联网的检索问句：强调出发日前后新景、快闪、临时活动与行程窗口。"""
    origin = (preferences.get("origin") or "").strip()
    depart = (preferences.get("depart_date") or "").strip()[:10]
    depart_text = (preferences.get("depart_date_text") or "").strip()
    days_raw = preferences.get("days")
    trip_days = 0
    if days_raw is not None and str(days_raw).strip():
        try:
            trip_days = int(days_raw)
        except (TypeError, ValueError):
            trip_days = 0

    window_desc = ""
    if depart:
        try:
            d0 = datetime.strptime(depart, "%Y-%m-%d")
            if trip_days > 0:
                d_end = d0 + timedelta(days=max(trip_days - 1, 0))
                window_desc = (
                    f"用户行程窗口约从 {depart} 到 {d_end.strftime('%Y-%m-%d')}（共约 {trip_days} 天），"
                    "请重点检索该时段内开放、需预约或临期结束的新展、快闪、限定活动、新开园区。"
                )
            else:
                window_desc = (
                    f"用户计划出发/抵达起始日约为 {depart}，请检索该日前后一两周内的新开景点、"
                    "临时展览、市集与短期活动。"
                )
        except ValueError:
            window_desc = (
                f"用户提到日期片段「{depart}」，请结合常识检索 {city} 近期新景与临时活动。"
            )
    elif depart_text:
        window_desc = (
            f"用户大致出行时间为「{depart_text}」，请重点检索该季节/月份/节假日"
            "的限时活动、当季展览、新景点与近郊新晋打卡点。"
        )
    else:
        window_desc = (
            "用户未提供确切出发日期，请检索该城市当前月份及近期官宣的新景点、快闪与临时活动。"
        )

    origin_desc = ""
    if origin:
        origin_desc = (
            f"用户从「{origin}」出发前往「{city}」，可适当包含目的地近郊、当日可达的新晋打卡点。"
        )

    return (
        f"请联网检索中国城市「{city}」的旅游资讯。{window_desc}{origin_desc}"
        "请归纳：新开或热门新景点、博物馆/美术馆临展与特展、主题公园新项目、限时装置艺术、"
        "短期市集或嘉年华、近郊新晋露营地等；每条请写清名称、大致开放时段或展期、是否需预约。"
        "勿编造不存在的官方馆名；不确定的门票写「见官方」。"
    )


def _fetch_web_attraction_context(city: str, preferences: Dict[str, Any]) -> str:
    """调用 web_search（内部走百炼 enable_search）获取摘要文本；失败返回空串。"""
    query = _build_attraction_search_query(city, preferences)
    try:
        result = web_search(query)
    except Exception:
        return ""
    if not result.get("ok"):
        return ""
    answer = (result.get("answer") or "").strip()
    if not answer:
        return ""
    if len(answer) > _WEB_SNIPPET_MAX:
        answer = answer[:_WEB_SNIPPET_MAX] + "\n…（联网摘要已截断）"
    return answer


def _reference_attractions(city: str) -> List[Dict[str, Any]]:
    """优先 FlyAI POI、否则 Mock 景点，作为 LLM 生成候选时的参考资料。"""
    if os.getenv("FLYAI_API_KEY"):
        try:
            pr = search_pois_flyai(city)
            if pr.get("ok") and pr.get("items"):
                return list(pr["items"])[:10]
        except Exception:
            pass
    try:
        return list(get_attractions(city) or [])
    except Exception:
        return []


def _normalize_candidate(item: Any) -> Optional[Dict[str, Any]]:
    """规范化单条 LLM 输出，缺字段给默认值；非 dict 返回 None。"""
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    if not name:
        return None
    return {
        "name": name,
        "highlight": str(item.get("highlight") or item.get("description") or "").strip(),
        "duration": str(item.get("duration") or "2-3 小时").strip(),
        "ticket_range": str(item.get("ticket_range") or "见现场").strip(),
        "emoji": str(item.get("emoji") or "📍").strip(),
    }


def _generate_candidates(
    city: str,
    preferences: Dict[str, Any],
    exclude: Optional[List[Dict[str, Any]]] = None,
    thinking_steps: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """调用 LLM 生成 6-8 个景点候选；exclude 为上一轮候选名称集合，避免重复。"""
    exclude_names = sorted({(c or {}).get("name", "") for c in (exclude or []) if c})
    refs = _reference_attractions(city)
    ref_lines = [
        f"- {a.get('name','?')}: {a.get('description') or ''}".strip()
        for a in refs[:8]
    ]
    web_ctx = ""
    has_time = _has_time_hint(preferences)
    has_key = bool((os.getenv("DASHSCOPE_API_KEY") or "").strip())
    if has_key and has_time:
        web_ctx = _fetch_web_attraction_context(city, preferences)
        if web_ctx and thinking_steps is not None:
            thinking_steps.append(
                "[景点选择] 检测到出行时间，已并入百炼联网检索摘要（新点 + 经典）。"
            )
    elif thinking_steps is not None and not has_time:
        thinking_steps.append(
            "[景点选择] 未提供出行时间，使用百炼非联网模式仅推荐该城市经典景点。"
        )

    depart_value = (preferences.get("depart_date") or "").strip() or (
        preferences.get("depart_date_text") or "").strip() or "未知"

    user_prompt = (
        f"目的地：{city}\n"
        f"出发地：{preferences.get('origin') or '未知'}\n"
        f"计划出发日：{depart_value}\n"
        f"用户天数：{preferences.get('days') or '未知'}\n"
        f"用户节奏：{preferences.get('travel_style') or '未知'}\n"
        f"用户预算：{preferences.get('budget_level') or '未知'}\n"
        + (f"已提供过的景点（请勿重复推荐）：{', '.join(exclude_names)}\n" if exclude_names else "")
        + ("可参考的本地景点（不必照搬）：\n" + "\n".join(ref_lines) + "\n" if ref_lines else "")
        + (
            "联网检索摘要（优先采用其中与行程日期相近的新景、临展、快闪与临时活动）：\n"
            f"{web_ctx}\n"
            if web_ctx
            else ""
        )
        + (
            "请给出 6 到 8 个新的候选景点，并按要求只输出 JSON 数组。"
            if has_time
            else "用户未提供出行时间，请仅推荐该城市稳定开放、四季皆宜的 6 到 8 个经典景点，"
                 "不要包含临展/限定活动；按要求只输出 JSON 数组。"
        )
    )

    fallback = [
        {"name": "城市地标 1", "highlight": "代表性打卡点", "duration": "半天",
         "ticket_range": "见现场", "emoji": "📍"},
        {"name": "本地美食街区", "highlight": "尝当地小吃", "duration": "2 小时",
         "ticket_range": "免费", "emoji": "🍜"},
        {"name": "热门博物馆", "highlight": "了解城市历史", "duration": "2-3 小时",
         "ticket_range": "免费(需预约)", "emoji": "🏛️"},
        {"name": "近郊自然景区", "highlight": "亲近自然", "duration": "半天",
         "ticket_range": "¥40-100", "emoji": "🌿"},
        # {"name": "夜景观光点", "highlight": "适合傍晚到访", "duration": "1-2 小时",
        #  "ticket_range": "见现场", "emoji": "🌃"},
        # {"name": "特色文创街区", "highlight": "拍照与购物", "duration": "2 小时",
        #  "ticket_range": "免费", "emoji": "🛍️"},
    ]

    raw = call_llm_json(
        messages=[
            SystemMessage(content=_CANDIDATE_SYSTEM),
            HumanMessage(content=user_prompt),
        ],
        fallback=fallback,
    )
    if not isinstance(raw, list):
        raw = fallback

    items: List[Dict[str, Any]] = []
    seen_names: set = set()
    for entry in raw:
        norm = _normalize_candidate(entry)
        if norm and norm["name"] not in seen_names:
            items.append(norm)
            seen_names.add(norm["name"])
        if len(items) >= 8:
            break
    if len(items) < 6:
        for entry in fallback:
            norm = _normalize_candidate(entry)
            if norm and norm["name"] not in seen_names:
                items.append(norm)
                seen_names.add(norm["name"])
            if len(items) >= 6:
                break
    return items[:8]


def _match_selections(
    user_text: str, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """从用户文本里挑出选中的候选景点；先做包含匹配，匹配为空再用 LLM 兜底。"""
    if not user_text or not candidates:
        return []

    text = user_text.strip()
    chosen: List[Dict[str, Any]] = []
    seen: set = set()
    for cand in candidates:
        name = (cand.get("name") or "").strip()
        if not name:
            continue
        if name in text and name not in seen:
            chosen.append(cand)
            seen.add(name)
    if chosen:
        return chosen

    name_to_cand = {(c.get("name") or ""): c for c in candidates if c.get("name")}
    cand_brief = [{"name": c.get("name", "")} for c in candidates]
    llm_names = call_llm_json(
        messages=[
            SystemMessage(content=_MATCH_SYSTEM),
            HumanMessage(
                content=(
                    f"候选景点：{cand_brief}\n用户回复：{user_text}"
                )
            ),
        ],
        fallback=[],
    )
    if not isinstance(llm_names, list):
        return []
    out: List[Dict[str, Any]] = []
    for name in llm_names:
        cand = name_to_cand.get(str(name).strip())
        if cand and cand not in out:
            out.append(cand)
    return out


def attraction_sampler_node(state: AgentState) -> Dict[str, Any]:
    """景点采样节点：未生成候选→生成；已生成→识别用户选择/换一批/换个城市。"""
    preferences: Dict[str, Any] = dict(state.get("preferences") or {})
    thinking_steps: List[str] = list(state.get("thinking_steps") or [])
    candidates: List[Dict[str, Any]] = list(state.get("attraction_candidates") or [])
    city: str = (state.get("confirmed_destination") or "").strip()

    if not city:
        thinking_steps.append("[景点选择] 暂无确认目的地，跳过。")
        return {"thinking_steps": thinking_steps}

    user_text = _last_human_text(state.get("messages") or [])

    if candidates and user_text:
        if any(kw in user_text for kw in _SWITCH_CITY_KEYWORDS):
            thinking_steps.append(
                f"[景点选择] 用户希望更换城市，清空 {city} 的候选景点并回到目的地推荐。"
            )
            new_prefs = dict(preferences)
            new_prefs.pop("destination", None)
            return {
                "thinking_steps": thinking_steps,
                "preferences": new_prefs,
                "confirmed_destination": None,
                "destination_candidates": None,
                "attraction_candidates": None,
                "selected_attractions": None,
                "last_search_cache": {},
                "final_itinerary": None,
                "pending_action": None,
            }

        if any(kw in user_text for kw in _RESHUFFLE_KEYWORDS):
            new_list = _generate_candidates(
                city, preferences, exclude=candidates, thinking_steps=thinking_steps
            )
            thinking_steps.append(
                f"[景点选择] 用户希望更换一批，已为 {city} 重新生成 {len(new_list)} 个景点。"
            )
            return {
                "thinking_steps": thinking_steps,
                "attraction_candidates": new_list,
                "selected_attractions": None,
                "pending_action": PENDING_FLAG,
            }

        selected = _match_selections(user_text, candidates)
        if selected:
            names = "、".join(c.get("name", "?") for c in selected)
            thinking_steps.append(
                f"[景点选择] 已识别用户选择 {len(selected)} 个景点：{names}。"
            )
            return {
                "thinking_steps": thinking_steps,
                "selected_attractions": selected,
                "attraction_candidates": None,
                "pending_action": None,
            }

        thinking_steps.append(
            "[景点选择] 暂未识别到具体选择，将再次提示用户挑选 2-3 个候选。"
        )
        return {
            "thinking_steps": thinking_steps,
            "pending_action": PENDING_FLAG,
        }

    new_list = _generate_candidates(city, preferences, thinking_steps=thinking_steps)
    thinking_steps.append(
        f"[景点选择] 已为 {city} 生成 {len(new_list)} 个候选景点，等待用户选择 2-3 个。"
    )
    return {
        "thinking_steps": thinking_steps,
        "attraction_candidates": new_list,
        "selected_attractions": None,
        "pending_action": PENDING_FLAG,
    }
