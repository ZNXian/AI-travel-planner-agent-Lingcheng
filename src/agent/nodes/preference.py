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
from src.lingcheng_logging import get_logger


_LOG = get_logger("agent.preference")


_EXTRACT_SYSTEM = (
    "你是一个旅行偏好抽取助手。请从用户消息中抽取下述字段（没有提到的留空字符串）：\n"
    "- destination: 目的地城市名（仅城市名，比如 '北京'）\n"
    "- origin: 出发地城市名\n"
    "- days: 旅行天数（整数，如 3）\n"
    "- budget_level: 预算档次，仅取 '经济'/'普通'/'豪华' 之一；"
    "若用户明确表示'没有具体预算/不限/无所谓/随便/都行/看情况'等，请填 '普通'。\n"
    "- travel_style: 旅行节奏，仅取 '特种兵'/'普通'/'悠闲' 之一\n"
    "- transport_mode: 出行方式，仅取 '高铁'/'飞机' 之一（提到火车/动车也归为高铁，提到航班/机票归为飞机）\n"
    "- depart_date: 出发日期，YYYY-MM-DD 格式（仅当用户给出可解析的具体日期时填写）\n"
    "- depart_date_text: 大致出行时间（季节/月份/节假日等模糊描述），如 '春天'、'9月'、'国庆假期'、'暑假'。"
    "若同时给了 depart_date，也尽量补上对应的季节或月份。用户表示'没想好/还没确定/随时'时留空。\n"
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

_BUDGET_SKIP_PATTERNS = (
    "没有具体预算",
    "没具体预算",
    "没什么预算",
    "没预算限制",
    "不限预算",
    "预算不限",
    "无所谓预算",
    "没想过预算",
    "预算随意",
    "预算无所谓",
    "预算都行",
    "预算看情况",
)

_BUDGET_GENERIC_SKIP_PATTERNS = (
    "随便",
    "都行",
    "看情况",
    "无所谓",
)

_COARSE_DATE_PATTERNS = (
    "春天", "夏天", "秋天", "冬天",
    "春季", "夏季", "秋季", "冬季",
    "春节", "元旦", "清明", "五一",
    "端午", "中秋", "国庆", "暑假", "寒假",
)

_DEST_REJECT_KEYWORDS = (
    "换一批", "换一个", "重新推荐", "重新推", "再推荐", "再推",
    "不满意", "不喜欢", "不感兴趣", "都不喜欢", "都不行", "都不要",
    "换个", "换城市", "再来", "别的城市", "其它城市", "其他城市",
)

_DEST_REFINE_KEYWORDS = (
    "北方", "南方", "沿海", "海边", "内陆", "山区", "山里",
    "雪", "温泉", "沙漠", "草原", "高原", "湖", "河",
    "古镇", "历史", "文化", "古城", "现代", "国际化",
    "小众", "冷门", "热门", "网红",
    "大城市", "县城", "小城",
    "江南", "西北", "西南", "东北", "华东", "华南", "华中",
    "面食", "米饭", "辣", "清淡", "海鲜", "烧烤", "面馆", "粤菜", "川菜", "湘菜",
    "湿热", "干燥", "凉爽", "避暑", "暖和", "温暖", "寒冷", "下雪",
)

_DEST_REMOVE_KEYWORDS = (
    "不要", "不需要", "去掉", "去除", "取消", "不再", "别要", "别再",
    "撤掉", "撤回", "删掉", "删除", "去掉那个", "不想要",
)


def _build_dest_feedback_system(
    candidate_names: List[str],
    existing_preferences: List[str],
) -> str:
    """构造目的地反馈分类的 system prompt（候选城市与已知偏好内联其中）。"""
    cand_str = "、".join(candidate_names) if candidate_names else "（无）"
    prefs_str = (
        "、".join(f"\"{p}\"" for p in existing_preferences)
        if existing_preferences
        else "（无）"
    )
    return (
        f"你是旅行助手。当前用户正在面对你之前推荐的候选目的地：{cand_str}。\n"
        f"已知用户之前明确表达过的目的地偏好（累积列表）：{prefs_str}。\n"
        "请判断用户最新一句话对候选的态度，并仅输出严格 JSON：\n"
        '{"intent":"confirm|switch|refine|reject|other",'
        '"chosen_city":"","feedback":"",'
        '"mismatched_cities":[],"removed_preferences":[]}\n'
        "字段说明：\n"
        "- intent:\n"
        "  * confirm = 用户从候选中明确选定一个；\n"
        "  * switch  = 用户提到候选之外的具体城市名作为目的地；\n"
        "  * refine  = 用户没点名某个城市，但补充了**与目的地相关**的偏好"
        "（区域/饮食/气候/景观/城市类型/历史/海边/北方南方/小众/大城市等），暗示当前候选有不合适的；\n"
        "  * reject  = 用户明确否定（'不满意/换一批/都不喜欢/不感兴趣' 等）但未给具体偏好；\n"
        "  * other   = 仅补充天数/预算/出发地/出行方式/出发日期等**与目的地无关**的字段；"
        "或仅仅是想撤回之前的偏好而无新偏好。\n"
        "- chosen_city: confirm 时填候选中的那个城市名；switch 时填新城市名；其它情况留空字符串。\n"
        "- feedback: refine/reject 时用一两句中文总结用户对目的地的**新增**偏好"
        "（如\"想去北方吃面食\"、\"不爱古镇\"），其它情况留空字符串；"
        "**不要**把已经在已知偏好里的旧条目重复进去。\n"
        f"- mismatched_cities: 从当前候选 [{cand_str}] 中挑出**明显不符合**用户最新偏好的城市；"
        "如果整组候选都不符合，把全部候选名都放进去；refine/reject 必须给出非空数组（至少 1 个）；"
        "confirm/switch/other 一律填空数组。**只能输出当前候选中的城市名，不要编造其它城市**。\n"
        f"- removed_preferences: 用户当前消息里**明确要求去除/取消/不要**的之前偏好；"
        f"只能从已知偏好列表 [{prefs_str}] 里挑选**完整字符串项**或其语义子串；"
        "举例：已知偏好=[\"想去北方吃面食\",\"凉爽避暑\"]，用户说\"不要避暑了\" → "
        "removed_preferences=[\"凉爽避暑\"]。不删则空数组。\n"
        "只输出 JSON，不要 markdown 代码围栏，不要前后解释。"
    )


def _heuristic_dest_feedback(
    text: str,
    candidate_names: List[str],
    existing_preferences: List[str],
) -> Dict[str, Any]:
    """关键字兜底：识别用户对目的地候选的反馈意图，并尝试推断 removed_preferences。"""
    removed: List[str] = []
    if text and any(kw in text for kw in _DEST_REMOVE_KEYWORDS):
        for pref in existing_preferences:
            if not pref:
                continue
            if pref in text:
                removed.append(pref)
                continue
            for token in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", pref):
                if len(token) >= 2 and token in text:
                    removed.append(pref)
                    break
        seen: Set[str] = set()
        removed = [p for p in removed if not (p in seen or seen.add(p))]

    empty: Dict[str, Any] = {
        "intent": "other",
        "chosen_city": "",
        "feedback": "",
        "mismatched_cities": [],
        "removed_preferences": removed,
    }
    if not text:
        return empty
    if any(kw in text for kw in _DEST_REJECT_KEYWORDS):
        return {
            "intent": "reject",
            "chosen_city": "",
            "feedback": text[:200],
            "mismatched_cities": list(candidate_names),
            "removed_preferences": removed,
        }
    if removed and not any(kw in text for kw in _DEST_REFINE_KEYWORDS):
        return empty
    if any(kw in text for kw in _DEST_REFINE_KEYWORDS):
        return {
            "intent": "refine",
            "chosen_city": "",
            "feedback": text[:200],
            "mismatched_cities": list(candidate_names),
            "removed_preferences": removed,
        }
    for name in candidate_names:
        if name and name in text:
            return {
                "intent": "confirm",
                "chosen_city": name,
                "feedback": "",
                "mismatched_cities": [],
                "removed_preferences": removed,
            }
    return empty


def _classify_destination_feedback(
    text: str,
    candidates: List[Dict[str, Any]],
    existing_preferences: List[str],
) -> Dict[str, Any]:
    """对"用户面对候选目的地"的回复做意图分类：LLM 优先，失败回退启发式。"""
    candidate_names: List[str] = [
        (c.get("name") or "").strip()
        for c in (candidates or [])
        if isinstance(c, dict) and c.get("name")
    ]
    prefs_clean: List[str] = [p for p in (existing_preferences or []) if p]
    if not text:
        return {
            "intent": "other",
            "chosen_city": "",
            "feedback": "",
            "mismatched_cities": [],
            "removed_preferences": [],
        }

    llm_result = call_llm_json(
        messages=[
            SystemMessage(
                content=_build_dest_feedback_system(candidate_names, prefs_clean)
            ),
            HumanMessage(content=text),
        ],
        fallback={},
    )
    if (
        isinstance(llm_result, dict)
        and llm_result.get("intent")
        in {"confirm", "switch", "refine", "reject", "other"}
    ):
        intent = llm_result["intent"]
        chosen = (llm_result.get("chosen_city") or "").strip()
        feedback = (llm_result.get("feedback") or "").strip()
        raw_mismatched = llm_result.get("mismatched_cities")
        if isinstance(raw_mismatched, list):
            cand_set = set(candidate_names)
            mismatched = [
                m.strip() for m in raw_mismatched
                if isinstance(m, str) and m.strip() and m.strip() in cand_set
            ]
        else:
            mismatched = []

        raw_removed = llm_result.get("removed_preferences")
        removed: List[str] = []
        if isinstance(raw_removed, list) and prefs_clean:
            for entry in raw_removed:
                if not isinstance(entry, str):
                    continue
                token = entry.strip()
                if not token:
                    continue
                for pref in prefs_clean:
                    if token == pref or (len(token) >= 2 and token in pref):
                        if pref not in removed:
                            removed.append(pref)
                        break
                else:
                    if token in prefs_clean and token not in removed:
                        removed.append(token)

        if intent == "confirm" and chosen and chosen not in candidate_names:
            intent = "switch"
        if intent == "switch" and not chosen:
            intent = "other"
        if intent == "other" and feedback:
            intent = "refine"
        if intent in {"refine", "reject"} and not mismatched:
            mismatched = list(candidate_names)
        if intent in {"confirm", "switch", "other"}:
            mismatched = []

        return {
            "intent": intent,
            "chosen_city": chosen,
            "feedback": feedback,
            "mismatched_cities": mismatched,
            "removed_preferences": removed,
        }

    return _heuristic_dest_feedback(text, candidate_names, prefs_clean)


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
    if "budget_level" not in extracted:
        if any(p in text for p in _BUDGET_SKIP_PATTERNS):
            extracted["budget_level"] = "普通"
        elif "预算" in text and any(p in text for p in _BUDGET_GENERIC_SKIP_PATTERNS):
            extracted["budget_level"] = "普通"
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
    month_match = re.search(r"(1[0-2]|[1-9])\s*月", text)
    if month_match:
        extracted["depart_date_text"] = f"{month_match.group(1)}月"
    if "depart_date_text" not in extracted:
        for kw in _COARSE_DATE_PATTERNS:
            if kw in text:
                extracted["depart_date_text"] = kw
                break
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
    if (
        "transport_mode" in changed
        or "origin" in changed
        or "depart_date" in changed
        or "depart_date_text" in changed
    ):
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

    existing_candidates: List[Dict[str, Any]] = list(
        state.get("destination_candidates") or []
    )
    rejected_destinations: List[str] = list(state.get("rejected_destinations") or [])
    destination_preferences: List[str] = [
        p
        for p in (state.get("destination_preferences") or [])
        if isinstance(p, str) and p.strip()
    ]

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

    in_dest_feedback_phase = (
        bool(existing_candidates) and not state.get("confirmed_destination")
    )
    dest_intent = "other"
    dest_chosen = ""
    dest_clear_candidates = False
    dest_mismatched: List[str] = []
    dest_removed: List[str] = []
    if in_dest_feedback_phase:
        feedback_result = _classify_destination_feedback(
            user_text, existing_candidates, destination_preferences
        )
        dest_intent = feedback_result.get("intent", "other")
        dest_chosen = (feedback_result.get("chosen_city") or "").strip()
        feedback_text = (feedback_result.get("feedback") or "").strip()
        dest_mismatched = list(feedback_result.get("mismatched_cities") or [])
        dest_removed = list(feedback_result.get("removed_preferences") or [])
        candidate_names = [
            (c.get("name") or "").strip()
            for c in existing_candidates
            if isinstance(c, dict) and c.get("name")
        ]

        if dest_removed:
            removed_set = set(dest_removed)
            destination_preferences = [
                p for p in destination_preferences if p not in removed_set
            ]
            thinking_steps.append(
                "[偏好收集] 已按用户要求移除偏好："
                + "、".join(dest_removed)
                + "；剩余偏好："
                + ("、".join(destination_preferences) if destination_preferences else "（无）")
            )

        if dest_intent in ("confirm", "switch") and dest_chosen:
            extracted["destination"] = dest_chosen
        elif dest_intent in ("refine", "reject"):
            extracted.pop("destination", None)
            dest_clear_candidates = True
            cities_to_blacklist = dest_mismatched or candidate_names
            for name in cities_to_blacklist:
                if name and name not in rejected_destinations:
                    rejected_destinations.append(name)
            blacklist_str = (
                "、".join(cities_to_blacklist) if cities_to_blacklist else "（无）"
            )
            if dest_intent == "refine":
                if feedback_text and feedback_text not in destination_preferences:
                    destination_preferences.append(feedback_text)
                thinking_steps.append(
                    "[偏好收集] 目的地反馈：检测到目的地偏好补充"
                    + (f"（{feedback_text}）" if feedback_text else "")
                    + f"，已加入黑名单：{blacklist_str}；"
                    + "累积偏好："
                    + ("、".join(destination_preferences) if destination_preferences else "（无）")
                    + "，将重新推荐。"
                )
            else:
                thinking_steps.append(
                    "[偏好收集] 目的地反馈：用户表示不满意，已加入黑名单："
                    + f"{blacklist_str}；保留累积偏好："
                    + ("、".join(destination_preferences) if destination_preferences else "（无）")
                    + "，将重新推荐。"
                )
        else:
            thinking_steps.append(
                f"[偏好收集] 目的地反馈：intent={dest_intent}，未触发重新推荐。"
            )

        _LOG.info(
            "pref_dest_feedback intent=%s chosen=%s mismatched=%s removed=%s prefs_after=%s rejected_after=%s",
            dest_intent,
            dest_chosen or "-",
            dest_mismatched,
            dest_removed,
            destination_preferences,
            rejected_destinations,
        )

    changed = merge_preferences(preferences, extracted)
    invalidated = _invalidate_caches(state, changed)

    summary_known = "、".join(f"{k}={v}" for k, v in preferences.items()) or "（暂无）"
    if changed:
        thinking_steps.append(
            f"[偏好收集] 检测到偏好更新：{', '.join(sorted(changed))}；当前偏好：{summary_known}"
        )
    else:
        thinking_steps.append(f"[偏好收集] 当前已知偏好：{summary_known}")

    if (
        "budget_level" in changed
        and preferences.get("budget_level") == "普通"
        and user_text
        and (
            any(p in user_text for p in _BUDGET_SKIP_PATTERNS)
            or (
                "预算" in user_text
                and any(p in user_text for p in _BUDGET_GENERIC_SKIP_PATTERNS)
            )
        )
    ):
        thinking_steps.append("[偏好收集] 用户未指明预算，按 '普通' 默认处理。")

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

    result: Dict[str, Any] = {
        "preferences": preferences,
        "thinking_steps": thinking_steps,
        "last_search_cache": invalidated["last_search_cache"],
        "confirmed_destination": invalidated["confirmed_destination"],
        "final_itinerary": invalidated["final_itinerary"],
        "attraction_candidates": invalidated["attraction_candidates"],
        "selected_attractions": invalidated["selected_attractions"],
        "pending_action": pending_action,
        "pending_question": pending_question,
        "rejected_destinations": rejected_destinations,
        "destination_preferences": destination_preferences,
    }
    if dest_clear_candidates:
        result["destination_candidates"] = None
    return result
