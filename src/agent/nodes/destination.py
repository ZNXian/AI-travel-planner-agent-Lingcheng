"""
文件名: destination.py
用途: 目的地推荐节点。当 preferences 已包含目的地（用户直接给出）时直接确认；
      否则用 LLM 推荐 1-3 个候选目的地，把候选写入 state.destination_candidates，
      由 response 节点向用户输出供其确认。
对外暴露:
  - destination_node(state) -> dict: LangGraph 节点函数
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm import call_llm_json
from src.agent.state import AgentState


_RECOMMEND_SYSTEM = (
    "你是一个旅行目的地推荐专家。基于用户偏好，请推荐 1 到 3 个国内城市作为候选目的地，"
    "每个候选包含 name(城市名)、reason(一句话推荐理由)、highlights(2-4 个亮点关键词的字符串数组)。"
    "如果提示词里出现 '已被排除的城市'，新候选**不得包含**其中任何一个，请挑选与之风格不同的城市。"
    "如果提示词里出现 '用户最近反馈'，请优先匹配反馈中的偏好（如海边/历史/小众等）。"
    "请仅输出 JSON 数组，不要 markdown 代码围栏、不要前后解释文字、不要使用单引号，键名必须用双引号。"
    "例如："
    '[{"name":"杭州","reason":"江南水乡，秋季桂花最美","highlights":["西湖","灵隐寺","龙井茶"]}]'
)

_DEST_FAIL_USER_MSG = (
    "**目的地推荐失败**：大模型返回的内容无法解析为合法 JSON（常见原因：多写了说明、格式不标准或网络异常）。\n\n"
    "请**直接回复你想去的城市名称**（例如：杭州），我将以此确认目的地并继续规划；"
    "或稍后重试让我再次推荐候选城市。"
)


def _build_recommend_prompt(
    preferences: Dict[str, Any],
    rejected: List[str] | None = None,
    feedback: str | None = None,
) -> str:
    """根据已知偏好组装推荐请求文本，可附带历史拒绝城市与最近反馈。"""
    parts: List[str] = []
    if preferences.get("days"):
        parts.append(f"旅行天数: {preferences['days']} 天")
    if preferences.get("budget_level"):
        parts.append(f"预算档次: {preferences['budget_level']}")
    if preferences.get("travel_style"):
        parts.append(f"旅行节奏: {preferences['travel_style']}")
    if preferences.get("origin"):
        parts.append(f"出发地: {preferences['origin']}")
    if preferences.get("transport_mode"):
        parts.append(f"出行方式偏好: {preferences['transport_mode']}")
    if preferences.get("depart_date_text"):
        parts.append(f"大致出行时间: {preferences['depart_date_text']}")
    sections: List[str] = ["用户偏好：\n" + ("\n".join(parts) if parts else "（未提供）")]
    rejected_clean = [n for n in (rejected or []) if n]
    if rejected_clean:
        sections.append("已被排除的城市（请避开）：" + "、".join(rejected_clean))
    feedback_text = (feedback or "").strip()
    if feedback_text:
        sections.append(f"用户最近反馈：{feedback_text}")
    return "\n\n".join(sections)


def destination_node(state: AgentState) -> Dict[str, Any]:
    """目的地推荐节点：preferences.destination 已存在则直接确认，否则给候选。"""
    preferences: Dict[str, Any] = dict(state.get("preferences") or {})
    thinking_steps: List[str] = list(state.get("thinking_steps") or [])
    confirmed: Any = state.get("confirmed_destination")
    candidates: Any = state.get("destination_candidates")

    if preferences.get("destination") and (
        not confirmed or confirmed != preferences["destination"]
    ):
        confirmed = preferences["destination"]
        thinking_steps.append(f"[目的地] 用户已明确目的地：{confirmed}")
        return {
            "thinking_steps": thinking_steps,
            "confirmed_destination": confirmed,
            "destination_candidates": None,
            "attraction_candidates": None,
            "selected_attractions": None,
            "pending_action": None,
        }

    if confirmed:
        thinking_steps.append(f"[目的地] 沿用已确认的目的地：{confirmed}")
        return {
            "thinking_steps": thinking_steps,
            "confirmed_destination": confirmed,
            "destination_candidates": None,
            "pending_action": None,
        }

    rejected: List[str] = list(state.get("rejected_destinations") or [])
    prefs_list: List[str] = [
        s for s in (state.get("destination_preferences") or [])
        if isinstance(s, str) and s.strip()
    ]
    feedback: str = "；".join(prefs_list)
    prompt = _build_recommend_prompt(preferences, rejected=rejected, feedback=feedback)
    if rejected or feedback:
        thinking_steps.append(
            "[目的地] 重新推荐："
            + (f"避开 {'、'.join(rejected)}" if rejected else "")
            + ("；" if rejected and feedback else "")
            + (f"参考反馈：{feedback}" if feedback else "")
        )
    result = call_llm_json(
        messages=[
            SystemMessage(content=_RECOMMEND_SYSTEM),
            HumanMessage(content=prompt),
        ],
        fallback=None,
    )

    if not isinstance(result, list) or not result:
        thinking_steps.append(
            "[目的地] 大模型调用失败或返回非 JSON 数组，未使用任何兜底城市；请用户直接说城市名或重试。"
        )
        return {
            "thinking_steps": thinking_steps,
            "destination_candidates": None,
            "confirmed_destination": None,
            "pending_action": "await_destination",
            "pending_question": _DEST_FAIL_USER_MSG,
        }

    rejected_set = {n for n in rejected if n}
    filtered = [
        c for c in result
        if isinstance(c, dict) and (c.get("name") or "").strip() not in rejected_set
    ]
    if not filtered:
        filtered = [c for c in result if isinstance(c, dict)]
    candidates = filtered[:3]
    thinking_steps.append(
        f"[目的地] 已生成 {len(candidates)} 个候选：{ '、'.join(c.get('name','?') for c in candidates) }，等待用户确认。"
    )

    return {
        "thinking_steps": thinking_steps,
        "destination_candidates": candidates,
        "confirmed_destination": None,
        "pending_action": "await_destination",
        "pending_question": None,
    }
