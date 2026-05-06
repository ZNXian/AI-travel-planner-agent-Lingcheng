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
    "请仅输出 JSON 数组，例如："
    '[{"name":"杭州","reason":"江南水乡，秋季桂花最美","highlights":["西湖","灵隐寺","龙井茶"]}]'
)


def _build_recommend_prompt(preferences: Dict[str, Any]) -> str:
    """根据已知偏好组装推荐请求文本。"""
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
    return "用户偏好：\n" + ("\n".join(parts) if parts else "（未提供）")


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

    prompt = _build_recommend_prompt(preferences)
    result = call_llm_json(
        messages=[
            SystemMessage(content=_RECOMMEND_SYSTEM),
            HumanMessage(content=prompt),
        ],
        fallback=[
            {
                "name": "杭州",
                "reason": "江南山水兼具，节奏可快可慢",
                "highlights": ["西湖", "灵隐寺", "龙井茶"],
            },
            {
                "name": "成都",
                "reason": "美食与休闲并重，熊猫之都",
                "highlights": ["熊猫基地", "宽窄巷子", "川菜"],
            },
        ],
    )

    if not isinstance(result, list) or not result:
        result = [
            {"name": "杭州", "reason": "经典之选", "highlights": ["西湖", "西溪"]}
        ]

    candidates = result[:3]
    thinking_steps.append(
        f"[目的地] 已生成 {len(candidates)} 个候选：{ '、'.join(c.get('name','?') for c in candidates) }，等待用户确认。"
    )

    return {
        "thinking_steps": thinking_steps,
        "destination_candidates": candidates,
        "confirmed_destination": None,
        "pending_action": "await_destination",
    }
