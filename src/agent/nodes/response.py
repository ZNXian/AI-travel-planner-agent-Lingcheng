"""
文件名: response.py
用途: 最终回复节点。把当前轮次的 thinking_steps 折叠展示，并将
      pending_question / destination_candidates / 交通 / 酒店 / 行程 等内容
      拼接成一条 AIMessage 输出，最后清空 thinking_steps 与一次性字段。
对外暴露:
  - response_node(state) -> dict: LangGraph 节点函数
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage

from src.agent.nodes.attraction_sampler import PENDING_FLAG as ATTRACTION_PENDING_FLAG
from src.agent.state import AgentState


def _format_thinking_block(steps: List[str]) -> str:
    """把思考步骤封装到 details 折叠块（HTML+Markdown 混排，Gradio Chatbot 支持）。"""
    if not steps:
        return ""
    body = "\n".join(f"- {s}" for s in steps)
    return (
        "<details>\n"
        "<summary>查看我的思考过程</summary>\n\n"
        f"{body}\n\n"
        "</details>\n"
    )


def _format_destination_candidates(candidates: List[Dict[str, Any]]) -> str:
    """把目的地候选列表渲染为 Markdown 段落，并提示用户确认。"""
    if not candidates:
        return ""
    lines = ["### 候选目的地", ""]
    for idx, item in enumerate(candidates, 1):
        name = item.get("name", "?")
        reason = item.get("reason", "")
        highlights = item.get("highlights") or []
        hl_text = "、".join(highlights) if highlights else ""
        lines.append(f"{idx}. **{name}** — {reason}")
        if hl_text:
            lines.append(f"   - 亮点：{hl_text}")
    lines.append("")
    lines.append("回复城市名（如 *杭州*）即可确认；不满意可以让我重新推荐~")
    return "\n".join(lines)


def _format_attraction_candidates(
    city: str, candidates: List[Dict[str, Any]]
) -> str:
    """把候选景点列表渲染为 Markdown 段，并提示用户挑选 2-3 个。"""
    if not candidates:
        return ""
    head_city = city or "目的地"
    lines = [f"### {head_city}·候选景点", ""]
    for item in candidates:
        emoji = (item.get("emoji") or "📍").strip() or "📍"
        name = item.get("name", "?")
        highlight = (item.get("highlight") or "").strip()
        duration = (item.get("duration") or "").strip()
        ticket = (item.get("ticket_range") or "").strip()
        suffix_parts: List[str] = []
        if duration:
            suffix_parts.append(f"建议 {duration}")
        if ticket:
            suffix_parts.append(f"门票 {ticket}")
        suffix = "（" + "，".join(suffix_parts) + "）" if suffix_parts else ""
        if highlight:
            lines.append(f"{emoji} **{name}** — {highlight}{suffix}")
        else:
            lines.append(f"{emoji} **{name}**{suffix}")
    lines.append("")
    lines.append(
        "请选出你最感兴趣的 2-3 个（直接说景点名即可）；"
        "如果都不太喜欢，可以说「换一批」；想换城市就说「换个城市」。"
    )
    return "\n".join(lines)


def _format_transport(transport: Dict[str, Any]) -> str:
    """把交通查询结果渲染为 Markdown 表格段。"""
    if not transport or not transport.get("items"):
        return ""
    mode = transport.get("mode", "高铁")
    items = transport["items"][:3]
    head = (
        f"### 交通推荐（{mode}）\n\n"
        f"{transport.get('origin','?')} → {transport.get('destination','?')}"
        f"，{transport.get('date','?')}（数据来源：{transport.get('source','mock')}）\n\n"
    )
    if mode == "飞机":
        rows = ["| 航班 | 航司 | 时刻 | 经济舱 | 备注 |", "| --- | --- | --- | --- | --- |"]
        for it in items:
            rows.append(
                f"| {it.get('flight_no','-')} | {it.get('airline','-')} |"
                f" {it.get('depart_time','-')}–{it.get('arrive_time','-')} |"
                f" ¥{it.get('price_economy','-')} | {it.get('note','')} |"
            )
    else:
        rows = ["| 车次 | 时刻 | 历时 | 二等座 | 余票 |", "| --- | --- | --- | --- | --- |"]
        for it in items:
            rows.append(
                f"| {it.get('train_no','-')} |"
                f" {it.get('depart_time','-')}–{it.get('arrive_time','-')} |"
                f" {it.get('duration','-')} | ¥{it.get('second_class_price','-')} |"
                f" {it.get('second_class_seats','-')} |"
            )
    return head + "\n".join(rows)


def _format_lodging(lodging: Dict[str, Any]) -> str:
    """把酒店推荐渲染为 Markdown 列表。"""
    if not lodging or not lodging.get("items"):
        return ""
    head = f"### 酒店推荐（{lodging.get('budget_level') or '不限预算'}）\n\n"
    lines: List[str] = []
    for hotel in lodging["items"]:
        tags = "、".join(hotel.get("tags") or [])
        lines.append(
            f"- **{hotel.get('name','?')}** ({hotel.get('level','?')}) — "
            f"{hotel.get('price_range','?')}，位置：{hotel.get('location','?')}"
            + (f"；标签：{tags}" if tags else "")
        )
    return head + "\n".join(lines)


def response_node(state: AgentState) -> Dict[str, Any]:
    """最终回复节点：组装思考折叠块 + 候选/交通/酒店/行程 → 一条 AIMessage 输出。"""
    thinking_steps: List[str] = list(state.get("thinking_steps") or [])
    pending_question: str = (state.get("pending_question") or "").strip()
    pending_action: str = (state.get("pending_action") or "").strip()
    candidates: List[Dict[str, Any]] = state.get("destination_candidates") or []
    attraction_candidates: List[Dict[str, Any]] = (
        state.get("attraction_candidates") or []
    )
    selected_attractions: List[Dict[str, Any]] = (
        state.get("selected_attractions") or []
    )
    cache: Dict[str, Any] = state.get("last_search_cache") or {}
    final_itinerary: str = (state.get("final_itinerary") or "").strip()
    confirmed: str = (state.get("confirmed_destination") or "").strip()

    sections: List[str] = []
    sections.append(_format_thinking_block(thinking_steps))

    awaiting_attractions = (
        bool(attraction_candidates)
        and not selected_attractions
        and pending_action == ATTRACTION_PENDING_FLAG
    )

    if pending_question:
        sections.append(pending_question)
    elif candidates:
        sections.append(_format_destination_candidates(candidates))
    elif awaiting_attractions:
        if confirmed:
            sections.append(f"### 已确认目的地：{confirmed}")
        sections.append(
            _format_attraction_candidates(confirmed, attraction_candidates)
        )
    else:
        if confirmed:
            sections.append(f"### 已确认目的地：{confirmed}")
        sections.append(_format_transport(cache.get("transport") or {}))
        sections.append(_format_lodging(cache.get("lodging") or {}))
        if final_itinerary:
            sections.append("### 每日行程\n\n" + final_itinerary)
        sections.append(
            "如需调整（比如：*改成豪华版* / *缩短到 3 天* / *换成高铁*），随时告诉我~"
        )

    reply = "\n\n".join(s for s in sections if s).strip()
    if not reply:
        reply = "（暂无可输出的内容，请再描述一下你的旅行想法吧。）"

    return {
        "messages": [AIMessage(content=reply)],
        "thinking_steps": [],
        "pending_question": None,
        "pending_action": None,
    }
