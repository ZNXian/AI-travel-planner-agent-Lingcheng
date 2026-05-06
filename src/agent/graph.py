"""
文件名: graph.py
用途: 定义 LangGraph 状态机，编排灵程旅游规划 Agent 的所有节点。
对外暴露:
  - create_agent_graph(): 编译并返回 LangGraph Runnable
  - NODE_NAMES: 节点名常量
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    attraction_sampler_node,
    destination_node,
    itinerary_node,
    lodging_node,
    preference_node,
    response_node,
    transport_node,
)
from src.agent.nodes.attraction_sampler import PENDING_FLAG as ATTRACTION_PENDING_FLAG
from src.agent.state import REQUIRED_PREFERENCE_FIELDS, AgentState
from src.lingcheng_logging import get_logger


NODE_NAMES = (
    "preference",
    "destination",
    "attraction_sampler",
    "transport",
    "lodging",
    "itinerary",
    "response",
)

_LOG = get_logger("agent.router")
_NODE_LOG = get_logger("agent.node")


def _route_decide(state: AgentState) -> str:
    """纯函数：根据 state 计算下一个节点名（不含日志）。"""
    if state.get("pending_question"):
        return "response"

    preferences: Dict[str, Any] = state.get("preferences") or {}
    if not all(preferences.get(field) for field in REQUIRED_PREFERENCE_FIELDS):
        return "response"

    if not state.get("confirmed_destination"):
        if preferences.get("destination"):
            return "destination"
        if state.get("destination_candidates"):
            return "response"
        return "destination"

    if not state.get("selected_attractions"):
        candidates = state.get("attraction_candidates")
        if candidates and state.get("pending_action") == ATTRACTION_PENDING_FLAG:
            return "response"
        return "attraction_sampler"

    cache: Dict[str, Any] = state.get("last_search_cache") or {}
    if not cache.get("transport"):
        return "transport"
    if not cache.get("lodging"):
        return "lodging"
    if not state.get("final_itinerary"):
        return "itinerary"

    return "response"


def _route(state: AgentState) -> str:
    """根据当前 state 决定下一个节点名，并写入 INFO 路由日志。"""
    next_node = _route_decide(state)
    prefs = state.get("preferences") or {}
    cache = state.get("last_search_cache") or {}
    _LOG.info(
        "route_decision next=%s pending_question=%s has_candidates=%s "
        "confirmed=%s prefs_keys=%s has_transport_cache=%s has_lodging_cache=%s has_itinerary=%s "
        "has_attractions_cands=%s has_selected_attractions=%s pending_action=%s",
        next_node,
        bool(state.get("pending_question")),
        bool(state.get("destination_candidates")),
        state.get("confirmed_destination"),
        list(prefs.keys()),
        bool(cache.get("transport")),
        bool(cache.get("lodging")),
        bool(state.get("final_itinerary")),
        bool(state.get("attraction_candidates")),
        bool(state.get("selected_attractions")),
        state.get("pending_action"),
    )
    return next_node


def _wrap_node(
    name: str, fn: Callable[[AgentState], Dict[str, Any]]
) -> Callable[[AgentState], Dict[str, Any]]:
    """包装 LangGraph 节点：记录进入、离开与耗时（不记录消息全文以免过长）。"""

    def _wrapped(state: AgentState) -> Dict[str, Any]:
        """执行原节点函数并打点日志。"""
        t0 = time.perf_counter()
        msg_count = len(state.get("messages") or [])
        _NODE_LOG.info("node_enter node=%s messages_count=%s", name, msg_count)
        try:
            out = fn(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            keys = list(out.keys()) if isinstance(out, dict) else []
            _NODE_LOG.info(
                "node_leave node=%s elapsed_ms=%.1f return_keys=%s",
                name,
                elapsed_ms,
                keys,
            )
            return out
        except Exception as exc:
            _NODE_LOG.info(
                "node_error node=%s elapsed_ms=%.1f err=%s",
                name,
                (time.perf_counter() - t0) * 1000,
                type(exc).__name__,
            )
            raise

    return _wrapped


def create_agent_graph():
    """编译并返回灵程 Agent 的 LangGraph 状态机。"""
    graph = StateGraph(AgentState)

    graph.add_node("preference", _wrap_node("preference", preference_node))
    graph.add_node("destination", _wrap_node("destination", destination_node))
    graph.add_node(
        "attraction_sampler", _wrap_node("attraction_sampler", attraction_sampler_node)
    )
    graph.add_node("transport", _wrap_node("transport", transport_node))
    graph.add_node("lodging", _wrap_node("lodging", lodging_node))
    graph.add_node("itinerary", _wrap_node("itinerary", itinerary_node))
    graph.add_node("response", _wrap_node("response", response_node))

    graph.add_edge(START, "preference")

    routing_targets = {name: name for name in NODE_NAMES}

    for node in (
        "preference",
        "destination",
        "attraction_sampler",
        "transport",
        "lodging",
        "itinerary",
    ):
        graph.add_conditional_edges(node, _route, routing_targets)

    graph.add_edge("response", END)

    return graph.compile()
