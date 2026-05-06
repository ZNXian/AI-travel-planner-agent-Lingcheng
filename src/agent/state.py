"""
文件名: state.py
用途: 定义 LangGraph 共享状态 AgentState 与初始状态工厂，供所有节点读写。
对外暴露:
  - AgentState: 整个图的共享状态 TypedDict
  - REQUIRED_PREFERENCE_FIELDS: 必填偏好字段
  - make_initial_state: 构造一个空白初始状态
  - merge_preferences: 合并新偏好并返回受影响的缓存键集合
"""

from typing import Annotated, Any, Dict, List, Optional, Set, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


REQUIRED_PREFERENCE_FIELDS = ("days", "budget_level", "travel_style")
"""偏好收集节点判定"可进入目的地推荐/后续规划"所必须的字段。

注意：destination 不放在此集合中——用户可以直接说出目的地，
也可以先只给天数/预算/节奏，由 destination 节点推荐候选城市后再确认。
"""


class AgentState(TypedDict, total=False):
    """灵程 Agent 的共享状态。

    字段说明:
      messages: 对话历史（由 add_messages 自动合并）。
      preferences: 已收集到的用户偏好字典。
      thinking_steps: 本轮各节点追加的中文思考步骤。
      last_search_cache: 缓存最近一次的交通/酒店查询结果。
      confirmed_destination: 用户确认的最终目的地。
      final_itinerary: 已生成的 Markdown 完整行程。
      pending_action: 路由器读取的临时跳转标记（增量调整时使用）。
      pending_question: preference 节点希望向用户追问的问题。
      destination_candidates: destination 节点产出的候选目的地列表。
      attraction_candidates: attraction_sampler 节点展示给用户的候选景点列表。
      selected_attractions: 用户最终选择的景点子集（行程节点核心约束）。
    """

    messages: Annotated[List[BaseMessage], add_messages]
    preferences: Dict[str, Any]
    thinking_steps: List[str]
    last_search_cache: Dict[str, Any]
    confirmed_destination: Optional[str]
    final_itinerary: Optional[str]
    pending_action: Optional[str]
    pending_question: Optional[str]
    destination_candidates: Optional[List[Dict[str, Any]]]
    attraction_candidates: Optional[List[Dict[str, Any]]]
    selected_attractions: Optional[List[Dict[str, Any]]]


def make_initial_state() -> AgentState:
    """生成一个全空的 AgentState，用于会话首次创建。"""
    return AgentState(
        messages=[],
        preferences={},
        thinking_steps=[],
        last_search_cache={},
        confirmed_destination=None,
        final_itinerary=None,
        pending_action=None,
        pending_question=None,
        destination_candidates=None,
        attraction_candidates=None,
        selected_attractions=None,
    )


def merge_preferences(
    old: Dict[str, Any], new: Dict[str, Any]
) -> Set[str]:
    """把 new 中的非空字段写入 old，并返回发生变化的字段名集合。"""
    changed: Set[str] = set()
    for key, value in (new or {}).items():
        if value in (None, "", []):
            continue
        if old.get(key) != value:
            old[key] = value
            changed.add(key)
    return changed
