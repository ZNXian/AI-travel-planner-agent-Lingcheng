"""
文件名: src/agent/tools/__init__.py
用途: 工具子包，集中导出外部接口与 Mock 数据访问函数。
对外暴露:
  - search_trains: 12306 高铁查询
  - search_flights: 机票查询（百炼联网搜索 MCP）
  - web_search: 通用联网搜索
  - get_hotels: Mock 酒店查询
  - get_attractions: Mock 景点查询
  - search_hotels_flyai / search_pois_flyai: 飞猪 FlyAI CLI 封装
"""

from src.agent.tools.flyai_api import search_hotels_flyai, search_pois_flyai
from src.agent.tools.mcp_12306 import search_trains
from src.agent.tools.mock_data import get_attractions, get_hotels
from src.agent.tools.web_search import search_flights, web_search

__all__ = [
    "search_trains",
    "search_flights",
    "web_search",
    "get_hotels",
    "get_attractions",
    "search_hotels_flyai",
    "search_pois_flyai",
]
