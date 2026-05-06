"""
文件名: mcp_12306.py
用途: 12306 高铁查询工具封装。当前为 Mock 实现，文件内附"真实 MCP 接入"完整步骤说明，
      切换到真实实现时只需替换 _real_search_trains_via_mcp 即可。
对外暴露:
  - search_trains(origin, destination, date): 查询车次列表，失败时降级到 Mock 数据
  - MCP_COMMAND: 启动 12306 MCP 的命令行（npx -y 12306-mcp）

================ 真实 MCP 接入指引（实现时取消注释 _real_search_trains_via_mcp 即可） ================

12306-mcp 是基于 Model Context Protocol 的 stdio JSON-RPC server，调用流程：

1. 用 subprocess.Popen 启动: ["npx", "-y", "12306-mcp"]，stdin/stdout 设为 PIPE。
2. 发送 JSON-RPC 'initialize' 请求，等待 server 返回能力信息。
3. 发送 'tools/list' 获取可用工具，找到查询车次的工具名（一般是
   "12306_search_tickets" 或类似）以及它的入参 schema。
4. 发送 'tools/call' 请求，参数包含 from / to / date 等字段。
5. 解析返回值（通常是文本/JSON-in-text），提取车次列表后再优雅关闭进程。

注意：
  - Windows 下需保证 PATH 中能找到 node 与 npx；否则 Popen 会抛 FileNotFoundError，
    本文件已统一捕获并回退到 Mock 数据。
  - JSON-RPC 消息以换行符分隔（NDJSON）；发送时记得 flush。
  - 整个会话需要保持 process 存活，不要每次查询都重启。可在 search_trains 内做单例缓存。

==============================================================================================
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.lingcheng_logging import get_logger


_LOG = get_logger("tool.12306")

MCP_COMMAND: List[str] = ["npx", "-y", "12306-mcp"]
"""启动 12306 MCP server 的命令行。"""


_MOCK_TRAINS: List[Dict[str, Any]] = [
    {
        "train_no": "G1",
        "depart_time": "08:00",
        "arrive_time": "12:28",
        "duration": "4小时28分",
        "second_class_price": 553.0,
        "first_class_price": 933.0,
        "business_class_price": 1748.0,
        "second_class_seats": "有票",
        "note": "复兴号 / 始发",
    },
    {
        "train_no": "G7",
        "depart_time": "09:00",
        "arrive_time": "13:32",
        "duration": "4小时32分",
        "second_class_price": 553.0,
        "first_class_price": 933.0,
        "business_class_price": 1748.0,
        "second_class_seats": "余票紧张",
        "note": "复兴号",
    },
    {
        "train_no": "G15",
        "depart_time": "11:00",
        "arrive_time": "15:38",
        "duration": "4小时38分",
        "second_class_price": 553.0,
        "first_class_price": 933.0,
        "business_class_price": 1748.0,
        "second_class_seats": "有票",
        "note": "复兴号 / 经停少",
    },
    {
        "train_no": "G55",
        "depart_time": "14:00",
        "arrive_time": "18:48",
        "duration": "4小时48分",
        "second_class_price": 553.0,
        "first_class_price": 933.0,
        "business_class_price": 1748.0,
        "second_class_seats": "有票",
        "note": "下午发车",
    },
]


def _build_mock_response(
    origin: str, destination: str, date: Optional[str]
) -> List[Dict[str, Any]]:
    """根据起讫城市与日期生成一份伪车次列表（节假日小幅涨价以模拟真实数据）。"""
    travel_date = date or datetime.now().strftime("%Y-%m-%d")
    multiplier = 1.0
    try:
        weekday = datetime.strptime(travel_date, "%Y-%m-%d").weekday()
        if weekday >= 5:
            multiplier = 1.05
    except ValueError:
        pass

    result: List[Dict[str, Any]] = []
    for train in _MOCK_TRAINS:
        copy = dict(train)
        copy["from"] = origin
        copy["to"] = destination
        copy["date"] = travel_date
        for price_key in ("second_class_price", "first_class_price", "business_class_price"):
            copy[price_key] = round(copy[price_key] * multiplier, 1)
        result.append(copy)
    return result


# ---------- 真实接入占位（实现时取消注释并替换 search_trains 内的调用） ----------
# def _real_search_trains_via_mcp(
#     origin: str, destination: str, date: Optional[str]
# ) -> List[Dict[str, Any]]:
#     """通过 stdio JSON-RPC 调用 12306-mcp 真实查询车次。"""
#     import json
#     import subprocess
#
#     proc = subprocess.Popen(
#         MCP_COMMAND,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#         encoding="utf-8",
#     )
#     try:
#         def send(req: Dict[str, Any]) -> Dict[str, Any]:
#             proc.stdin.write(json.dumps(req) + "\n")
#             proc.stdin.flush()
#             return json.loads(proc.stdout.readline())
#
#         send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
#         send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
#         result = send({
#             "jsonrpc": "2.0",
#             "id": 3,
#             "method": "tools/call",
#             "params": {
#                 "name": "search_tickets",
#                 "arguments": {"from": origin, "to": destination, "date": date},
#             },
#         })
#         # TODO: 根据真实 server 的返回结构解析为统一格式
#         return result.get("result", {}).get("content", [])
#     finally:
#         proc.terminate()


def search_trains(
    origin: str, destination: str, date: Optional[str] = None
) -> Dict[str, Any]:
    """查询高铁车次。返回 {"ok": bool, "trains": [...], "source": "mock|mcp", "message": str}。

    当前默认走 Mock；如需切换真实 MCP，参考文件顶部接入指引并启用
    _real_search_trains_via_mcp 即可。失败统一降级到 Mock，避免阻塞主流程。
    """
    t0 = time.perf_counter()
    _LOG.info(
        "12306_mcp search_trains start origin=%s destination=%s date=%s",
        origin,
        destination,
        date,
    )
    if not origin or not destination:
        _LOG.info(
            "12306_mcp search_trains skip elapsed_ms=%.1f reason=empty_origin_or_destination",
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "ok": False,
            "trains": [],
            "source": "mock",
            "message": "起讫站不能为空。",
        }

    try:
        trains = _build_mock_response(origin, destination, date)
        out = {
            "ok": True,
            "trains": trains,
            "source": "mock",
            "message": "已使用 Mock 数据返回 4 个候选车次（接入真实 12306 MCP 后将被替换）。",
        }
        _LOG.info(
            "12306_mcp search_trains ok elapsed_ms=%.1f source=%s train_count=%s",
            (time.perf_counter() - t0) * 1000,
            out["source"],
            len(trains),
        )
        return out
    except Exception as exc:  # 兜底：任何异常都不应阻塞主流程
        _LOG.info(
            "12306_mcp search_trains error elapsed_ms=%.1f err=%s",
            (time.perf_counter() - t0) * 1000,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "trains": [],
            "source": "mock",
            "message": f"高铁查询失败：{type(exc).__name__}",
        }
