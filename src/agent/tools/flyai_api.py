"""
文件名: flyai_api.py
用途: 通过飞猪官方 @fly-ai/flyai-cli（npx）调用 FlyAI：酒店 search-hotel、景点 search-poi；
      需配置环境变量 FLYAI_API_KEY；失败时返回 ok=False 供上层回退 Mock。
对外暴露:
  - search_hotels_flyai: 按目的地/预算/日期查询酒店，返回统一 items 列表
  - search_pois_flyai: 按城市查询 POI/景点，返回统一 items 列表
  - FLYAI_CLI_PACKAGE: npx 包名常量
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.lingcheng_logging import get_logger


_LOG = get_logger("tool.flyai")
FLYAI_CLI_PACKAGE = "@fly-ai/flyai-cli"


def _budget_to_max_price(budget_level: Optional[str]) -> Optional[int]:
    """把经济/普通/豪华映射为 search-hotel 的每晚大致上限（人民币），无则不限。"""
    if not budget_level:
        return None
    mapping = {"经济": 450, "普通": 1000, "豪华": 3500}
    return mapping.get(str(budget_level).strip())


def _budget_to_stars(budget_level: Optional[str]) -> Optional[str]:
    """可选：按预算给出星级过滤字符串（逗号分隔）。"""
    if not budget_level:
        return None
    b = str(budget_level).strip()
    if b == "经济":
        return "3,4"
    if b == "豪华":
        return "4,5"
    return None


def _default_check_dates(
    depart_date: Optional[str], days: Optional[Any]
) -> tuple[str, str]:
    """生成入住/离店日期；缺省为明天起住 2 晚。"""
    try:
        nights = max(1, int(days)) if days is not None else 2
    except (TypeError, ValueError):
        nights = 2
    if depart_date and re.match(r"^\d{4}-\d{2}-\d{2}$", str(depart_date).strip()):
        check_in = str(depart_date).strip()
    else:
        check_in = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    d0 = datetime.strptime(check_in, "%Y-%m-%d")
    check_out = (d0 + timedelta(days=min(nights, 14))).strftime("%Y-%m-%d")
    return check_in, check_out


def _extract_json_from_stdout(stdout: str) -> Any:
    """从 CLI 标准输出中解析 JSON（支持单行或首尾空白）。"""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _extract_list(payload: Any) -> List[Dict[str, Any]]:
    """从任意 JSON 结构中尽量取出酒店/POI 对象列表。"""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("hotels", "data", "items", "results", "list", "records", "rows"):
        v = payload.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    inner = payload.get("data")
    if isinstance(inner, dict):
        return _extract_list(inner)
    return []


def _normalize_hotel(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把 FlyAI 返回的单条酒店记录映射为与 Mock 接近的字段。"""
    name = (
        raw.get("hotelName")
        or raw.get("name")
        or raw.get("title")
        or raw.get("hotel_name")
        or "酒店"
    )
    price = raw.get("price") or raw.get("minPrice") or raw.get("lowestPrice") or raw.get("avgPrice")
    if price is not None:
        try:
            price_range = f"¥{int(float(price))}/晚起"
        except (TypeError, ValueError):
            price_range = str(price)
    else:
        price_range = raw.get("priceRange") or raw.get("priceDesc") or "价格见预订页"
    location = (
        raw.get("address")
        or raw.get("location")
        or raw.get("district")
        or raw.get("city")
        or ""
    )
    tags: List[str] = []
    if raw.get("star"):
        tags.append(f"{raw.get('star')}星")
    if raw.get("score") or raw.get("rating"):
        tags.append(f"评分 {raw.get('score') or raw.get('rating')}")
    if raw.get("url") or raw.get("detailUrl") or raw.get("h5Url"):
        tags.append("可预订")
    pnum: Optional[float] = None
    if price is not None:
        try:
            pnum = float(price)
        except (TypeError, ValueError):
            pnum = None
    level = "普通"
    if pnum is not None and pnum < 400:
        level = "经济"
    elif pnum is not None and pnum > 1500:
        level = "豪华"
    rating_val: Optional[float] = None
    for key in ("score", "rating", "avgScore"):
        v = raw.get(key)
        if v is not None:
            try:
                rating_val = float(v)
                break
            except (TypeError, ValueError):
                pass

    out: Dict[str, Any] = {
        "name": str(name)[:200],
        "level": level,
        "price_range": str(price_range)[:80],
        "tags": tags[:6],
        "location": str(location)[:200],
        "rating": rating_val,
        "raw_ref": {k: raw[k] for k in ("url", "detailUrl", "h5Url", "itemId", "hotelId") if k in raw},
    }
    return out


def _normalize_poi(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把 FlyAI POI 记录映射为与 mock 景点接近的结构。"""
    name = raw.get("poiName") or raw.get("name") or raw.get("title") or "景点"
    return {
        "name": str(name)[:120],
        "duration": str(raw.get("suggestDuration") or raw.get("duration") or "2-3 小时")[:40],
        "best_time": str(raw.get("bestTime") or "上午")[:20],
        "tags": raw.get("tags") if isinstance(raw.get("tags"), list) else [str(raw.get("category", "景点"))],
        "description": str(raw.get("description") or raw.get("intro") or "")[:500],
    }


def search_hotels_flyai(
    dest_name: str,
    budget_level: Optional[str] = None,
    depart_date: Optional[str] = None,
    days: Optional[Any] = None,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    """调用 flyai search-hotel，返回 {ok, items, source, message}；不记录 API Key。"""
    t0 = time.perf_counter()
    if not os.getenv("FLYAI_API_KEY"):
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": "未配置 FLYAI_API_KEY。",
        }
    if not (dest_name or "").strip():
        return {"ok": False, "items": [], "source": "flyai", "message": "目的地为空。"}

    check_in, check_out = _default_check_dates(depart_date, days)
    cmd: List[str] = [
        "npx",
        "-y",
        FLYAI_CLI_PACKAGE,
        "search-hotel",
        "--dest-name",
        dest_name.strip(),
        "--check-in-date",
        check_in,
        "--check-out-date",
        check_out,
        "--sort",
        "rate_desc",
    ]
    max_p = _budget_to_max_price(budget_level)
    if max_p is not None:
        cmd.extend(["--max-price", str(max_p)])
    stars = _budget_to_stars(budget_level)
    if stars:
        cmd.extend(["--hotel-stars", stars])

    _LOG.info(
        "flyai_cli search_hotel start dest=%s check_in=%s check_out=%s max_price=%s",
        dest_name.strip(),
        check_in,
        check_out,
        max_p,
    )
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        _LOG.info("flyai_cli search_hotel fail err=npx_not_found elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": "未找到 npx，请先安装 Node.js：https://nodejs.org/",
        }
    except subprocess.TimeoutExpired:
        _LOG.info("flyai_cli search_hotel fail err=timeout elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": f"flyai 查询超时（>{timeout_sec}s）。",
        }
    except Exception as exc:
        _LOG.info(
            "flyai_cli search_hotel fail err=%s elapsed_ms=%.1f",
            type(exc).__name__,
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": f"调用异常：{type(exc).__name__}",
        }

    if proc.returncode != 0:
        err_hint = (proc.stderr or proc.stdout or "")[:500]
        _LOG.info(
            "flyai_cli search_hotel fail returncode=%s stderr_chars=%s elapsed_ms=%.1f",
            proc.returncode,
            len(err_hint),
            (time.perf_counter() - t0) * 1000,
        )
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": f"flyai 退出码 {proc.returncode}。请确认已安装 Node 且可执行 npx @fly-ai/flyai-cli。",
        }

    payload = _extract_json_from_stdout(proc.stdout or "")
    raw_list = _extract_list(payload)
    items = [_normalize_hotel(x) for x in raw_list[:8]]
    items = items[:3]

    _LOG.info(
        "flyai_cli search_hotel done elapsed_ms=%.1f raw_count=%s mapped_count=%s",
        (time.perf_counter() - t0) * 1000,
        len(raw_list),
        len(items),
    )
    if not items:
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": "flyai 返回中未解析到酒店列表（JSON 结构可能与预期不符）。",
        }
    return {"ok": True, "items": items, "source": "flyai", "message": "ok"}


def search_pois_flyai(
    city_name: str,
    timeout_sec: int = 90,
) -> Dict[str, Any]:
    """调用 flyai search-poi，返回 {ok, items, source, message}。"""
    t0 = time.perf_counter()
    if not os.getenv("FLYAI_API_KEY"):
        return {"ok": False, "items": [], "source": "flyai", "message": "未配置 FLYAI_API_KEY。"}
    if not (city_name or "").strip():
        return {"ok": False, "items": [], "source": "flyai", "message": "城市名为空。"}

    cmd = [
        "npx",
        "-y",
        FLYAI_CLI_PACKAGE,
        "search-poi",
        "--city-name",
        city_name.strip(),
    ]
    _LOG.info("flyai_cli search_poi start city=%s", city_name.strip())
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": "未找到 npx，请先安装 Node.js。",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "items": [], "source": "flyai", "message": "flyai POI 查询超时。"}
    except Exception as exc:
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": f"调用异常：{type(exc).__name__}",
        }

    if proc.returncode != 0:
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": f"flyai POI 退出码 {proc.returncode}。",
        }

    payload = _extract_json_from_stdout(proc.stdout or "")
    raw_list = _extract_list(payload)
    items = [_normalize_poi(x) for x in raw_list[:12]]

    _LOG.info(
        "flyai_cli search_poi done elapsed_ms=%.1f raw_count=%s",
        (time.perf_counter() - t0) * 1000,
        len(raw_list),
    )
    if not items:
        return {
            "ok": False,
            "items": [],
            "source": "flyai",
            "message": "未解析到 POI 列表。",
        }
    return {"ok": True, "items": items, "source": "flyai", "message": "ok"}
