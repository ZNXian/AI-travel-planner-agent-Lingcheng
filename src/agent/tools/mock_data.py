"""
文件名: mock_data.py
用途: 提供 5 个热门城市的酒店与景点 Mock 数据（接入飞猪 flyai 前的占位数据），
      数据结构尽量贴近真实业务，方便后续直接替换。
对外暴露:
  - HOTELS: dict[城市 -> 酒店列表]
  - ATTRACTIONS: dict[城市 -> 景点列表]
  - get_hotels(city, budget_level): 按预算等级筛选酒店
  - get_attractions(city): 获取该城市景点
"""

import time
from typing import Any, Dict, List, Optional

from src.lingcheng_logging import get_logger


_LOG = get_logger("tool.flyai")


HOTELS: Dict[str, List[Dict[str, Any]]] = {
    "北京": [
        {
            "name": "如家精选酒店（王府井店）",
            "level": "经济",
            "price_range": "¥260-340/晚",
            "tags": ["地铁口", "免费早餐", "家庭房"],
            "location": "东城区王府井大街",
            "rating": 4.5,
        },
        {
            "name": "全季酒店（前门店）",
            "level": "普通",
            "price_range": "¥520-680/晚",
            "tags": ["近天安门", "商务", "安静"],
            "location": "东城区前门东大街",
            "rating": 4.7,
        },
        {
            "name": "北京瑰丽酒店",
            "level": "豪华",
            "price_range": "¥2380-4200/晚",
            "tags": ["奢华", "城景房", "米其林餐厅"],
            "location": "朝阳区呼家楼",
            "rating": 4.9,
        },
    ],
    "上海": [
        {
            "name": "汉庭优佳酒店（南京路步行街店）",
            "level": "经济",
            "price_range": "¥280-360/晚",
            "tags": ["步行街", "性价比高"],
            "location": "黄浦区南京东路",
            "rating": 4.4,
        },
        {
            "name": "锦江都城（外滩店）",
            "level": "普通",
            "price_range": "¥640-820/晚",
            "tags": ["外滩景观", "老洋房风格"],
            "location": "黄浦区延安东路",
            "rating": 4.6,
        },
        {
            "name": "上海半岛酒店",
            "level": "豪华",
            "price_range": "¥3600-6800/晚",
            "tags": ["外滩江景", "下午茶", "顶级服务"],
            "location": "黄浦区中山东一路 32 号",
            "rating": 4.9,
        },
    ],
    "杭州": [
        {
            "name": "莫泰酒店（西湖文化广场店）",
            "level": "经济",
            "price_range": "¥220-310/晚",
            "tags": ["近地铁", "干净整洁"],
            "location": "下城区中山北路",
            "rating": 4.4,
        },
        {
            "name": "杭州西湖国宾馆",
            "level": "普通",
            "price_range": "¥780-1100/晚",
            "tags": ["西湖边", "园林风", "早餐丰盛"],
            "location": "西湖区杨公堤 18 号",
            "rating": 4.8,
        },
        {
            "name": "杭州西子湖四季酒店",
            "level": "豪华",
            "price_range": "¥3200-5800/晚",
            "tags": ["湖景套房", "私汤", "亲子设施"],
            "location": "西湖区灵隐路 5 号",
            "rating": 4.9,
        },
    ],
    "成都": [
        {
            "name": "7 天连锁酒店（春熙路店）",
            "level": "经济",
            "price_range": "¥200-270/晚",
            "tags": ["春熙路", "美食街区"],
            "location": "锦江区红星路",
            "rating": 4.3,
        },
        {
            "name": "成都博舍酒店",
            "level": "普通",
            "price_range": "¥820-1180/晚",
            "tags": ["太古里旁", "设计感", "茶室"],
            "location": "锦江区笔帖式街",
            "rating": 4.8,
        },
        {
            "name": "成都瑞吉酒店",
            "level": "豪华",
            "price_range": "¥2280-3800/晚",
            "tags": ["管家服务", "顶层泳池", "川菜大厨"],
            "location": "锦江区下东大街 88 号",
            "rating": 4.9,
        },
    ],
    "西安": [
        {
            "name": "锦江之星（钟楼店）",
            "level": "经济",
            "price_range": "¥230-300/晚",
            "tags": ["钟楼旁", "老城核心"],
            "location": "碑林区西大街",
            "rating": 4.4,
        },
        {
            "name": "西安索菲特人民大厦",
            "level": "普通",
            "price_range": "¥760-980/晚",
            "tags": ["民国风建筑", "园林", "自助早餐好评"],
            "location": "新城区东新街 319 号",
            "rating": 4.7,
        },
        {
            "name": "西安君悦酒店",
            "level": "豪华",
            "price_range": "¥1980-3200/晚",
            "tags": ["大唐不夜城", "城景房", "高空酒廊"],
            "location": "雁塔区慈恩西路",
            "rating": 4.8,
        },
    ],
}


ATTRACTIONS: Dict[str, List[Dict[str, Any]]] = {
    "北京": [
        {
            "name": "故宫博物院",
            "duration": "3-4 小时",
            "best_time": "上午",
            "tags": ["历史", "必打卡", "需预约"],
            "description": "明清两代皇家宫殿，建议从午门入、神武门出。",
        },
        {
            "name": "天安门广场 & 国家博物馆",
            "duration": "2-3 小时",
            "best_time": "上午",
            "tags": ["地标", "免费", "需预约"],
            "description": "升旗仪式与国博常设展可深度体验。",
        },
        {
            "name": "八达岭长城",
            "duration": "半天",
            "best_time": "全天",
            "tags": ["户外", "经典"],
            "description": "建议早出发避高峰，特种兵推荐徒步北 8 楼。",
        },
        {
            "name": "颐和园",
            "duration": "3 小时",
            "best_time": "下午",
            "tags": ["园林", "避暑"],
            "description": "昆明湖游船 + 长廊 + 万寿山，节奏舒适。",
        },
        {
            "name": "南锣鼓巷 & 什刹海",
            "duration": "2-3 小时",
            "best_time": "晚上",
            "tags": ["胡同", "夜景", "美食"],
            "description": "胡同漫步 + 银锭桥赏夜景，附近小吃丰富。",
        },
    ],
    "上海": [
        {
            "name": "外滩",
            "duration": "1.5 小时",
            "best_time": "晚上",
            "tags": ["夜景", "免费"],
            "description": "东方明珠对望万国建筑群，建议晚 7:30 前抵达。",
        },
        {
            "name": "豫园 & 城隍庙",
            "duration": "2 小时",
            "best_time": "上午",
            "tags": ["古典园林", "小吃"],
            "description": "南翔小笼 + 园林观赏，节奏悠闲。",
        },
        {
            "name": "迪士尼乐园",
            "duration": "全天",
            "best_time": "全天",
            "tags": ["亲子", "网红"],
            "description": "建议提早购票，推荐项目：创极速光轮、小熊维尼。",
        },
        {
            "name": "武康路 & 安福路",
            "duration": "2 小时",
            "best_time": "下午",
            "tags": ["街拍", "Citywalk"],
            "description": "武康大楼 + 网红咖啡 + 买手店，文艺路线。",
        },
        {
            "name": "上海博物馆（人民广场馆）",
            "duration": "2-3 小时",
            "best_time": "上午",
            "tags": ["文化", "免费", "需预约"],
            "description": "青铜、瓷器馆藏国内顶级，雨天首选。",
        },
    ],
    "杭州": [
        {
            "name": "西湖（断桥-苏堤）",
            "duration": "半天",
            "best_time": "上午",
            "tags": ["必打卡", "免费"],
            "description": "断桥 → 白堤 → 苏堤，可骑行可步行。",
        },
        {
            "name": "灵隐寺 & 飞来峰",
            "duration": "3 小时",
            "best_time": "上午",
            "tags": ["佛教", "登山"],
            "description": "飞来峰摩崖造像很出片，需另购灵隐寺香花券。",
        },
        {
            "name": "南宋御街 & 河坊街",
            "duration": "2 小时",
            "best_time": "晚上",
            "tags": ["夜市", "美食"],
            "description": "定胜糕、片儿川夜宵首选。",
        },
        {
            "name": "西溪国家湿地公园",
            "duration": "半天",
            "best_time": "下午",
            "tags": ["自然", "摇橹船"],
            "description": "适合悠闲节奏，乘船体验最佳。",
        },
        {
            "name": "九溪烟树 & 龙井村",
            "duration": "3-4 小时",
            "best_time": "下午",
            "tags": ["徒步", "茶园"],
            "description": "特种兵线路：九溪十八涧穿越至龙井村。",
        },
    ],
    "成都": [
        {
            "name": "大熊猫繁育研究基地",
            "duration": "半天",
            "best_time": "上午",
            "tags": ["必打卡", "需预约"],
            "description": "建议 7:30 前到，9 点前看活跃熊猫。",
        },
        {
            "name": "宽窄巷子",
            "duration": "2 小时",
            "best_time": "下午",
            "tags": ["古街", "茶馆"],
            "description": "老茶馆 + 川剧变脸短演出。",
        },
        {
            "name": "锦里 & 武侯祠",
            "duration": "3 小时",
            "best_time": "晚上",
            "tags": ["三国文化", "夜市"],
            "description": "晚饭后逛锦里，灯笼夜景出片。",
        },
        {
            "name": "都江堰 & 青城山",
            "duration": "全天",
            "best_time": "全天",
            "tags": ["世界遗产", "登山"],
            "description": "需一整天，建议高铁前往，特种兵可走前后山。",
        },
        {
            "name": "人民公园（鹤鸣茶社）",
            "duration": "1-2 小时",
            "best_time": "下午",
            "tags": ["市井", "采耳"],
            "description": "盖碗茶 + 采耳，体验地道成都慢生活。",
        },
    ],
    "西安": [
        {
            "name": "秦始皇兵马俑",
            "duration": "半天",
            "best_time": "上午",
            "tags": ["世界遗产", "必打卡"],
            "description": "建议先看一号坑，请讲解或租设备。",
        },
        {
            "name": "西安城墙（永宁门）",
            "duration": "2-3 小时",
            "best_time": "傍晚",
            "tags": ["骑行", "夜景"],
            "description": "建议租自行车环城，日落与夜景兼得。",
        },
        {
            "name": "大唐不夜城 & 大雁塔",
            "duration": "2-3 小时",
            "best_time": "晚上",
            "tags": ["夜景", "网红"],
            "description": "不倒翁小姐姐表演 + 喷泉演出。",
        },
        {
            "name": "陕西历史博物馆",
            "duration": "3 小时",
            "best_time": "上午",
            "tags": ["文化", "需预约"],
            "description": "需提前 7 天预约，珍宝馆建议加购。",
        },
        {
            "name": "回民街 & 鼓楼",
            "duration": "2 小时",
            "best_time": "晚上",
            "tags": ["小吃", "市井"],
            "description": "肉夹馍、biangbiang 面、镜糕一条龙。",
        },
    ],
}


def get_hotels(city: str, budget_level: Optional[str] = None) -> List[Dict[str, Any]]:
    """按城市与预算等级返回酒店推荐列表（最多 3 条）；budget_level 为空时按价格阶梯各取 1 条。"""
    t0 = time.perf_counter()
    _LOG.info(
        "flyai_mock get_hotels start city=%s budget_level=%s",
        city,
        budget_level,
    )
    pool = HOTELS.get(city, [])
    if not pool:
        _LOG.info(
            "flyai_mock get_hotels done elapsed_ms=%.1f result_count=0 reason=no_mock_city",
            (time.perf_counter() - t0) * 1000,
        )
        return []
    if not budget_level:
        out = pool[:3]
    else:
        matched = [h for h in pool if h.get("level") == budget_level]
        out = matched[:3] if matched else pool[:3]
    _LOG.info(
        "flyai_mock get_hotels done elapsed_ms=%.1f result_count=%s",
        (time.perf_counter() - t0) * 1000,
        len(out),
    )
    return out


def get_attractions(city: str) -> List[Dict[str, Any]]:
    """返回该城市的景点 Mock 列表，城市不存在时返回空列表。"""
    t0 = time.perf_counter()
    _LOG.info("flyai_mock get_attractions start city=%s", city)
    out = list(ATTRACTIONS.get(city, []))
    _LOG.info(
        "flyai_mock get_attractions done elapsed_ms=%.1f result_count=%s",
        (time.perf_counter() - t0) * 1000,
        len(out),
    )
    return out
