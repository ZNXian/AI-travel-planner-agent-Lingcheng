# 🧳 灵程 · Lingcheng Agent

> 你的智能旅行规划师 | Your AI Travel Planner

**灵程**是一款基于自然语言对话的智能旅行规划 Agent。告诉它你想去哪、玩几天、预算多少，它会为你生成一份包含交通、酒店、行程的完整攻略。

**Lingcheng** is a conversational AI travel planner. Just tell it where you want to go, how many days, and your budget — it will generate a complete plan covering transportation, accommodation, and daily activities.

## ✨ 亮点

- 🗣️ 多轮对话：像和朋友聊天一样规划旅行
- 🚆 实时价格：高铁班次与票价（12306 MCP）+ 机票比价（百炼搜索）
- 🏨 智能推荐：酒店、景点、餐饮，符合你的节奏和预算
- 🔄 增量调整：说“改成豪华版”或“缩短到3天”，Agent 自动重新规划
- 🧠 思考可见：每一步决策过程都展示给你

## 🛠️ 技术栈

- LangGraph — 状态机与多轮记忆
- 阿里百炼 — LLM 推理 + 联网搜索
- MCP 协议 — 12306 高铁数据
- Gradio — Web 对话界面
