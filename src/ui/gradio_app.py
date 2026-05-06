"""
文件名: gradio_app.py
用途: Gradio 聊天界面。维护一份会话级 AgentState，每次用户发言时通过 LangGraph 推理，
      最后展示带"思考过程"折叠块的 AI 回复。
对外暴露:
  - build_demo: 构造 gr.Blocks 应用
  - launch: 启动 Gradio 服务（供 run.py 调用）
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import create_agent_graph
from src.agent.state import make_initial_state
from src.lingcheng_logging import get_logger, setup_lingcheng_logging


_LOG = get_logger("ui.gradio")

# 对话区字体与行高（作用于 Chatbot 容器及 Markdown 正文）
_UI_CSS = """
.lingcheng-chatbot { font-size: 1.125rem !important; line-height: 1.65 !important; }
.lingcheng-chatbot .prose, .lingcheng-chatbot .md { font-size: 1.125rem !important; }
.lingcheng-chatbot p, .lingcheng-chatbot li { font-size: 1.125rem !important; }
.lingcheng-input textarea { font-size: 1.0625rem !important; line-height: 1.55 !important; }
"""

_TITLE = "灵程 Lingcheng · 智能旅行规划助手"
_SUBTITLE = (
    "多轮对话收集偏好 → 推荐目的地 → 查询交通 / 酒店 → 生成每日行程。"
    "随时说 *改成豪华版* / *缩短到 3 天* / *换成高铁* 让我重新规划。"
)
_GREETING = (
    "你好呀，我是灵程~\n\n"
    "告诉我你想去哪里玩、玩几天、预算多少、偏爱什么节奏，"
    "我可以为你定制完整的旅行攻略。也可以直接说一句 *帮我规划一次 3 天杭州悠闲游* 试试看。"
)


def _ensure_state(session: Dict[str, Any]) -> Dict[str, Any]:
    """确保会话字典里有 agent_state，缺则初始化。"""
    if not session or "agent_state" not in session:
        session = {"agent_state": make_initial_state()}
    return session


def _agent_state_to_chat_history(agent_state: Dict[str, Any]) -> List[Dict[str, str]]:
    """把 AgentState.messages 转成 Gradio Chatbot(messages 格式) 的对话历史。"""
    history: List[Dict[str, str]] = []
    for msg in agent_state.get("messages") or []:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    return history


def _run_turn(
    user_text: str,
    session: Dict[str, Any],
    graph,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """执行一轮对话：把用户输入塞进 state，调用 graph，返回新历史与新会话。"""
    session = _ensure_state(session)
    agent_state: Dict[str, Any] = session["agent_state"]

    messages = list(agent_state.get("messages") or [])
    text = (user_text or "").strip()
    messages.append(HumanMessage(content=text))
    agent_state["messages"] = messages
    agent_state["thinking_steps"] = []

    t_turn = time.perf_counter()
    _LOG.info(
        "conversation_turn start user_chars=%s messages_before=%s",
        len(text),
        len(messages),
    )
    try:
        new_state = graph.invoke(agent_state)
        msgs_after = new_state.get("messages") or []
        last = msgs_after[-1] if msgs_after else None
        last_len = (
            len(last.content)
            if last is not None and hasattr(last, "content") and isinstance(last.content, str)
            else 0
        )
        _LOG.info(
            "conversation_turn done elapsed_ms=%.1f messages_after=%s last_msg_type=%s last_content_chars=%s",
            (time.perf_counter() - t_turn) * 1000,
            len(msgs_after),
            type(last).__name__ if last is not None else "none",
            last_len,
        )
    except Exception as exc:
        _LOG.info(
            "conversation_turn error elapsed_ms=%.1f err=%s",
            (time.perf_counter() - t_turn) * 1000,
            type(exc).__name__,
        )
        new_state = dict(agent_state)
        new_state["messages"] = list(messages) + [
            AIMessage(
                content=(
                    "[Agent 异常] 调用过程中出现问题：" + type(exc).__name__ + "。\n"
                    "请检查 .env 中的 DASHSCOPE_API_KEY 是否填写正确，或稍后重试。"
                )
            )
        ]

    session = {"agent_state": new_state}
    history = _agent_state_to_chat_history(new_state)
    return history, session


def _on_reset():
    """重置会话，回到欢迎状态。"""
    _LOG.info("conversation_reset")
    initial_history = [{"role": "assistant", "content": _GREETING}]
    initial_state = make_initial_state()
    initial_state["messages"] = [AIMessage(content=_GREETING)]
    return initial_history, {"agent_state": initial_state}, ""


def build_demo() -> gr.Blocks:
    """构造 Gradio 应用，包含聊天框、发送/重置按钮与会话状态。"""
    compiled_graph = create_agent_graph()

    with gr.Blocks(title=_TITLE, css=_UI_CSS) as demo:
        gr.Markdown(f"# {_TITLE}\n\n{_SUBTITLE}")

        chatbot = gr.Chatbot(
            label="灵程",
            height=820,
            buttons=["copy"],
            value=[{"role": "assistant", "content": _GREETING}],
            elem_classes=["lingcheng-chatbot"],
        )
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="例如：我想 5 月底从上海出发去成都玩 4 天，预算普通，节奏悠闲...",
                show_label=False,
                lines=3,
                scale=7,
                elem_classes=["lingcheng-input"],
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
            reset_btn = gr.Button("重置", scale=1)

        initial_state = make_initial_state()
        initial_state["messages"] = [AIMessage(content=_GREETING)]
        session_state = gr.State({"agent_state": initial_state})

        def _submit(user_text: str, session: Dict[str, Any]):
            """Gradio 提交回调：闭包持有 compiled_graph，返回新的 chatbot/session/输入框值。"""
            text_ok = user_text and user_text.strip()
            if not text_ok:
                history = _agent_state_to_chat_history(
                    (session or {}).get("agent_state") or {}
                )
                return history, session or {}, ""
            history, new_session = _run_turn(
                (user_text or "").strip(),
                session or {},
                compiled_graph,
            )
            return history, new_session, ""

        send_btn.click(
            _submit,
            inputs=[user_input, session_state],
            outputs=[chatbot, session_state, user_input],
        )
        user_input.submit(
            _submit,
            inputs=[user_input, session_state],
            outputs=[chatbot, session_state, user_input],
        )
        reset_btn.click(
            _on_reset,
            inputs=[],
            outputs=[chatbot, session_state, user_input],
        )

    return demo


def launch() -> None:
    """启动 Gradio 服务，host/port 从环境变量读取。"""
    _console = os.getenv("LINGCHENG_LOG_CONSOLE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    setup_lingcheng_logging(
        Path(__file__).resolve().parents[2],
        console=_console,
    )

    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("GRADIO_PORT", "7860"))
    except ValueError:
        port = 7860

    demo = build_demo()
    _LOG.info("gradio_launch server_name=%s server_port=%s", host, port)
    demo.queue().launch(
        server_name=host,
        server_port=port,
        theme=gr.themes.Soft(),
    )
