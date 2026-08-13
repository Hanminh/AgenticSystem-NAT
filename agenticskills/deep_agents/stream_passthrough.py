# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""stream_passthrough — adapter stream cho NAT `langgraph_wrapper`, mở rộng `todo_event_stream`.

Làm ĐỒNG THỜI ba việc (đối xứng với `todo_event_stream.TodoEventStreamGraph`):

  1. Stream token ĐÁP ÁN của node `model` (lọc `<think>`) -> agent VẪN trả lời bằng chữ như hiện
     tại, không đổi.
  2. Mỗi khi state `todos` đổi -> ĐẨY một NAT intermediate step `write_todos` (để bridge dựng box
     tiến trình). Dùng lại `todo_event_stream._push_todo_step` -> KHÔNG lặp code.
  3. Đọc kênh **custom** của LangGraph (`stream_mode` có "custom"): khi `tool_passthrough` ghi
     `{"script_output": {...}}`, ĐẨY một NAT step tên `package_details` mang JSON -> bridge phát ra
     item type RIÊNG cho client.

Vì sao đẩy step ở WRAPPER chứ không ở tool: `Context.get().intermediate_step_manager` chắc chắn
có mặt ở đây (wrapper chạy trực tiếp trong Context của workflow khi `langgraph_wrapper._astream`
gọi `graph.astream`). Tool chỉ việc ghi kênh custom (không cần biết NAT). Đây đúng cơ chế đã kiểm
chứng của `_push_todo_step`.

`ainvoke`/`invoke` ủy quyền nguyên vẹn (đường `/generate` không stream -> không có step nào).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agenticskills.deep_agents.todo_event_stream import _push_todo_step
from agenticskills.deep_agents.tool_passthrough import SCRIPT_STEP_NAME, _push_script_step
from agenticskills.langgraph_agents.stream import ThinkStripper, _text_of, strip_think_full

logger = logging.getLogger(__name__)


class PassthroughEventStreamGraph:
    """Bọc graph deepagents cho NAT langgraph_wrapper: token đáp án + step write_todos + step
    `package_details` (từ kênh custom của `tool_passthrough`).
    """

    def __init__(self, graph: Any, *, stream_nodes=("model", ), strip_think: bool = True,
                 emit_todos: bool = True):
        self._graph = graph
        self._stream_nodes = set(stream_nodes) if stream_nodes else None
        self._strip_think = strip_think
        self._emit_todos = emit_todos

    @property
    def graph(self) -> Any:
        return self._graph

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return await self._graph.ainvoke(input, config, **kwargs)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._graph.invoke(input, config, **kwargs)

    def _wanted(self, metadata: dict) -> bool:
        if self._stream_nodes is None:
            return True
        return metadata.get("langgraph_node") in self._stream_nodes

    async def astream(self, input: Any, config: Any = None, **kwargs: Any):
        from langchain_core.messages import AIMessage, AIMessageChunk

        kwargs.pop("stream_mode", None)
        streamed_a_token = False
        last_assistant = None
        last_todos_repr = None
        stripper = ThinkStripper() if self._strip_think else None

        async for mode, payload in self._graph.astream(
                input, config, stream_mode=["messages", "updates", "custom"], **kwargs):

            if mode == "messages":
                chunk, metadata = payload
                if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                    continue
                text = _text_of(chunk)
                if not text or not self._wanted(metadata or {}):
                    continue
                if stripper is not None:
                    text = stripper.feed(text)
                    if not text:
                        continue
                    chunk = AIMessageChunk(content=text)
                streamed_a_token = True
                yield {"messages": [chunk]}
                continue

            # mode == "custom": tool_passthrough ghi {"script_output": {...}} -> NAT step riêng.
            if mode == "custom":
                if isinstance(payload, dict) and isinstance(payload.get("script_output"), dict):
                    so = payload["script_output"]
                    name = str(so.get("name") or SCRIPT_STEP_NAME)
                    text = so.get("text")
                    if isinstance(text, str) and text:
                        ok = _push_script_step(name, text)   # -> NAT intermediate step (KHÔNG vào data:)
                        logger.info("[stream_passthrough] nhận custom '%s' (%d ký tự) -> push step=%s",
                                    name, len(text), ok)
                continue

            # mode == "updates": bắt todos đổi -> ĐẨY step write_todos; nhớ assistant cuối (fallback).
            if not isinstance(payload, dict):
                continue
            for node_update in payload.values():
                if not isinstance(node_update, dict):
                    continue
                if self._emit_todos and node_update.get("todos") is not None:
                    todos = node_update["todos"]
                    rep = json.dumps(todos, ensure_ascii=False, sort_keys=True)
                    if rep != last_todos_repr:          # chỉ đẩy khi ĐỔI
                        last_todos_repr = rep
                        _push_todo_step(todos)           # -> NAT intermediate step (không vào data:)
                for message in node_update.get("messages") or []:
                    if isinstance(message, (AIMessage, AIMessageChunk)) and _text_of(message):
                        last_assistant = message

        if stripper is not None:
            tail = stripper.flush()
            if tail:
                streamed_a_token = True
                yield {"messages": [AIMessageChunk(content=tail)]}

        if not streamed_a_token and last_assistant is not None:
            text = _text_of(last_assistant)
            if self._strip_think:
                text = strip_think_full(text)
            yield {"messages": [AIMessageChunk(content=text)]}


def as_nat_graph_passthrough(graph: Any, *, stream_nodes=("model", ), strip_think: bool = True,
                             emit_todos: bool = True):
    """Expose graph cho NAT langgraph_wrapper: token đáp án + step write_todos + step package_details.

    Dùng cặp với `cardbridge`::

        my_agent = as_nat_graph_passthrough(
            build_native_optimize_deep_agent_passthrough(skills_subdir="Telecom_Skills",
                                                         enable_todos=True))
    """

    def _factory(config: Any = None, **_: Any) -> PassthroughEventStreamGraph:
        return PassthroughEventStreamGraph(graph, stream_nodes=stream_nodes, strip_think=strip_think,
                                           emit_todos=emit_todos)

    return _factory
