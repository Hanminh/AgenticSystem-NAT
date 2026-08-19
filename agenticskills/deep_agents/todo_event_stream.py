"""Emitter phía WORKFLOW: biến mỗi lần `write_todos` (TodoListMiddleware) thành một NAT
INTERMEDIATE STEP (tool step) — để bridge hiển thị "dựa trên lời gọi tool", đúng cách thinkbridge.

Vì sao (đã kiểm chứng trong mã nguồn)
------------------------------------
`langgraph_wrapper._astream` gọi `graph.astream(...)` KHÔNG truyền callbacks, và NAT không
`register_configure_hook`. Nên tool `write_todos` chạy BÊN TRONG deep agent **không** tự sinh
NAT intermediate step -> thinkbridge/todobridge không thấy gì để hiển thị.

Cách xử lý: wrapper này chạy TRONG Context của workflow (khi `graph.astream` được gọi từ
langgraph_wrapper). Mỗi khi state `todos` đổi (bắt qua `stream_mode="updates"`), ta ĐẨY một cặp
`TOOL_START` + `TOOL_END` vào `Context.get().intermediate_step_manager` với:
    name = "write_todos"
    data.input  = JSON {"todos":[...]}   (bản snapshot của lần write_todos đó)
    data.output = JSON {"todos":[...]}
Front-end FastAPI (`generate_streaming_response` -> `pull_intermediate` -> `StepAdaptor`) sẽ phát
ra `intermediate_data:` với name = "Tool: write_todos" và payload chứa JSON todos trong
`**Input:**`. todobridge đọc step này, parse todos, dựng box (giống thinkbridge đọc box tool).

Token ĐÁP ÁN vẫn stream bình thường qua `data:` (node `model`, đã lọc `<think>`). Todos KHÔNG
còn chảy vào `data:` -> luồng đáp án sạch.

`StepAdaptor._handle_tool` YÊU CẦU có TOOL_START trùng UUID mới sinh step (nếu không, trả None),
nên bắt buộc đẩy CẢ START lẫn END.
"""

import json
import logging
import uuid
from typing import Any

from agenticskills.langgraph_agents.stream import (
    ThinkStripper,
    _text_of,
    reduce_to_final_message,
    strip_think_full,
)

logger = logging.getLogger(__name__)

# Tên tool step cho write_todos (front-end sẽ hiển thị thành "Tool: write_todos").
TODO_TOOL_NAME = "write_todos"


def _push_todo_step(todos: list[dict]) -> bool:
    """Đẩy TOOL_START + TOOL_END cho một snapshot todos vào Context hiện tại.

    Trả True nếu đẩy được (đang trong Context của workflow), False nếu không (vd chạy ngoài
    NAT / /generate không stream) -> caller bỏ qua êm.
    """
    try:
        from nat.builder.context import Context  # import lazy: chỉ có ý nghĩa khi chạy trong NAT
        from nat.data_models.intermediate_step import (
            IntermediateStepPayload,
            IntermediateStepType,
            StreamEventData,
        )
    except Exception:  # pragma: no cover
        return False

    try:
        mgr = Context.get().intermediate_step_manager
    except Exception:  # pragma: no cover - không có Context
        return False

    payload = json.dumps({"todos": todos}, ensure_ascii=False)
    uid = f"todo-{uuid.uuid4().hex[:12]}"
    try:
        mgr.push_intermediate_step(IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_START, name=TODO_TOOL_NAME, UUID=uid,
            data=StreamEventData(input=payload)))
        mgr.push_intermediate_step(IntermediateStepPayload(
            event_type=IntermediateStepType.TOOL_END, name=TODO_TOOL_NAME, UUID=uid,
            data=StreamEventData(input=payload, output=payload)))
        return True
    except Exception:  # pragma: no cover
        logger.exception("[todo_event_stream] không đẩy được intermediate step write_todos")
        return False


class TodoEventStreamGraph:
    """Bọc graph deepagents (có TodoListMiddleware) cho NAT langgraph_wrapper.

    astream:
      * stream token ĐÁP ÁN của node `model` (lọc `<think>`), và
      * mỗi khi `todos` ĐỔI -> ĐẨY một NAT tool step `write_todos` (không chảy vào `data:`).
    `ainvoke`/`invoke` ủy quyền nguyên vẹn (đường /generate không có todos step vì không stream).
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
        # Rút còn ĐÁP ÁN CUỐI: khi deep agent là tool của agent cha (react), tránh nhét cả trace vào cha.
        return reduce_to_final_message(await self._graph.ainvoke(input, config, **kwargs))

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return reduce_to_final_message(self._graph.invoke(input, config, **kwargs))

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
                input, config, stream_mode=["messages", "updates"], **kwargs):

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

            # mode == "updates": bắt todos đổi -> ĐẨY tool step; nhớ assistant cuối (fallback).
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


def as_nat_graph_todo_events(graph: Any, *, stream_nodes=("model", ), strip_think: bool = True,
                             emit_todos: bool = True):
    """Expose graph cho NAT langgraph_wrapper: stream đáp án + đẩy write_todos thành NAT tool step.

    Dùng cặp với todobridge::

        my_agent = as_nat_graph_todo_events(
            build_native_optimize_deep_agent_v2(skills_subdir="Telecom_Skills"))
    """

    def _factory(config: Any = None, **_: Any) -> TodoEventStreamGraph:
        return TodoEventStreamGraph(graph, stream_nodes=stream_nodes, strip_think=strip_think,
                                    emit_todos=emit_todos)

    return _factory
