"""Streaming adapter that makes these agents work with NAT's `/chat/stream`.

Why this exists
---------------
NAT's `langgraph_wrapper` streams by calling `graph.astream(...)` with LangGraph's
DEFAULT `stream_mode="updates"`, then validating EVERY chunk against
`LanggraphWrapperOutput`, whose `messages` field is REQUIRED:

    async for output in self._graph.astream(value.model_dump()):
        yield self._parse_stream_output(output)      # -> model_validate(...)

An agent graph emits plenty of state updates that carry NO messages:
  * `SkillMiddleware.before_agent`  -> {"skills_prompt": ..., "skills_metadata": [...]}
  * deepagents planning             -> {"todos": [...]}
  * deepagents filesystem           -> {"files": {...}}
The first such update raises

    RuntimeError: Error in LangGraph workflow: 1 validation error for
    LanggraphWrapperOutput / messages / Field required

That is exactly why `/generate` works (it uses `ainvoke`, which returns the full
final state — always containing `messages`) while `/chat/stream` fails.

What this does
--------------
`StreamSafeGraph` streams with `stream_mode=["messages", "updates"]`:
  * "messages" gives real LLM TOKEN chunks -> streamed one by one. This is the
    right granularity: NAT turns each yielded chunk into a ChatResponseChunk
    *delta*, so emitting whole messages instead would duplicate text.
  * ToolMessages and empty tool-call-only chunks are skipped (the user must not
    see raw bash/python output streamed as assistant text).
  * "updates" is only used as a FALLBACK: if the model never streamed a token
    (e.g. streaming unsupported by the endpoint), the final assistant message is
    emitted as a single chunk so the client still gets an answer.

`ainvoke` is delegated untouched, so `/generate` keeps working exactly as before.
"""

import re
from collections.abc import Collection
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.messages import BaseMessage


def _text_of(message: BaseMessage) -> str:
    """Assistant text of a message ("" for tool-call-only / reasoning-only chunks)."""
    content = message.content
    if isinstance(content, str):
        return content
    # Multimodal / reasoning-block content: keep only the text parts.
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think_full(text: str) -> str:
    """Bỏ mọi khối `<think>...</think>` trong một chuỗi HOÀN CHỈNH (dùng cho fallback)."""
    return _THINK_BLOCK.sub("", text or "")


def _holdback_len(buf: str, marker: str) -> int:
    """Độ dài hậu tố dài nhất của `buf` mà là tiền tố của `marker`.

    Dùng để KHÔNG phát vội phần đuôi buffer có thể là một nửa của `<think>` / `</think>`
    bị cắt ngang giữa hai token chunk.
    """
    for k in range(min(len(buf), len(marker) - 1), 0, -1):
        if buf[-k:] == marker[:k].lower() or buf[-k:].lower() == marker[:k].lower():
            return k
    return 0


class ThinkStripper:
    """Bộ lọc STREAMING loại bỏ khối `<think>...</think>` khỏi token đang chảy.

    Qwen (khi bật thinking) trả reasoning INLINE trong `content` dạng
    `<think> ...suy nghĩ... </think>đáp án`. Vì token được stream từng mảnh, thẻ mở/đóng có
    thể bị cắt ngang giữa hai chunk, nên phải dùng máy trạng thái + giữ đệm (holdback), không
    thể chỉ `str.replace`.

    Chỉ phát ra phần NẰM NGOÀI cặp thẻ. Không mở thẻ nào -> hành vi như cũ (đi thẳng).
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            if not self._in_think:
                idx = self._buf.lower().find(self._OPEN)
                if idx >= 0:                                   # thấy <think> đầy đủ
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self._OPEN):]
                    self._in_think = True
                    continue
                hold = _holdback_len(self._buf, self._OPEN)    # có thể là <think> cắt ngang
                if hold < len(self._buf):
                    out.append(self._buf[:len(self._buf) - hold])
                    self._buf = self._buf[len(self._buf) - hold:]
                break
            else:
                idx = self._buf.lower().find(self._CLOSE)
                if idx >= 0:                                   # thấy </think> -> hết thinking
                    self._buf = self._buf[idx + len(self._CLOSE):]
                    self._in_think = False
                    continue
                self._buf = self._buf[len(self._buf) - _holdback_len(self._buf, self._CLOSE):]
                break
        return "".join(out)

    def flush(self) -> str:
        """Phần còn lại khi stream kết thúc (đệm chưa phát, chỉ khi đang NGOÀI thẻ think)."""
        out = "" if self._in_think else self._buf
        self._buf = ""
        return out


# --------------------------------------------------------------------------- #
# Rút gọn output NON-STREAM về ĐÁP ÁN CUỐI (cho sub-agent làm tool của react)
# --------------------------------------------------------------------------- #
def final_answer_message(messages: list) -> BaseMessage | None:
    """Message đáp án CUỐI: AIMessage có text và KHÔNG phải lượt gọi tool.

    Lý do (đã verify qua code + Phoenix): khi một deep agent là TOOL của agent cha (workflow react),
    tool trả về `LanggraphWrapperOutput` NGUYÊN GỐC = TOÀN BỘ messages (mọi bước trung gian) ->
    LangChain serialize cả cụm vào ToolMessage -> react LLM phải đọc lại toàn bộ trace -> phình ngữ
    cảnh + chậm. Rút output non-stream còn ĐÁP ÁN CUỐI để cha chỉ thấy kết quả, không thấy scratchpad.
    """
    msgs = messages or []
    for m in reversed(msgs):
        if isinstance(m, (AIMessage, AIMessageChunk)) and _text_of(m).strip() \
                and not getattr(m, "tool_calls", None):
            return m
    return msgs[-1] if msgs else None


def reduce_to_final_message(state: Any) -> Any:
    """Thay `messages` của state bằng CHỈ message đáp án cuối (giữ mọi key khác).

    `state` không phải dict / không có `messages` -> trả nguyên. Chỉ tác động đường non-stream
    (`ainvoke`/`invoke`); `/chat/stream` dùng `astream` nên KHÔNG bị ảnh hưởng. `/generate` cũng chỉ
    lấy `messages[-1].text` nên không đổi kết quả.
    """
    if not isinstance(state, dict):
        return state
    msgs = state.get("messages")
    if not msgs:
        return state
    final = final_answer_message(msgs)
    if final is None:
        return state
    return {**state, "messages": [final]}


class StreamSafeGraph:
    """Wraps a compiled LangGraph so NAT's langgraph_wrapper can stream it.

    Only `ainvoke` / `invoke` / `astream` are exposed — that is everything
    `LanggraphWrapperFunction` calls.

    Lọc token (rất quan trọng với agent nhiều lần gọi LLM)
    ------------------------------------------------------
    `stream_mode="messages"` phát ra token của **MỌI** lần gọi LLM bên trong graph —
    kể cả những lần KHÔNG phải câu trả lời cho người dùng:

      * LLM chọn skill chạy trong `before_agent`  -> stream ra `["ten-skill"]` (!),
      * câu nói trung gian trước mỗi tool call    -> stream ra "Tôi sẽ gọi API...",
      * rồi mới tới câu trả lời thật.

    Client sẽ nối tất cả lại thành một mớ hỗn độn. Hai bộ lọc để chặn:

      * `stream_nodes`: chỉ stream token của các node này (VD `("model",)` -> bỏ
        node `before_agent` của middleware).
      * `stream_tags`: chỉ stream token của lần gọi LLM được gắn nhãn (tag). Dùng khi
        cùng một node (`model`) vừa sinh câu trung gian vừa sinh câu trả lời cuối —
        chỉ lần cuối được gắn tag (xem `optimize_agent_v2`).

    Mặc định cả hai là `None` = stream tất cả (giữ nguyên hành vi cũ).

    Lọc `<think>` (mặc định BẬT)
    ---------------------------
    Qwen bật thinking trả reasoning inline: `<think>...</think>đáp án`. `strip_think=True`
    (mặc định) loại bỏ khối `<think>...</think>` khỏi token stream để client chỉ nhận đáp án.
    """

    def __init__(
        self,
        graph: Any,
        *,
        stream_nodes: Collection[str] | None = None,
        stream_tags: Collection[str] | None = None,
        strip_think: bool = True,
    ):
        self._graph = graph
        self._stream_nodes = set(stream_nodes) if stream_nodes else None
        self._stream_tags = set(stream_tags) if stream_tags else None
        self._strip_think = strip_think

    @property
    def graph(self) -> Any:
        """The underlying compiled graph (for tests / introspection)."""
        return self._graph

    # -- non-streaming: rút output còn ĐÁP ÁN CUỐI (để sub-agent-làm-tool không phình ngữ cảnh cha) --

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return reduce_to_final_message(await self._graph.ainvoke(input, config, **kwargs))

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return reduce_to_final_message(self._graph.invoke(input, config, **kwargs))

    # -- streaming: token chunks only, never a message-less state update --------

    def _wanted(self, metadata: dict[str, Any]) -> bool:
        """Token này có thuộc lần gọi LLM mà người dùng nên thấy không?"""
        if self._stream_nodes is not None:
            if metadata.get("langgraph_node") not in self._stream_nodes:
                return False
        if self._stream_tags is not None:
            tags = set(metadata.get("tags") or ())
            if not (tags & self._stream_tags):
                return False
        return True

    async def astream(self, input: Any, config: Any = None, **kwargs: Any):
        kwargs.pop("stream_mode", None)  # we choose the mode; NAT passes none anyway

        streamed_a_token = False
        last_assistant: BaseMessage | None = None
        stripper = ThinkStripper() if self._strip_think else None

        async for mode, payload in self._graph.astream(input,
                                                       config,
                                                       stream_mode=["messages", "updates"],
                                                       **kwargs):
            if mode == "messages":
                chunk, metadata = payload
                # Skip ToolMessages: raw tool output is not assistant text.
                if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                    continue
                text = _text_of(chunk)
                # Skip chunks that carry only tool_calls / reasoning and no text.
                if not text:
                    continue
                # Skip các lần gọi LLM không dành cho người dùng (chọn skill, câu trung gian).
                if not self._wanted(metadata or {}):
                    continue
                if stripper is not None:                       # bỏ <think>...</think> inline
                    text = stripper.feed(text)
                    if not text:                               # đang trong khối think -> chưa phát
                        continue
                    chunk = AIMessageChunk(content=text)
                streamed_a_token = True
                yield {"messages": [chunk]}
                continue

            # mode == "updates": remember the last assistant message, in case the
            # model produced no token chunks at all (non-streaming endpoint).
            if not isinstance(payload, dict):
                continue
            for node_update in payload.values():
                if not isinstance(node_update, dict):
                    continue
                for message in node_update.get("messages") or []:
                    if isinstance(message, (AIMessage, AIMessageChunk)) and _text_of(message):
                        last_assistant = message

        # Xả nốt đệm của bộ lọc think (phần đáp án cuối chưa kịp phát).
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


def as_nat_graph(
    graph: Any,
    *,
    stream_nodes: Collection[str] | None = None,
    stream_tags: Collection[str] | None = None,
    strip_think: bool = True,
):
    """Expose a compiled graph to NAT's `langgraph_wrapper` with safe streaming.

    `langgraph_wrapper` accepts either a `CompiledStateGraph` or a CALLABLE that
    returns one; we use the callable form so we can hand it a `StreamSafeGraph`.

    Usage in a deployment module::

        # stream MỌI câu model nói (kể cả câu trung gian)
        my_agent = as_nat_graph(build_agent(skills_subdir="anthropic_skills"))

        # CHỈ stream câu trả lời cuối (xem optimize_agent_v2.FINAL_ANSWER_TAG)
        my_agent = as_nat_graph(build_optimized_agent_v2(...), stream_tags=["nat_final_answer"])
    """

    def _factory(config: Any = None, **_: Any) -> StreamSafeGraph:
        return StreamSafeGraph(graph, stream_nodes=stream_nodes, stream_tags=stream_tags,
                               strip_think=strip_think)

    return _factory
