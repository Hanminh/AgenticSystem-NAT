"""native_optimize_deep_agent_v2 — như v1, THÊM `TodoListMiddleware` để "phát trực tiếp" cho
người dùng biết agent **đang dùng skill nào** và **đang định làm gì**.

Khác v1 (`native_optimize_deep_agent.py`) ở hai điểm:

1. **Gắn `TodoListMiddleware`** (langchain, không nằm trong stack mặc định 0.7.0) qua tham số
   `middleware=` của `create_deep_agent`. Middleware này cấp tool `write_todos(todos=[{content,
   status}])`; mỗi lần model gọi, graph phát một state update `{"todos":[...]}`. Ta hướng dẫn
   model ghi mỗi todo dạng "Skill <tên>: <việc>" → todo trở thành nhật ký ý định theo thời gian.

2. **Adapter stream `as_nat_graph_todos`** — ngoài việc stream token câu trả lời (như
   `StreamSafeGraph`), còn CHẶN các update `{"todos":[...]}` và phát ra một khối tiến trình
   người-đọc-được ("📋 Kế hoạch: ✅/🔧/⬜ ...") NGAY khi todo đổi. Nhờ vậy người dùng thấy
   agent chọn skill gì và đang làm bước nào TRƯỚC khi có câu trả lời cuối.

Mọi tối ưu của v1 giữ nguyên: LLM thinking OFF + temperature 0, HarnessProfile gọn (tắt
general-purpose subagent, bỏ Summarization). Xem `native_optimize_deep_agent.py` và
`my_instruction/DEEPAGENTS_0_7_NATIVE_OPTIMIZE.md`.

Đánh đổi: `write_todos` tốn thêm token/lượt (model phải lập + cập nhật kế hoạch) → hơi chậm
hơn v1. Đây là cái giá của việc "nhìn thấy agent đang làm gì". Bật/tắt bằng `enable_todos`.
"""

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore

from agenticskills.common import (
    PROJECT_ROOT,
    _resolve_mcp_servers,
    resolve_skills_dir,
)
from agenticskills.deep_agents.native_optimize_deep_agent import _register_lean_profile
from agenticskills.langgraph_agents.optimize_agent_v2 import build_langchain_llm
from agenticskills.langgraph_agents.stream import ThinkStripper, _text_of, strip_think_full
from agenticskills.mcp import load_mcp_tools

load_dotenv()

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt cho write_todos: ép mỗi todo ghi RÕ skill đang dùng + việc đang làm.
# --------------------------------------------------------------------------- #

# TODOS_SYSTEM_PROMPT = """\
# ## write_todos — nhật ký công việc (BẮT BUỘC dùng cho tác vụ skill)

# Bạn có tool `write_todos(todos=[{"content": ..., "status": ...}])` để CÔNG KHAI kế hoạch cho
# người dùng thấy bạn đang làm gì. `status` ∈ {"pending","in_progress","completed"}.

# Quy tắc:
# 1. NGAY khi xác định được skill, gọi `write_todos` ghi kế hoạch. Mỗi `content` phải nêu RÕ
#    **skill đang dùng** và **việc đang làm**, ví dụ:
#      - "Skill dien-giai-cuoc-skill-optimize: đọc SKILL.md"
#      - "Skill dien-giai-cuoc-skill-optimize: chạy API 411,vtfree,consumption,maxconsum"
#      - "Tổng hợp & diễn giải cước cho người dùng"
# 2. Trước khi làm một bước, đặt bước đó `in_progress`; XONG là đổi ngay sang `completed` (gọi
#    lại `write_todos` với danh sách mới). KHÔNG gộp nhiều bước rồi mới cập nhật.
# 3. `write_todos` chỉ để theo dõi — KHÔNG phải câu trả lời. Câu trả lời cuối viết ở message SAU
#    lần `write_todos` cuối cùng.
# 4. Tác vụ 1 bước đơn giản thì có thể bỏ qua todos.

# 5. KHÔNG GHI RÕ ĐƯỜNG DẪN FILE THỰC TẾ. Chỉ ghi tên skill + việc, KHÔNG ghi đường dẫn tuyệt đối.
# """

TODOS_SYSTEM_PROMPT = """\
## write_todos — nhật ký công việc (BẮT BUỘC dùng cho các tác vụ cần tra cứu/xử lý nhiều bước)

Bạn có công cụ `write_todos(todos=[{"content": ..., "status": ...}])` để hiển thị công khai tiến trình xử lý cho người dùng (anh/chị) biết em đang làm gì. `status` ∈ {"pending", "in_progress", "completed"}.

Quy tắc:
1. NỘI DUNG TỔNG QUÁT, DỄ HIỂU: NGAY khi xác định được kế hoạch, hãy gọi `write_todos`. Mỗi `content` chỉ được mô tả khái quát các bước bằng ngôn ngữ đời thường, thân thiện. 
   - TUYỆT ĐỐI KHÔNG ghi các chi tiết kỹ thuật (như tên file SKILL.md, đường dẫn hệ thống, tên API, mã code, hay tên skill nội bộ).
   - Ví dụ NÊN dùng:
     - "Đang tra cứu thông tin cước trên hệ thống cho anh/chị"
     - "Đang kiểm tra dữ liệu sử dụng"
     - "Đang tổng hợp kết quả để phản hồi"
2. CẬP NHẬT TỪNG BƯỚC: Trước khi làm một bước, chuyển trạng thái bước đó thành `in_progress`; XONG là đổi ngay sang `completed` (bằng cách gọi lại `write_todos` với danh sách mới cập nhật). KHÔNG gộp nhiều bước làm xong mới cập nhật một lần.
3. PHÂN BIỆT VỚI CÂU TRẢ LỜI: `write_todos` chỉ dùng để hiển thị trạng thái đang chờ xử lý — KHÔNG phải là câu trả lời giải quyết vấn đề. Câu trả lời cuối cùng giao tiếp với anh/chị sẽ được viết ở message bình thường SAU lệnh `write_todos` cuối cùng.
4. LINH HOẠT: Với các câu hỏi giao tiếp cơ bản hoặc tác vụ 1 bước rất đơn giản, có thể bỏ qua việc gọi todos để phản hồi nhanh hơn.
5. BẢO MẬT: Nhắc lại, KHÔNG ĐƯỢC để lọt bất kỳ thông tin nhạy cảm, log lỗi, hay kiến trúc hệ thống (như tên skill gốc, đường dẫn tuyệt đối) vào nội dung của todos.
"""

# Prompt gốc của agent — kèm chỉ dẫn dùng todos để lộ tiến trình.
NATIVE_OPTIMIZE_V2_SYSTEM_PROMPT = """\
Bạn là Elite Enterprise AI Agent có một thư viện SKILL. Ưu tiên: ĐÚNG, MINH BẠCH tiến trình, NHANH.

Quy trình mỗi yêu cầu:
1. Đối chiếu yêu cầu với danh sách skill (tên + mô tả) ở "Skills System".
2. Khớp skill -> **gọi `write_todos`** ghi kế hoạch (mỗi todo nêu rõ skill + việc), rồi
   **`read_file(path, limit=1000)`** đọc trọn SKILL.md của skill đó. ĐỪNG đoán nội dung skill.
3. Làm ĐÚNG như SKILL.md: lệnh trong đó thường đã có đường dẫn tuyệt đối — copy nguyên, chỉ
   thay tham số, rồi **`execute` NGAY** trong cùng lượt. Cập nhật todos theo tiến trình.
4. Trả lời NGẮN GỌN sau khi có kết quả tool (ở message SAU lần write_todos cuối).

TUYỆT ĐỐI KHÔNG kết thúc lượt bằng "Tôi sẽ gọi...", "Em sẽ chạy..." mà KHÔNG kèm một lời gọi
tool trong CHÍNH lượt đó.
"""

_STATUS_ICON = {"completed": "✅", "in_progress": "🔧", "pending": "⬜"}


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_native_optimize_deep_agent_v2(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    disable_thinking: bool = True,
    temperature: float | None = 0.0,
    lean_profile: bool = True,
    disable_general_purpose: bool = True,
    exclude_summarization: bool = True,
    excluded_tools: tuple[str, ...] = (),
    profile_key: str | None = None,
    enable_todos: bool = True,
    todos_system_prompt: str = TODOS_SYSTEM_PROMPT,
    system_prompt: str | None = None,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    backend=None,
    inherit_env: bool = True,
    backend_env: dict | None = None,
):
    """Như `build_native_optimize_deep_agent` nhưng thêm `TodoListMiddleware` (lộ tiến trình).

    Args (chỉ nêu phần MỚI so với v1):
        enable_todos: `True` (mặc định) -> gắn `TodoListMiddleware` (tool `write_todos`) để agent
            công khai "đang dùng skill nào / định làm gì". `False` -> hành vi giống hệt v1.
        todos_system_prompt: prompt hướng dẫn cách ghi todos (mặc định ép ghi rõ skill + việc).

    Các tham số còn lại xem `native_optimize_deep_agent.build_native_optimize_deep_agent`.

    Returns:
        Graph deepagents đã compile. Bọc bằng `as_nat_graph_todos(...)` (trong module này) để
        stream CẢ token trả lời LẪN tiến trình todos ra cho người dùng.
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    builder = SyncBuilder.current()
    config_llm = builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if disable_thinking:
        llm = build_langchain_llm(config_llm, enable_thinking=False, temperature=temperature,
                                  label="native deep LLM v2")
    else:
        llm = config_llm
        if temperature is not None:
            try:
                llm = config_llm.bind(temperature=temperature)
            except Exception:  # pragma: no cover
                logger.warning("[native_optimize_v2] không bind được temperature vào config LLM.")

    model_name = getattr(config_llm, "model_name", None) or getattr(llm, "model_name", "unknown")

    if lean_profile:
        _register_lean_profile(
            model_name,
            disable_general_purpose=disable_general_purpose,
            exclude_summarization=exclude_summarization,
            excluded_tools=excluded_tools,
            profile_key=profile_key,
        )

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    mcp_tools = load_mcp_tools(servers)

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("Cần 'deepagents' (>=0.7). Cài: pip install deepagents") from exc

    # TodoListMiddleware nằm ở langchain (không thuộc stack mặc định 0.7.0) -> gắn qua middleware=.
    middleware: list[Any] = []
    if enable_todos:
        try:
            from langchain.agents.middleware import TodoListMiddleware
            middleware.append(TodoListMiddleware(system_prompt=todos_system_prompt))
        except ImportError:  # pragma: no cover
            logger.warning("[native_optimize_v2] không import được TodoListMiddleware -> bỏ qua todos.")

    if backend is None:
        backend = LocalShellBackend(
            root_dir=str(PROJECT_ROOT),
            virtual_mode=False,        # đọc được đường dẫn TUYỆT ĐỐI của SKILL.md
            inherit_env=inherit_env,
            env=backend_env,
        )

    logger.info(
        "[native_optimize_v2] skills=%s | thinking=%s | temperature=%s | todos=%s | lean_profile=%s",
        resolved,
        "OFF" if disable_thinking else "theo config",
        temperature,
        enable_todos,
        lean_profile,
    )

    return create_deep_agent(
        model=llm,
        tools=mcp_tools,
        system_prompt=system_prompt or NATIVE_OPTIMIZE_V2_SYSTEM_PROMPT,
        middleware=middleware,         # <- TodoListMiddleware ở đây
        skills=[str(resolved)],
        backend=backend,
        checkpointer=memory,
    )


# --------------------------------------------------------------------------- #
# Streaming adapter — stream token trả lời + LỘ tiến trình todos
# --------------------------------------------------------------------------- #
def _render_todos_block(todos: list[dict]) -> str:
    """Biến danh sách todo thành khối tiến trình người-đọc-được."""
    lines = ["", "📋 Kế hoạch hiện tại:"]
    for t in todos:
        icon = _STATUS_ICON.get(str(t.get("status")), "•")
        lines.append(f"  {icon} {t.get('content', '')}")
    lines.append("")
    return "\n".join(lines)


class TodoStreamGraph:
    """Bọc graph deepagents cho NAT langgraph_wrapper: stream token trả lời + tiến trình todos.

    Giống `StreamSafeGraph` (chỉ phát token của node model, lọc `<think>`), nhưng THÊM: mỗi khi
    state update chứa `todos` thay đổi, phát một khối "📋 Kế hoạch..." để người dùng thấy agent
    đang dùng skill nào / làm bước nào. `ainvoke`/`invoke` ủy quyền nguyên vẹn.
    """

    def __init__(self, graph: Any, *, stream_nodes=("model", ), strip_think: bool = True,
                 show_todos: bool = True):
        self._graph = graph
        self._stream_nodes = set(stream_nodes) if stream_nodes else None
        self._strip_think = strip_think
        self._show_todos = show_todos

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

            # mode == "updates": tìm todos (bất kể tên node) + nhớ assistant cuối (fallback).
            if not isinstance(payload, dict):
                continue
            for node_update in payload.values():
                if not isinstance(node_update, dict):
                    continue
                if self._show_todos and node_update.get("todos") is not None:
                    todos = node_update["todos"]
                    rep = json.dumps(todos, ensure_ascii=False, sort_keys=True)
                    if rep != last_todos_repr:      # chỉ phát khi ĐỔI
                        last_todos_repr = rep
                        streamed_a_token = True
                        yield {"messages": [AIMessageChunk(content=_render_todos_block(todos))]}
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


def as_nat_graph_todos(graph: Any, *, stream_nodes=("model", ), strip_think: bool = True,
                       show_todos: bool = True):
    """Expose graph cho NAT langgraph_wrapper với stream token + tiến trình todos.

    Dùng thay `as_nat_graph` cho agent v2::

        my_agent = as_nat_graph_todos(build_native_optimize_deep_agent_v2(skills_subdir="Telecom_Skills"))
    """

    def _factory(config: Any = None, **_: Any) -> TodoStreamGraph:
        return TodoStreamGraph(graph, stream_nodes=stream_nodes, strip_think=strip_think,
                               show_todos=show_todos)

    return _factory
