"""FlatTodoMiddleware — bản thay thế `TodoListMiddleware` với schema tool PHẲNG.

VÌ SAO CẦN: `write_todos` gốc nhận args LỒNG `{"todos": [{"content","status"}, ...]}` (mảng object
+ enum). Khi deep agent chạy STREAMING, tool-parser tăng-dần của vLLM/Qwen phải ráp JSON lồng nhiều
tầng qua từng token -> ráp sai/nối thừa -> `json.loads` báo `Extra data: line 1 column 8 (char 7)`.
Các tool phẳng (`execute`, `read_file`) không dính vì args chỉ 1 tầng.

CÁCH SỬA (giữ được todos + streaming, KHÔNG đụng proxy): đổi schema args của tool sang **một chuỗi**
`todos_text: str` — mỗi dòng một bước dạng `status|nội dung`. Đây là schema PHẲNG y hệt `execute`
(`command: str`) nên parser streaming xử lý an toàn. Middleware tự parse chuỗi -> `list[{content,
status}]` rồi phát `Command(update={"todos": [...]})` **GIỐNG HỆT** middleware gốc.

DROP-IN: tool vẫn TÊN `write_todos`, state vẫn KHOÁ `todos` với shape `list[{content,status}]`, nên
`todo_event_stream.as_nat_graph_todo_events` và `todobridge` đọc y như cũ — KHÔNG cần sửa downstream.
Chỉ khác duy nhất là SCHEMA ARGS của tool (chuỗi thay vì mảng-object), tức thứ gây crash.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain.agents.middleware.todo import (  # tái dùng state schema (Todo) + kế thừa middleware
    Todo,
    TodoListMiddleware,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

_VALID_STATUS = ("pending", "in_progress", "completed")


def parse_flat_todos(todos_text: str) -> list[Todo]:
    """Chuỗi 'status|nội dung' (mỗi dòng) -> list[{content, status}]. Lỏng tay: dòng thiếu status
    coi như in_progress; status lạ -> in_progress; bỏ dòng rỗng."""
    todos: list[Todo] = []
    for raw in (todos_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "|" in line:
            status, content = line.split("|", 1)
            status = status.strip().lower()
            content = content.strip()
        else:
            status, content = "in_progress", line
        if status not in _VALID_STATUS:
            status = "in_progress"
        if content:
            todos.append({"content": content, "status": status})  # type: ignore[typeddict-item]
    return todos


def _make_write_todos_tool(description: str):
    """Tool `write_todos(todos_text)` — args PHẲNG + inject `tool_call_id` bằng CHUỖI.

    ⚠️ QUAN TRỌNG: KHÔNG dùng `ToolRuntime` (như `TodoListMiddleware` gốc dựng tool bằng
    `StructuredTool.from_function(func=_write_todos(runtime: ToolRuntime, ...))`). Lý do: NAT
    `LangchainProfilerHandler.on_tool_start` gọi `copy.deepcopy(inputs)` (callback_handler.py:343);
    `inputs` gồm cả arg được inject. `ToolRuntime` chứa coroutine + `_contextvars.Context` ->
    deepcopy fail `cannot pickle 'coroutine'/'_contextvars.Context'` -> hỏng workflow. Dùng
    `InjectedToolCallId` (chỉ là CHUỖI) -> `inputs` picklable -> hết lỗi. Giống bản `@tool
    write_todos` gốc của langchain (`todo.py:139`). State phát ra vẫn `{todos:[...]}` -> downstream
    (todobridge/planbridge/todo_event_stream) không đổi.
    """

    # KHÔNG truyền args_schema: để `@tool` suy schema TỪ CHỮ KÝ hàm — nhờ đó nó nhận ra
    # `InjectedToolCallId` và tự inject `tool_call_id` (nếu truyền args_schema thì langchain bỏ qua
    # phần introspect -> tool_call_id không được inject -> TypeError missing argument). Mô tả của
    # `todos_text` lấy từ docstring Args bên dưới.
    @tool("write_todos", description=description)
    def write_todos(todos_text: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command[Any]:
        """Ghi/cập nhật checklist tiến trình (state `todos`).

        Args:
            todos_text: MỖI DÒNG một bước dạng `status|nội dung`
                (status ∈ pending | in_progress | completed).
        """
        todos = parse_flat_todos(todos_text)
        return Command(update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        })

    return write_todos


FLAT_TODOS_TOOL_DESCRIPTION = (
    "Hiển thị công khai tiến trình xử lý cho người dùng. Truyền MỘT chuỗi `todos_text`, MỖI DÒNG một "
    "bước dạng `status|nội dung` (status ∈ pending|in_progress|completed). Mỗi lần cập nhật, gửi LẠI "
    "TOÀN BỘ danh sách với status mới. KHÔNG gọi song song nhiều lần trong một lượt.")


class FlatTodoMiddleware(TodoListMiddleware):
    """Như `TodoListMiddleware` nhưng tool `write_todos` có schema PHẲNG (`todos_text: str`).

    Kế thừa `wrap_model_call` (chèn system prompt) và `after_model` (chặn gọi song song) của bản gốc;
    CHỈ THAY `self.tools` bằng tool schema phẳng. `system_prompt` mặc định hướng dẫn dạng `status|nội
    dung`. `state_schema = PlanningState` (thừa kế) -> state `todos` tương thích hoàn toàn.
    """

    def __init__(self, *, system_prompt: str | None = None, tool_description: str = FLAT_TODOS_TOOL_DESCRIPTION):
        prompt = system_prompt if system_prompt is not None else _DEFAULT_FLAT_SYSTEM_PROMPT
        # Gọi __init__ gốc (set prompt + tool nested), rồi THAY tool bằng bản phẳng (tên GIỮ
        # `write_todos` -> todobridge/todo_event_stream không đổi; inject tool_call_id kiểu CHUỖI
        # để tránh lỗi deepcopy ToolRuntime của NAT profiler).
        super().__init__(system_prompt=prompt, tool_description=tool_description)
        self.tools = [_make_write_todos_tool(tool_description)]


_DEFAULT_FLAT_SYSTEM_PROMPT = """\
## write_todos — nhật ký tiến trình (schema PHẲNG)

Dùng `write_todos(todos_text="...")` để hiển thị công khai tiến trình cho người dùng (anh/chị).
`todos_text` là MỘT chuỗi, MỖI DÒNG một bước theo dạng:

    status|nội dung

`status` ∈ pending | in_progress | completed. Ví dụ:

    in_progress|Đang tra cứu thông tin trên hệ thống cho anh/chị
    pending|Tổng hợp kết quả để phản hồi

Quy tắc:
1. Nội dung TỔNG QUÁT, thân thiện; KHÔNG ghi chi tiết kỹ thuật (tên file/skill, đường dẫn, API).
2. Trước khi làm một bước, đặt bước đó `in_progress`; XONG đổi ngay sang `completed` bằng cách gọi
   lại `write_todos` với TOÀN BỘ danh sách (status đã cập nhật). KHÔNG gộp nhiều bước.
3. `write_todos` chỉ để theo dõi — KHÔNG phải câu trả lời. Đáp án cuối viết ở message SAU lần gọi
   `write_todos` cuối cùng.
4. KHÔNG gọi `write_todos` song song nhiều lần trong một lượt.
"""
