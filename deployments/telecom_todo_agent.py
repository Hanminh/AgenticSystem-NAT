"""Deployment cho todobridge — deep agent v2 (TodoListMiddleware) + emitter todos sentinel.

BẮT BUỘC dùng CẶP này để todobridge thấy box:
  1. `build_native_optimize_deep_agent_v2(...)` -> deep agent CÓ `write_todos` (TodoListMiddleware).
  2. `as_nat_graph_todo_events(...)` -> đẩy mỗi lần `todos` đổi ra chunk `data:` có SENTINEL
     (\\x1fTODO\\x1f{json}) để todobridge dựng box.

Nếu bọc bằng `as_nat_graph` thường (không phải `as_nat_graph_todo_events`) thì KHÔNG có sentinel
-> todobridge KHÔNG hiện box nào. Đây là lỗi wiring hay gặp.

NAT wire qua `deployments/telecom_todo_agent.py:my_agent`.
"""

from agenticskills.deep_agents.native_optimize_deep_agent_v2 import build_native_optimize_deep_agent_v2
from agenticskills.deep_agents.todo_event_stream import as_nat_graph_todo_events

my_agent = as_nat_graph_todo_events(
    build_native_optimize_deep_agent_v2(
        skills_subdir="Telecom_Skills",
        env_var="TELECOM_SKILL_DIR",
        llm_name="llm",
        # enable_todos=True (mặc định) — model được cấp tool write_todos + prompt ép lập kế hoạch.
    ),
    stream_nodes=("model", ),   # chỉ stream token node model; todos đi kênh riêng (updates)
)
