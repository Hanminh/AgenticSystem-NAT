# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""native_optimize_deep_agent_passthrough — như v3 nhưng THÊM tool `send_package_details`.

Vì sao là file MỚI (không sửa v3): `build_native_optimize_deep_agent_v3` truyền cứng
`tools=mcp_tools` cho `create_deep_agent` -> không có hook chèn thêm tool cục bộ. File này tái
dùng TOÀN BỘ tối ưu của v3 (llm_from_config, HarnessProfile gọn, FlatTodoMiddleware, prompt) và chỉ
bổ sung một tool passthrough:

    tools = [*mcp_tools, send_package_details]

`send_package_details` (xem `tool_passthrough.py`) chạy `package_details.py` rồi gửi list JSON
THẲNG tới client qua kênh custom (được `stream_passthrough` biến thành NAT step `package_details`).
Ngữ cảnh agent chỉ nhận ACK ngắn -> agent vẫn trả lời bằng chữ như hiện tại.

Bọc graph trả về bằng `stream_passthrough.as_nat_graph_passthrough(...)` và phục vụ cùng
`cardbridge`.
"""

from __future__ import annotations

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
from agenticskills.deep_agents.native_optimize_deep_agent import (
    NATIVE_OPTIMIZE_SYSTEM_PROMPT,  # prompt v1 (không nhắc write_todos) — dùng khi enable_todos=False
    _register_lean_profile,
    llm_from_config,
    resolve_skill_placeholders,
)
from agenticskills.deep_agents.native_optimize_deep_agent_v3 import (
    NATIVE_OPTIMIZE_V3_SYSTEM_PROMPT,  # prompt v3 (có write_todos)
    TODOS_SYSTEM_PROMPT,
)
from agenticskills.mcp import load_mcp_tools

load_dotenv()

logger = logging.getLogger(__name__)

# Tên tool (instance trong YAML `functions:`) mà deep agent lấy qua builder.get_tool.
# Đây là NAT function `telecom_package_details` (package telecom_functions) — thay cho script cũ.
DEFAULT_PACKAGE_TOOL_NAME = "send_package_details"

# Chỉ dẫn thêm cho model: khi nào dùng tool passthrough (nối vào system prompt).
_PASSTHROUGH_PROMPT_SUFFIX = """\

## send_package_details — gửi thẻ gói tới màn hình người dùng
Bạn có tool `send_package_details(package_codes, isdn)` để đẩy CHI TIẾT gói cước xuống màn hình
người dùng (client tự hiển thị dạng thẻ). Quy tắc:
  * CHỈ gọi trong HAI trường hợp: (1) ngay sau khi bạn ĐỀ XUẤT gói cho người dùng -> truyền các mã
    gói vừa đề xuất; (2) người dùng yêu cầu xem MỘT gói cụ thể -> truyền mã gói đó.
  * TUYỆT ĐỐI KHÔNG gọi ở các trường hợp khác (chào hỏi, hỏi số dư, "tôi đang dùng gói nào"...).
  * Sau khi gọi tool này, cứ viết câu tư vấn bằng lời như bình thường — KHÔNG dán lại JSON, KHÔNG
    liệt kê lại toàn bộ chi tiết (client đã hiển thị riêng).
"""


def build_native_optimize_deep_agent_passthrough(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    lean_profile: bool = True,
    disable_general_purpose: bool = True,
    exclude_summarization: bool = True,
    excluded_tools: tuple[str, ...] = ("edit_file", ),
    profile_key: str | None = None,
    enable_todos: bool = True,
    todos_flat_schema: bool = True,
    todos_system_prompt: str = TODOS_SYSTEM_PROMPT,
    system_prompt: str | None = None,
    package_tool_name: str = DEFAULT_PACKAGE_TOOL_NAME,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    backend=None,
    inherit_env: bool = True,
    backend_env: dict | None = None,
):
    """Như `build_native_optimize_deep_agent_v3` nhưng thêm tool `send_package_details`.

    Tool `send_package_details` giờ là một **NAT function** (`_type: telecom_package_details`,
    package `telecom_functions`) — KHÔNG còn dùng script `package_details.py`. Nó fetch chi tiết
    gói, đẩy step `package_details_payload` (JSON) tới client, và trả ack ngắn (ngữ cảnh sạch).

    Args (chỉ nêu phần MỚI so với v3):
        package_tool_name: tên instance function trong YAML `functions:` để lấy làm tool
            (mặc định "send_package_details"). PHẢI khai báo trong config:
                functions: { send_package_details: { _type: telecom_package_details } }

    Các tham số còn lại xem `native_optimize_deep_agent_v3.build_native_optimize_deep_agent_v3`.

    Returns:
        Graph deepagents đã compile. Bọc `stream_passthrough.as_nat_graph_passthrough(...)`.
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)
    _n = resolve_skill_placeholders(resolved)
    if _n:
        logger.info("[passthrough] đã resolve '<skill_dir>' -> path tuyệt đối trong %d SKILL.md.", _n)

    builder = SyncBuilder.current()
    llm = llm_from_config(builder, llm_name)
    model_name = getattr(llm, "model_name", None) or "unknown"

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

    # Tool passthrough giờ là NAT function -> lấy làm tool LangChain qua builder.
    # Yêu cầu YAML: functions: { <package_tool_name>: { _type: telecom_package_details } }.
    tools: list[Any] = [*mcp_tools]
    try:
        send_package_details = builder.get_tool(package_tool_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        tools.append(send_package_details)
        logger.info("[passthrough] gắn NAT function '%s' làm tool send_package_details.", package_tool_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[passthrough] KHÔNG lấy được tool '%s' (%s). Khai báo trong YAML: "
                       "functions: { %s: { _type: telecom_package_details } } và cài package "
                       "telecom_functions. Agent vẫn chạy nhưng KHÔNG gửi được thẻ gói tới client.",
                       package_tool_name, exc, package_tool_name)

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("Cần 'deepagents' (>=0.7). Cài: pip install deepagents") from exc

    middleware: list[Any] = []
    if enable_todos:
        if todos_flat_schema:
            from agenticskills.deep_agents.flat_todo_middleware import FlatTodoMiddleware
            middleware.append(FlatTodoMiddleware())
            logger.info("[passthrough] enable_todos=True (schema PHẲNG) -> write_todos(todos_text) né 'Extra data'.")
        else:
            try:
                from langchain.agents.middleware import TodoListMiddleware
                middleware.append(TodoListMiddleware(system_prompt=todos_system_prompt))
                logger.warning("[passthrough] enable_todos=True + todos_flat_schema=False -> write_todos SCHEMA LỒNG "
                               "(có thể 'Extra data' khi streaming trên proxy parser kén).")
            except ImportError:  # pragma: no cover
                logger.warning("[passthrough] không import được TodoListMiddleware -> bỏ qua todos.")

    if backend is None:
        backend = LocalShellBackend(
            root_dir=str(PROJECT_ROOT),
            virtual_mode=False,
            inherit_env=inherit_env,
            env=backend_env,
        )

    # Prompt gốc khớp bộ tool (giống v3), + phần chỉ dẫn send_package_details.
    base_prompt = system_prompt or (NATIVE_OPTIMIZE_V3_SYSTEM_PROMPT if enable_todos
                                    else NATIVE_OPTIMIZE_SYSTEM_PROMPT)
    final_prompt = base_prompt + _PASSTHROUGH_PROMPT_SUFFIX

    logger.info("[passthrough] skills=%s | llm=%s | todos=%s | lean_profile=%s | +send_package_details",
                resolved, model_name, enable_todos, lean_profile)

    return create_deep_agent(
        model=llm,
        tools=tools,                       # <- mcp_tools + send_package_details
        system_prompt=final_prompt,
        middleware=middleware,
        skills=[str(resolved)],
        backend=backend,
        checkpointer=memory,
    )
