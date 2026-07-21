"""Supervisor định tuyến giữa các sub-agent — thay cho `router_agent`/`react_agent` của NAT.

Vì sao KHÔNG dùng `router_agent` (hay `react_agent`) với các branch là `langgraph_wrapper`
------------------------------------------------------------------------------------------
1. **Input schema không tương thích (lỗi validation).** `langgraph_wrapper` có input schema là
   `LanggraphWrapperInput(messages: list[...])`. Khi router/react gọi một branch như một TOOL,
   nó truyền một CHUỖI ("tại sao cước..."). NAT chỉ tự bọc chuỗi khi tool schema có field
   `input_message` hoặc là `ChatRequest` (xem tool_wrapper.py:52-56) — `langgraph_wrapper` có
   `messages` nên chuỗi bị từ chối:
       `2 validation errors for LanggraphWrapperInput / messages / Input should be a valid list`
   Nói cách khác: `langgraph_wrapper` là WORKFLOW-type, không dùng được như TOOL-type.

2. **Router chọn nhầm branch khi bật thinking.** `router_agent.agent_node` chọn branch bằng cách
   dò TÊN branch xuất hiện trong output LLM theo thứ tự (agent.py:135). Qwen bật thinking sinh
   một khối `<think>` liệt kê MỌI branch, nên nó khớp `anthropic_agent` trước dù kết luận là
   `telecom_agent`.

Cách làm ở đây
--------------
Xây một supervisor bằng `create_agent` (LangChain), trong đó MỖI sub-agent được bọc thành một
LangChain tool NHẬN CHUỖI (không qua NAT tool interface). Supervisor chọn tool bằng native
tool-calling (không phải dò chuỗi), và LLM supervisor tắt thinking. Nhờ vậy tránh được cả hai
vấn đề trên. Toàn bộ được bọc `as_nat_graph` để `nat serve` /chat/stream hoạt động.

NAT wire qua `deployments/router_supervisor.py:my_agent` với `workflow._type: langgraph_wrapper`
(KHÔNG cần khối `functions:`).
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from agenticskills.langgraph_agents.agent import build_agent
from agenticskills.langgraph_agents.optimize_agent_v2 import build_langchain_llm
from agenticskills.langgraph_agents.optimize_agent_v3 import build_optimized_agent_v3
from agenticskills.langgraph_agents.stream import StreamSafeGraph, strip_think_full

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore

SUPERVISOR_PROMPT = """\
Bạn là supervisor định tuyến. Chọn ĐÚNG MỘT công cụ phù hợp nhất với yêu cầu của người dùng,
gọi nó, rồi trả lại kết quả của nó cho người dùng.

- Câu hỏi về CƯỚC, gói cước, thuê bao di động, hoặc làm slide Viettel  -> `telecom_agent`
- Yêu cầu xử lý file pdf/xlsx/docx/pptx, thiết kế, web, tài liệu kỹ thuật -> `anthropic_agent`
- Tạo ảnh, chỉnh sửa ảnh, đánh giá biển hiệu cửa hàng                    -> `oneai_agent`

Luôn gọi đúng một công cụ. Nếu không công cụ nào phù hợp, trả lời trực tiếp và ngắn gọn.
"""

_SUBAGENTS = [
    ("telecom_agent",
     "Diễn giải cước, tra cứu thông tin cước/thuê bao di động, và làm slide PowerPoint theo chuẩn Viettel.",
     lambda: build_optimized_agent_v3(skills_subdir="Telecom_Skills", env_var="TELECOM_SKILL_DIR")),
    ("anthropic_agent",
     "Kỹ năng kỹ thuật: xử lý pdf/xlsx/docx/pptx, thiết kế, frontend, tài liệu, web-artifacts, ...",
     lambda: build_agent(skills_subdir="anthropic_skills", env_var="ANTHROPIC_SKILLS_DIR")),
    ("oneai_agent",
     "Tạo ảnh, chỉnh sửa ảnh, đánh giá biển hiệu cửa hàng.",
     lambda: build_agent(skills_subdir="skills_oneai", env_var="ONEAI_SKILL_DIR")),
]


def _as_tool(agent_graph, name: str, description: str) -> StructuredTool:
    """Bọc một sub-agent (LangGraph) thành LangChain tool NHẬN CHUỖI, trả CHUỖI."""

    async def _call(request: str) -> str:
        result = await agent_graph.ainvoke({"messages": [HumanMessage(content=request)]})
        messages = result.get("messages") if isinstance(result, dict) else None
        text = messages[-1].text if messages else str(result)
        return strip_think_full(text)   # sub-agent có thể trả <think>...</think>

    return StructuredTool.from_function(coroutine=_call, name=name, description=description)


def build_supervisor():
    """Supervisor LangChain: LLM (thinking OFF) + mỗi sub-agent là một tool nhận chuỗi."""
    config_llm = SyncBuilder.current().get_llm("llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = build_langchain_llm(config_llm, enable_thinking=False, label="supervisor LLM")

    tools = [_as_tool(build(), name, desc) for name, desc, build in _SUBAGENTS]
    return create_agent(model=llm, tools=tools, system_prompt=SUPERVISOR_PROMPT)


_cached: StreamSafeGraph | None = None


def my_agent(config: Any = None, **_: Any) -> StreamSafeGraph:
    """Factory langgraph_wrapper gọi (trong builder context) — build LAZY, cache 1 lần.

    `stream_nodes=["model"]` -> chỉ stream câu trả lời của supervisor, ẩn hoạt động sub-agent.
    """
    global _cached
    if _cached is None:
        _cached = StreamSafeGraph(build_supervisor(), stream_nodes=["model"])
    return _cached
