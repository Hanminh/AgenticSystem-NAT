"""Skills agent giống optimize_agent_v3 nhưng có thêm bước PLANNER (lập kế hoạch).

Luồng
-----
    PHA 1   — CHỌN SKILL       (1 lần gọi LLM LangChain, thinking TẮT)
       ↓  (code, KHÔNG gọi LLM)
    PHA 1.5 — CAT SKILL.md     ← đọc thẳng bằng Python, nạp vào ngữ cảnh
       ↓
    PHA 1.75 — PLANNER         ← MỚI: 1 lần gọi LLM lập KẾ HOẠCH từ (câu hỏi + SKILL.md)
       ↓
    ReAct với NGỮ CẢNH ĐẦY ĐỦ (như v3): system = persona + SKILL.md + KẾ HOẠCH;
       messages = "## Các bước ĐÃ THỰC HIỆN" (output mọi lần gọi LLM + tool trước đó)

So với v3, khác ĐÚNG một điểm: sau khi đọc xong SKILL.md, agent gọi LLM một lần để **lập kế
hoạch** (liệt kê các bước sẽ làm, gọi tool nào, tham số gì) TRƯỚC khi bắt đầu thực thi. Kế hoạch
được nhét vào system prompt và luôn hiện diện ở mọi vòng ReAct, giúp agent bám sát mục tiêu —
đặc biệt hữu ích với skill nhiều bước / nhiều tool.

Planner là bước "nghĩ trước khi làm": nó KHÔNG gọi tool, chỉ sinh kế hoạch. Vẫn dùng LLM dựng
bằng LangChain với thinking TẮT (như selector và main LLM — xem `optimize_agent_v2`).
"""

import logging
from pathlib import Path
from typing import Any, NotRequired

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore

from agenticskills.common import _resolve_mcp_servers, resolve_skills_dir
from agenticskills.mcp import load_mcp_tools
from agenticskills.langgraph_agents.optimize_agent import _text_of, _user_request
from agenticskills.langgraph_agents.optimize_agent_v2 import (
    SELECTOR_MAX_TOKENS,
    build_langchain_llm,
    build_selector_llm,
    strip_think,
)
from agenticskills.langgraph_agents.optimize_agent_v3 import (
    EXEC_PROMPT_V3,
    NO_SKILL_PROMPT_V3,
    FullContextSkillMiddleware,
    FullContextState,
)
from agenticskills.tools import TOOLS

logger = logging.getLogger(__name__)

PLANNER_MAX_TOKENS = 1024   # kế hoạch ngắn gọn, không cần dài


# --------------------------------------------------------------------------- #
# Prompt của PLANNER và của vòng thực thi (v3 + phần kế hoạch)
# --------------------------------------------------------------------------- #
PLANNER_PROMPT = """\
Bạn là planner. Dựa trên YÊU CẦU của người dùng và TÀI LIỆU SKILL bên dưới, hãy lập một KẾ HOẠCH
NGẮN GỌN các bước sẽ thực hiện để hoàn thành yêu cầu.

Quy tắc:
- Liệt kê các bước theo thứ tự (1, 2, 3, ...).
- Mỗi bước nêu rõ: gọi tool nào và tham số quan trọng (VD: lệnh `bash`, chọn `--apis` nào, số tháng).
- Bám sát đúng cách tài liệu skill hướng dẫn — lệnh trong tài liệu đã có đường dẫn tuyệt đối.
- CHỈ lập kế hoạch, TUYỆT ĐỐI KHÔNG thực thi, không gọi tool ở bước này.
- Ngắn gọn, không giải thích dài dòng.

## Tài liệu skill (đã đọc sẵn)
{docs}
"""

# Nối phần KẾ HOẠCH vào ngay sau prompt thực thi của v3.
EXEC_PROMPT_V4 = EXEC_PROMPT_V3 + """
---

## KẾ HOẠCH đã lập (bám theo, điều chỉnh nếu kết quả thực tế khác)
{plan}
"""


def build_exec_prompt_v4(docs: str, plan: str) -> str:
    if not docs:
        return NO_SKILL_PROMPT_V3
    if not plan:
        return EXEC_PROMPT_V3.format(docs=docs)   # planner lỗi/tắt -> quay về prompt v3
    return EXEC_PROMPT_V4.format(docs=docs, plan=plan)


def planner_messages(query: str, docs: str) -> list[BaseMessage]:
    return [SystemMessage(content=PLANNER_PROMPT.format(docs=docs)), HumanMessage(content=query)]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class PlanState(FullContextState):
    """State v3 + kế hoạch do planner sinh ra."""

    plan: NotRequired[str]


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class PlannerSkillMiddleware(FullContextSkillMiddleware):
    """v3 + bước PLANNER. `_finish_selection` giữ nguyên (chọn skill + nạp SKILL.md); planner chạy
    SAU đó trong `before_agent` / `abefore_agent` (cần một lần gọi LLM riêng)."""

    state_schema = PlanState  # type: ignore

    def __init__(
        self,
        skills_dir: Path,
        selector_llm: BaseChatModel,
        planner_llm: BaseChatModel | None,
        *,
        compact_context: bool = True,
    ):
        super().__init__(skills_dir, selector_llm, compact_context=compact_context)
        self.planner_llm = planner_llm

    def _with_plan(self, update: dict[str, Any], plan: str) -> dict[str, Any]:
        """Gắn kế hoạch vào state + dựng lại exec_prompt (persona + SKILL.md + KẾ HOẠCH)."""
        docs = update.get("skill_docs") or ""
        logger.info("[v4] PHA 1.75 planner: lập kế hoạch %d ký tự", len(plan))
        return {**update, "plan": plan, "exec_prompt": build_exec_prompt_v4(docs, plan)}

    def _should_plan(self, update: dict[str, Any]) -> bool:
        return bool(update.get("skill_docs")) and self.planner_llm is not None

    def before_agent(self, state: PlanState, runtime) -> dict[str, Any]:  # type: ignore
        update = super().before_agent(state, runtime)   # PHA 1 + 1.5
        if not self._should_plan(update):
            return update
        query = _user_request(list(state["messages"]))
        response = self.planner_llm.invoke(planner_messages(query, update["skill_docs"]))
        return self._with_plan(update, strip_think(_text_of(response)))

    async def abefore_agent(self, state: PlanState, runtime) -> dict[str, Any]:  # type: ignore
        update = await super().abefore_agent(state, runtime)
        if not self._should_plan(update):
            return update
        query = _user_request(list(state["messages"]))
        response = await self.planner_llm.ainvoke(planner_messages(query, update["skill_docs"]))
        return self._with_plan(update, strip_think(_text_of(response)))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_optimized_agent_v4(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    selector_llm: BaseChatModel | None = None,
    selector_llm_name: str | None = None,
    selector_use_langchain: bool = True,
    selector_enable_thinking: bool = False,
    selector_max_tokens: int = SELECTOR_MAX_TOKENS,
    planner_llm: BaseChatModel | None = None,
    planner_llm_name: str | None = None,
    planner_use_langchain: bool = True,
    planner_enable_thinking: bool = False,
    planner_max_tokens: int = PLANNER_MAX_TOKENS,
    enable_planner: bool = True,
    main_llm_disable_thinking: bool = True,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    compact_context: bool = True,
):
    """Agent giống v3 + bước PLANNER lập kế hoạch sau khi đọc SKILL.md.

    Args (ngoài phần giống v3):
        planner_llm: chat model cho bước lập kế hoạch (ưu tiên cao nhất).
        planner_llm_name: tên LLM trong NAT config dùng làm gốc để clone (mặc định dùng `llm_name`).
        planner_use_langchain: `True` (mặc định) -> planner dựng bằng LangChain, thinking TẮT.
        planner_enable_thinking: bật lại thinking cho planner (mặc định `False`).
        planner_max_tokens: trần token cho kế hoạch (mặc định 1024).
        enable_planner: `False` -> bỏ bước planner, hành vi y hệt v3.
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    builder = SyncBuilder.current()
    config_llm = builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # LLM CHÍNH (vòng ReAct) — LangChain + thinking TẮT (như v3).
    llm = (build_langchain_llm(config_llm, enable_thinking=False, label="v4 main LLM")
           if main_llm_disable_thinking else config_llm)

    # LLM chọn skill (PHA 1).
    if selector_llm is None:
        base = (builder.get_llm(selector_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
                if selector_llm_name else config_llm)
        selector_llm = (build_selector_llm(base,
                                           enable_thinking=selector_enable_thinking,
                                           max_tokens=selector_max_tokens) if selector_use_langchain else base)

    # LLM planner (PHA 1.75).
    if not enable_planner:
        planner_llm = None
    elif planner_llm is None:
        base = (builder.get_llm(planner_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
                if planner_llm_name else config_llm)
        planner_llm = (build_langchain_llm(base,
                                           enable_thinking=planner_enable_thinking,
                                           max_tokens=planner_max_tokens,
                                           label="v4 planner LLM") if planner_use_langchain else base)

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    tools = [*TOOLS, *load_mcp_tools(servers)]

    middleware = PlannerSkillMiddleware(resolved, selector_llm, planner_llm, compact_context=compact_context)

    return create_agent(
        model=llm,
        tools=tools,
        middleware=[middleware],
        checkpointer=memory,
    )
