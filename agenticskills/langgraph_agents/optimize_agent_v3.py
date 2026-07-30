"""Skills agent 2 GIAI ĐOẠN + NGỮ CẢNH ĐẦY ĐỦ — lai giữa optimize_agent_v2 và optimize_agent.

Ý tưởng
-------
Lấy phần ĐẦU của `optimize_agent_v2`:

    PHA 1  — CHỌN SKILL       (1 lần gọi LLM LangChain, thinking TẮT)
       ↓  (code, KHÔNG gọi LLM)
    PHA 1.5 — CAT SKILL.md    ← đọc thẳng bằng Python, nạp vào system prompt

nhưng phần SAU thì dùng chiến lược NGỮ CẢNH ĐẦY ĐỦ của `optimize_agent` (KHÔNG phải kiểu
"chỉ nhớ 1 bước" của v2, cũng không phải "vứt tài liệu ở bước tổng hợp"):

    Ở MỌI vòng gọi LLM, ngữ cảnh gồm:
      [System]  persona + TOÀN BỘ nội dung SKILL.md đã chọn   (nạp sẵn, luôn có mặt)
      [Human]   yêu cầu gốc của người dùng
      [Human]   "## Các bước ĐÃ THỰC HIỆN": output của TẤT CẢ các lần gọi LLM + tool trước đó

Nhờ vậy model luôn thấy đủ tài liệu VÀ toàn bộ lịch sử -> chính xác nhất, phù hợp skill cần
nhiều bước / nhiều tool phụ thuộc nhau. Đổi lại ngữ cảnh dài hơn v2 (v2 nhanh hơn nhưng "quên").

Giống v2 và KHÁC optimize_agent:
  * SKILL.md được `cat` bằng CODE ở PHA 1.5 rồi nạp thẳng vào system prompt -> tiết kiệm một
    vòng LLM (model không phải tự nghĩ ra việc `cat SKILL.md`).
  * LLM (cả chọn skill lẫn LLM chính) dựng bằng **LangChain** với **thinking TẮT** — xem
    `optimize_agent_v2.build_langchain_llm`. Đây là điểm quyết định tốc độ khi stream.

Giống optimize_agent và KHÁC v2:
  * Nhật ký `build_execution_log` giữ **TẤT CẢ các bước** (không cắt tài liệu, không bỏ bước cũ).
    Vẫn KHÔNG bao giờ đưa lại TOOL INPUT (lệnh/mã do model tự sinh) và KHÔNG cắt output tool.
"""

import logging
from pathlib import Path
from typing import Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest  # type: ignore
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore
from skills_ref.models import SkillProperties

from agenticskills.common import _resolve_mcp_servers, resolve_skills_dir
from agenticskills.mcp import load_mcp_tools
from agenticskills.langgraph_agents.optimize_agent import (
    TwoPhaseSkillMiddleware,
    TwoPhaseState,
    _text_of,
    build_context,
    parse_selection,
)
from agenticskills.langgraph_agents.optimize_agent_v2 import (
    SELECTOR_MAX_TOKENS,
    build_langchain_llm,
    build_selector_llm,
    load_skill_docs,
    strip_think,
)
from agenticskills.tools import TOOLS

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt — tài liệu nạp sẵn + LỊCH SỬ ĐẦY ĐỦ
# --------------------------------------------------------------------------- #
EXEC_PROMPT_V3 = """\
Bạn là Elite Enterprise AI Agent.

Skill phù hợp ĐÃ ĐƯỢC CHỌN SẴN và **toàn bộ tài liệu của nó ĐÃ ĐƯỢC ĐỌC SẴN cho bạn** ở
dưới. Bạn KHÔNG cần tìm skill, KHÔNG cần `cat` lại bất cứ file nào.

Ở mỗi lượt, bạn được cung cấp:
  * tài liệu skill đầy đủ (ngay dưới đây),
  * yêu cầu gốc của người dùng,
  * và "## Các bước ĐÃ THỰC HIỆN" — LỊCH SỬ ĐẦY ĐỦ những gì bạn đã nói và mọi KẾT QUẢ tool.
Bạn nhớ được toàn bộ tiến trình, cứ dựa vào đó mà làm tiếp.

## Cách làm việc
1. Đọc tài liệu skill + lịch sử (nếu có). Làm ĐÚNG như tài liệu hướng dẫn — lệnh trong tài
   liệu đã có đường dẫn tuyệt đối, copy y nguyên, chỉ thay tham số.
2. Còn thiếu dữ liệu -> **GỌI TOOL NGAY trong lượt này**. Đủ dữ liệu -> đưa ra CÂU TRẢ LỜI CUỐI.
3. KHÔNG gọi lại tool nào đã cho kết quả trong lịch sử. KHÔNG `cat` lại SKILL.md.

> ❌ SAI: kết thúc lượt bằng *"Tôi sẽ gọi..."* mà không gọi tool.
> ✅ ĐÚNG: nói một câu ngắn về việc đang làm, VÀ gọi tool trong cùng lượt đó.

## Tài liệu skill (đã đọc sẵn — toàn bộ nội dung SKILL.md)

{docs}
"""

NO_SKILL_PROMPT_V3 = """\
Bạn là Elite Enterprise AI Agent. Không có skill nào phù hợp với yêu cầu này.

Hãy trả lời trực tiếp, hoặc dùng `bash` / `python_repl` nếu cần. Nếu gọi tool, gọi NGAY
trong lượt này. Bạn thấy đầy đủ lịch sử các bước đã làm ở "## Các bước ĐÃ THỰC HIỆN".
"""


def build_exec_prompt_v3(docs: str) -> str:
    return EXEC_PROMPT_V3.format(docs=docs) if docs else NO_SKILL_PROMPT_V3


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class FullContextState(TwoPhaseState):
    """State: kết quả PHA 1 + nội dung SKILL.md nạp ở PHA 1.5."""

    skill_docs: NotRequired[str]


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class FullContextSkillMiddleware(TwoPhaseSkillMiddleware):
    """PHA 1 (chọn skill) + PHA 1.5 (nạp SKILL.md bằng code) trong `before_agent`;
    vòng ReAct dùng NGỮ CẢNH ĐẦY ĐỦ (kế thừa `_prepare` của `TwoPhaseSkillMiddleware`).

    Chỉ override `_finish_selection`: thay vì đặt đường dẫn skill vào prompt (buộc model tự
    `cat`), ta nạp SẴN toàn bộ SKILL.md vào system prompt. Phần `_prepare` (dựng ngữ cảnh đầy
    đủ bằng `build_context`) giữ nguyên từ lớp cha.
    """

    state_schema = FullContextState  # type: ignore

    def _finish_selection(self, raw: str, skills: list[SkillProperties]) -> dict[str, Any]:
        selected = parse_selection(strip_think(raw), skills)
        docs = load_skill_docs(selected, self.skills_dir)   # PHA 1.5 — không tốn vòng LLM

        logger.info(
            "[v3] PHA 1: chọn %d/%d skill %s | PHA 1.5: nạp sẵn %s ký tự SKILL.md | ngữ cảnh: ĐẦY ĐỦ",
            len(selected),
            len(skills),
            [s.name for s in selected] or "(không có)",
            f"{len(docs):,}",
        )
        return {
            "selected_skills": [str(self.skills_dir / s.name) for s in selected],
            "skill_docs": docs,
            "exec_prompt": build_exec_prompt_v3(docs),
        }

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        """Y như lớp cha (ngữ cảnh đầy đủ) nhưng fallback prompt dùng bản v3."""
        exec_prompt = request.state.get("exec_prompt") or NO_SKILL_PROMPT_V3
        system_message = SystemMessage(content=exec_prompt)

        if not self.compact_context:
            return request.override(system_message=system_message)

        raw = list(request.messages)
        context = build_context(raw)   # type: ignore # LỊCH SỬ ĐẦY ĐỦ: LLM output + tool output (no tool input)

        before = sum(len(_text_of(m)) for m in raw)
        after = sum(len(_text_of(m)) for m in context)
        if before and after != before:
            logger.info("[v3] ngữ cảnh gửi model: %s -> %s ký tự (đã bỏ tool input)",
                        f"{before:,}", f"{after:,}")

        return request.override(system_message=system_message, messages=context) # type: ignore


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_optimized_agent_v3(
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
    main_llm_disable_thinking: bool = True,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    compact_context: bool = True,
):
    """Agent 2 giai đoạn, NGỮ CẢNH ĐẦY ĐỦ: chọn skill -> nạp sẵn SKILL.md -> ReAct giữ toàn bộ lịch sử.

    Dùng khi skill cần NHIỀU BƯỚC / nhiều tool phụ thuộc nhau (v2 nhanh hơn nhưng "quên" các
    bước cũ). LLM dựng bằng LangChain với thinking TẮT giống `optimize_agent_v2`.

    Args:
        main_llm_disable_thinking: `True` (mặc định) -> LLM chính dựng bằng LangChain (clone
            endpoint từ NAT config) với thinking TẮT. `False` -> dùng thẳng LLM của NAT config.
        selector_use_langchain / selector_enable_thinking / selector_max_tokens: cấu hình LLM
            chọn skill (PHA 1). Mặc định: LangChain, thinking TẮT, 256 token.
        compact_context: `False` => chỉ chọn skill + nạp SKILL.md, không dựng lại ngữ cảnh (A/B).
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    builder = SyncBuilder.current()
    config_llm = builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # LLM CHÍNH (vòng ReAct). Mặc định LangChain + thinking TẮT (clone endpoint từ config).
    llm = (build_langchain_llm(config_llm, enable_thinking=False, label="v3 main LLM")
           if main_llm_disable_thinking else config_llm)

    if selector_llm is None:
        base = (builder.get_llm(selector_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
                if selector_llm_name else config_llm)
        selector_llm = (build_selector_llm(base,
                                           enable_thinking=selector_enable_thinking,
                                           max_tokens=selector_max_tokens) if selector_use_langchain else base)

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    tools = [*TOOLS, *load_mcp_tools(servers)]

    middleware = FullContextSkillMiddleware(resolved, selector_llm, compact_context=compact_context) # pyright: ignore[reportArgumentType]

    return create_agent(
        model=llm,
        tools=tools,
        middleware=[middleware],
        checkpointer=memory,
    )
