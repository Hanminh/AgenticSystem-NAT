"""Skills agent 2 GIAI ĐOẠN, ngữ cảnh tối giản — bản tối ưu tốc độ của `agent.py`.

VẤN ĐỀ của `build_agent` (ReAct chuẩn)
-------------------------------------
Mỗi vòng lặp, LangGraph append vào `state["messages"]` cả **đầu vào** lẫn **đầu ra**:

    AIMessage(tool_calls=[python_repl(code=<200 dòng script>)])   <- TOOL INPUT
    ToolMessage("...kết quả...")                                   <- tool output

và **toàn bộ** danh sách đó được gửi lại cho LLM ở MỌI bước sau. Cộng thêm system
prompt luôn mang theo **catalog mô tả TẤT CẢ skill**. Prefill vì thế phình theo bậc
hai, và bước tổng hợp cuối — bước chậm nhất — phải nuốt lại nguyên cả catalog, cả
mã nguồn script, dù thứ nó cần chỉ là **kết quả** các tool.

THIẾT KẾ MỚI — 2 GIAI ĐOẠN
--------------------------
**Giai đoạn 1 — CHỌN SKILL** (`before_agent`, một lần duy nhất, một lần gọi LLM rẻ):
    câu hỏi người dùng + catalog mô tả NGẮN của các skill
        -> LLM chọn skill phù hợp -> trả về ĐƯỜNG DẪN THƯ MỤC skill.
    Prompt chọn skill và catalog **bị vứt khỏi ngữ cảnh ngay sau đó** — vòng ReAct
    không bao giờ nhìn thấy chúng nữa.

**Giai đoạn 2 — REACT + gọi tool**, với ngữ cảnh mỗi vòng CHỈ gồm:

    [System]  persona + ĐƯỜNG DẪN skill đã chọn   (không còn catalog)
    [Human]   yêu cầu gốc của người dùng          (nguyên vẹn)
    [Human]   nhật ký: LLM OUTPUT + TOOL OUTPUT   (KHÔNG có LLM input, KHÔNG có tool input)

Cụ thể ở mỗi bước chỉ append: **output của model** (phần text nó tự nói) và
**output của tool**. TUYỆT ĐỐI không đưa lại `tool_calls` (lệnh bash, mã nguồn
script) — đó là dữ liệu model TỰ SINH RA, nó không cần đọc lại để làm bước sau.

VÌ SAO VẪN CHÍNH XÁC
--------------------
Khác bản trước (cắt bớt output tool => agent làm sai), bản này **KHÔNG BAO GIỜ cắt
output tool**: nội dung SKILL.md và kết quả script được giữ **nguyên vẹn 100%** suốt
cả phiên. Số skill được chọn cũng không bị chặn trên. Toàn bộ phần tiết kiệm đến từ
những thứ KHÔNG mang thông tin mới:
  * catalog mô tả skill (chỉ cần một lần, ở giai đoạn chọn),
  * tool input (script/lệnh do chính model viết ra),
  * prompt chọn skill.

Để model vẫn "nhớ" mình vừa làm gì dù không thấy lại tool input, system prompt yêu
cầu nó **nói ngắn gọn ý định trước mỗi tool call** — câu đó là LLM output nên được
giữ lại.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import (  # type: ignore
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore
from skills_ref.models import SkillProperties
from skills_ref.utils import list_skills

from agenticskills.common import _resolve_mcp_servers, resolve_skills_dir
from agenticskills.mcp import load_mcp_tools
from agenticskills.tools import TOOLS

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# GIAI ĐOẠN 1 — chọn skill
# --------------------------------------------------------------------------- #
SELECTOR_PROMPT = """\
Bạn là bộ định tuyến skill. Dựa vào yêu cầu của người dùng, hãy chọn skill phù hợp \
từ danh sách dưới đây.

## Danh sách skill
{catalog}

## Quy tắc
- Chọn ĐỦ những skill THỰC SỰ cần cho yêu cầu — có thể một hoặc nhiều.
- Nếu không skill nào phù hợp (chào hỏi, hỏi đáp thường), trả về danh sách rỗng.
- CHỈ trả lời bằng một mảng JSON tên skill, không giải thích, không markdown.

Ví dụ: ["<skill_name_1>", "<skill_name_2>"]
Hoặc:  []
"""

_JSON_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


def build_selector_catalog(skills: list[SkillProperties]) -> str:
    """Catalog cho GIAI ĐOẠN 1: tên + mô tả. Chỉ tồn tại trong lần gọi chọn skill."""
    return "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)


def parse_selection(text: str, skills: list[SkillProperties]) -> list[SkillProperties]:
    """Đọc mảng JSON tên skill từ output của model; lọc theo tên hợp lệ.

    Không chặn trên số skill — model chọn bao nhiêu thì lấy bấy nhiêu.

    Có fallback nhiều lớp vì đây là bước quyết định — chọn sai/parse hỏng thì cả
    phiên hỏng. Ưu tiên AN TOÀN hơn tiết kiệm: parse hỏng thì dùng TẤT CẢ skill
    (chỉ tốn vài dòng đường dẫn trong prompt, không phải cả tài liệu).
    """
    by_name = {skill.name: skill for skill in skills}

    match = _JSON_ARRAY.search(text or "")
    if match:
        try:
            names = json.loads(match.group(0))
            if isinstance(names, list):
                # [] hợp lệ: model chủ động nói "không cần skill nào".
                return [by_name[n] for n in names if isinstance(n, str) and n in by_name]
        except json.JSONDecodeError:
            pass

    # Fallback 1: tên skill xuất hiện nguyên văn trong câu trả lời.
    mentioned = [skill for name, skill in by_name.items() if name in (text or "")]
    if mentioned:
        return mentioned

    # Fallback 2: không hiểu model nói gì -> đưa hết, thà chậm còn hơn chọn thiếu.
    logger.warning("[optimize] không parse được lựa chọn skill (%r) -> dùng toàn bộ skill", (text or "")[:200])
    return skills


def _user_request(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _text_of(message)
    return ""


# --------------------------------------------------------------------------- #
# GIAI ĐOẠN 2 — prompt thực thi (KHÔNG còn catalog)
# --------------------------------------------------------------------------- #
EXEC_PROMPT = """\
Bạn là Elite Enterprise AI Agent. Skill phù hợp cho yêu cầu này ĐÃ ĐƯỢC CHỌN SẴN.

## Skill được chọn cho yêu cầu này
{selected}

## Cách làm việc
1. **LEARN:** Đọc tài liệu của skill bằng `bash` TRƯỚC KHI làm gì khác. Đọc mọi thứ \
cần thiết trong MỘT lệnh, ví dụ `cat <skill_dir>/SKILL.md <skill_dir>/references/*.md`. \
KHÔNG BAO GIỜ đoán cách làm.
2. **EXECUTE:** Dùng `python_repl` hoặc `bash` để chạy đúng như tài liệu skill hướng dẫn.
3. **VERIFY:** Chỉ khi skill yêu cầu (chạy script kiểm tra, xem file kết quả).
4. **RESPOND:** Trả lời người dùng ngắn gọn, chính xác, dựa trên KẾT QUẢ tool.

## QUAN TRỌNG — cách bạn ghi nhớ
Để tiết kiệm thời gian suy luận, ở các bước sau bạn sẽ **KHÔNG nhìn thấy lại lệnh/mã \
nguồn mình đã gọi** — chỉ thấy **lời bạn nói** và **kết quả tool**.
=> Trước MỖI lần gọi tool, hãy nói MỘT CÂU ngắn gọn bạn đang làm gì và tại sao \
(ví dụ: "Đọc SKILL.md của dien-giai-cuoc-skill để biết cách gọi API"). Câu đó là thứ \
duy nhất bạn còn nhớ ở bước sau.

## Hiệu năng
- **Gộp lệnh:** nối nhiều lệnh vào MỘT lần gọi `bash` bằng `&&` hoặc `;`. Đọc nhiều \
file bằng một lệnh `cat a.md b.md c.md`.
- **Một khối Python:** viết MỘT script tự chứa làm hết mọi việc không phụ thuộc kết quả \
in ra của bước trước. Chỉ tách call khi thực sự cần xem kết quả trước đó.
- Mỗi tool call thừa = một vòng LLM chậm.
"""

_LOG_HEADER = ("## Nhật ký thực thi (những gì BẠN đã nói và KẾT QUẢ tool trả về)\n"
               "Lưu ý: lệnh/mã nguồn bạn đã gọi KHÔNG được hiển thị lại (để tiết kiệm ngữ cảnh).\n"
               "Kết quả tool dưới đây là ĐẦY ĐỦ, không bị cắt.\n")

_LOG_FOOTER = ("\n---\n"
               "Dựa TRÊN KẾT QUẢ Ở TRÊN: gọi tool tiếp theo nếu còn thiếu dữ liệu, "
               "hoặc đưa ra CÂU TRẢ LỜI CUỐI CÙNG.\n"
               "Không gọi lại tool nào đã cho kết quả ở trên.")


def build_exec_prompt(selected: list[SkillProperties], skills_dir: Path) -> str:
    """System prompt của giai đoạn 2: chỉ ĐƯỜNG DẪN skill đã chọn, không có catalog."""
    if selected:
        block = "\n".join(f"- **{skills_dir / skill.name}**: {skill.description}" for skill in selected)
    else:
        block = ("(Không skill nào phù hợp — trả lời trực tiếp, hoặc tự xử lý bằng "
                 "`bash` / `python_repl` nếu cần.)")
    return EXEC_PROMPT.format(selected=block)


# --------------------------------------------------------------------------- #
# Dựng ngữ cảnh: CHỈ LLM output + tool output
# --------------------------------------------------------------------------- #
def _text_of(message: BaseMessage | None) -> str:
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)




def _split_scratchpad(messages: list[BaseMessage]) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Ranh giới = HumanMessage cuối. Trước đó = hội thoại (giữ nguyên); sau đó = vòng tool."""
    last_human = -1
    for idx, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = idx
    if last_human < 0:
        return messages, []
    return messages[:last_human + 1], messages[last_human + 1:]


def build_execution_log(scratchpad: list[BaseMessage]) -> str:
    """Nhật ký các bước: CHỈ text model tự nói + output tool.

    Bỏ hoàn toàn `tool_calls` (tool input). Chỉ giữ TÊN tool để model biết kết quả
    đến từ đâu — tên tool là nhãn định danh, không phải đầu vào.

    Output tool được giữ NGUYÊN VẸN, không cắt — cắt chính là thứ từng làm agent
    đọc thiếu tài liệu skill và trả lời sai.
    """
    results: dict[str, ToolMessage] = {
        m.tool_call_id: m
        for m in scratchpad if isinstance(m, ToolMessage) and m.tool_call_id
    }

    lines: list[str] = [_LOG_HEADER]
    step = 0
    for message in scratchpad:
        if not isinstance(message, AIMessage):
            continue

        step += 1
        block = [f"### Bước {step}"]

        said = _text_of(message)  # LLM OUTPUT — giữ nguyên
        if said:
            block.append(f"Bạn nói: {said}")

        for call in message.tool_calls:  # tool INPUT bị bỏ, chỉ lấy output
            result = results.get(call.get("id") or "")
            output = _text_of(result) if result is not None else "(chưa có kết quả)"
            block.append(f"Kết quả `{call['name']}`:\n```\n{output}\n```")

        if len(block) > 1:
            lines.append("\n".join(block) + "\n")

    lines.append(_LOG_FOOTER)
    return "\n".join(lines)


def build_context(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Danh sách message gửi cho model: hội thoại + MỘT nhật ký (output-only)."""
    history, scratchpad = _split_scratchpad(messages)
    if not scratchpad:
        return messages  # vòng đầu: chưa có bước nào
    return [*history, HumanMessage(content=build_execution_log(scratchpad))]


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class TwoPhaseState(AgentState):
    """State bổ sung kết quả GIAI ĐOẠN 1."""

    selected_skills: NotRequired[list[str]]
    exec_prompt: NotRequired[str]


class TwoPhaseSkillMiddleware(AgentMiddleware):
    """Giai đoạn 1 (chọn skill) trong `before_agent`; giai đoạn 2 (ReAct ngữ cảnh tối giản)
    trong `wrap_model_call`."""

    state_schema = TwoPhaseState  # type: ignore

    def __init__(
        self,
        skills_dir: Path,
        selector_llm: BaseChatModel,
        *,
        compact_context: bool = True,
    ):
        self.skills_dir = skills_dir
        self.selector_llm = selector_llm
        self.compact_context = compact_context

    # ---- GIAI ĐOẠN 1 ----------------------------------------------------- #

    def _selector_messages(self, state: TwoPhaseState) -> tuple[list[BaseMessage], list[SkillProperties]]:
        skills = list_skills(self.skills_dir)
        prompt = SELECTOR_PROMPT.format(catalog=build_selector_catalog(skills))
        request = _user_request(list(state["messages"]))
        return [SystemMessage(content=prompt), HumanMessage(content=request)], skills

    def _finish_selection(self, raw: str, skills: list[SkillProperties]) -> dict[str, Any]:
        selected = parse_selection(raw, skills)
        logger.info(
            "[optimize] giai đoạn 1 — chọn %d/%d skill: %s",
            len(selected),
            len(skills),
            [s.name for s in selected] or "(không có)",
        )
        # Catalog + prompt chọn skill DỪNG TẠI ĐÂY: chỉ đường dẫn skill đã chọn đi tiếp.
        return {
            "selected_skills": [str(self.skills_dir / s.name) for s in selected],
            "exec_prompt": build_exec_prompt(selected, self.skills_dir),
        }

    def before_agent(self, state: TwoPhaseState, runtime) -> dict[str, Any]:  # type: ignore
        messages, skills = self._selector_messages(state)
        if not skills:
            return {"selected_skills": [], "exec_prompt": build_exec_prompt([], self.skills_dir)}
        response = self.selector_llm.invoke(messages)
        return self._finish_selection(_text_of(response), skills)

    async def abefore_agent(self, state: TwoPhaseState, runtime) -> dict[str, Any]:  # type: ignore
        messages, skills = self._selector_messages(state)
        if not skills:
            return {"selected_skills": [], "exec_prompt": build_exec_prompt([], self.skills_dir)}
        response = await self.selector_llm.ainvoke(messages)
        return self._finish_selection(_text_of(response), skills)

    # ---- GIAI ĐOẠN 2 ----------------------------------------------------- #

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        exec_prompt = request.state.get("exec_prompt") or build_exec_prompt([], self.skills_dir)
        system_message = SystemMessage(content=exec_prompt)

        if not self.compact_context:
            return request.override(system_message=system_message)

        raw = list(request.messages)
        context = build_context(raw)

        before = sum(len(_text_of(m)) for m in raw)
        after = sum(len(_text_of(m)) for m in context)
        if before and after < before:
            logger.info(
                "[optimize] ngữ cảnh gửi model: %s -> %s ký tự (giảm %.0f%%)",
                f"{before:,}",
                f"{after:,}",
                (1 - after / before) * 100,
            )

        return request.override(system_message=system_message, messages=context)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        return await handler(self._prepare(request))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_optimized_agent(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    selector_llm_name: str | None = None,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    compact_context: bool = True,
):
    """Agent 2 giai đoạn: chọn skill trước, rồi ReAct với ngữ cảnh chỉ-output.

    KHÔNG có giới hạn số skill được chọn, cũng KHÔNG cắt output tool — tiết kiệm
    đến hoàn toàn từ việc bỏ những thứ không mang thông tin mới (catalog skill,
    tool input, prompt chọn skill).

    Args (ngoài phần giống `build_agent`):
        selector_llm_name: LLM riêng cho GIAI ĐOẠN 1 (chọn skill). Mặc định dùng
            chung `llm_name`. Trỏ sang một model NHỎ/NHANH trong config sẽ rẻ hơn —
            đây chỉ là bài toán phân loại.
        compact_context: False => chỉ chọn skill, không nén ngữ cảnh (để A/B so sánh).
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    builder = SyncBuilder.current()
    llm = builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    selector_llm = (builder.get_llm(selector_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
                    if selector_llm_name else llm)

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    tools = [*TOOLS, *load_mcp_tools(servers)]

    middleware = TwoPhaseSkillMiddleware(resolved, selector_llm, compact_context=compact_context)

    return create_agent(
        model=llm,
        tools=tools,
        middleware=[middleware],
        checkpointer=memory,
    )
