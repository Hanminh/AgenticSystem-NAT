"""Skills agent 3 PHA, tối thiểu số vòng LLM — bản nhanh nhất.

⚠️ PHẠM VI ÁP DỤNG — ĐỌC TRƯỚC KHI DÙNG
=======================================
Module này được thiết kế **có chủ đích cho một lớp skill HẸP**, KHÔNG phải agent đa dụng
thay thế `agent.py` / `optimize_agent.py`. Skill phải thoả **cả hai** điều kiện:

  1. **Chỉ cần đọc DUY NHẤT một file `SKILL.md`** là đủ để thực thi — không có
     `references/*.md`, không phải khám phá thêm file nào.
  2. Chính `SKILL.md` đó **liệt kê sẵn danh sách API/tham số**; model chỉ việc **chọn ra
     tập con cần gọi rồi gọi MỘT lệnh duy nhất** là xong.

Skill mẫu chuẩn: `skills/Telecom_Skills/dien-giai-cuoc-skill-optimize` (6 API SOAP, một
script gọi song song, một lệnh cho cả câu hỏi).

Vì sao phải hẹp: kiến trúc 3 pha dưới đây **giả định vòng đời đúng 2 lượt LLM sau khi chọn
skill**. Nó cắt bỏ hẳn vòng LEARN, và ở bước tổng hợp thì **vứt bỏ tài liệu skill, chỉ giữ
câu hỏi + kết quả tool** (`request.override(tools=[])`). Với skill cần khám phá nhiều bước,
đọc nhiều file, hay chuỗi tool phụ thuộc lẫn nhau, những giả định đó SAI và agent sẽ mất
thông tin giữa chừng.

  → Skill nhiều file / nhiều bước: dùng `optimize_agent.py` (giữ toàn bộ nhật ký) hoặc `agent.py`.
  → Đừng "sửa v2 cho tổng quát hơn" — nó nhanh chính vì nó hẹp.

Ý tưởng: **mọi bước không cần trí thông minh thì đừng để LLM làm.**

    PHA 1 — CHỌN SKILL      (1 lần gọi LLM, thinking TẮT, trả về mảng JSON ngắn)
       ↓  (code, KHÔNG gọi LLM)
    PHA 1.5 — CAT SKILL.md  ← đọc thẳng bằng Python, nạp vào system prompt
       ↓
    PHA 2 — GỌI TOOL        (1 lần gọi LLM: đọc tài liệu đã có sẵn -> gọi tool ngay)
       ↓
    PHA 3 — TỔNG HỢP        (1 lần gọi LLM: CHỈ câu hỏi + kết quả tool, không gì khác)

So với `optimize_agent.py` / ReAct chuẩn, hai thứ bị cắt hẳn:

1. **Vòng LEARN biến mất.** ReAct chuẩn tốn một vòng LLM chỉ để model nghĩ ra
   "à, mình nên `cat SKILL.md`" rồi gọi `bash`. Nhưng skill đã được chọn ở PHA 1 rồi,
   nội dung file thì đọc bằng `Path.read_text()` là xong — không cần LLM suy luận gì cả.
   Tài liệu được nạp SẴN vào system prompt của PHA 2.

2. **Bước tổng hợp cuối cực nhẹ.** Nó chỉ nhận **câu hỏi người dùng + kết quả tool**.
   Không tài liệu skill, không mã nguồn script, không exec prompt, không nhật ký các
   bước. Đây vốn là bước chậm nhất trong workflow (ngữ cảnh dài nhất) — giờ nó ngắn nhất.

Kết quả: đường đi lý tưởng chỉ còn **3 lần gọi LLM** (chọn skill → gọi tool → trả lời).

THINKING TẮT Ở CẢ HAI LLM (mặc định)
------------------------------------
Cả LLM chọn skill (PHA 1) LẪN LLM chính (PHA 2 gọi tool + PHA 3 trả lời) đều được dựng bằng
LangChain (`ChatOpenAI`, clone endpoint từ NAT config) với **thinking TẮT** — vì khối `llms:`
của NAT không có knob `enable_thinking`. Tắt thinking cho LLM chính là điểm quyết định tốc độ
khi stream: Qwen bật thinking sinh nguyên khối `<think>...</think>` trước MỖI câu trả lời, ở
chế độ stream tất cả token đó phải chảy qua đường ống SSE + nhiều tầng xử lý -> chậm hơn hẳn.
Bỏ nó vừa làm đáp án sạch (không lẫn reasoning) vừa nhanh. Đổi bằng
`main_llm_disable_thinking=False` / `selector_use_langchain=False` nếu muốn giữ thinking.

AN TOÀN KHI TOOL LỖI
--------------------
Nếu tool chạy lỗi, agent **không** nhảy sang tổng hợp: nó quay lại chế độ EXEC với tài
liệu skill còn nguyên để model sửa và chạy lại (`keep_doc_on_error=True`). Chỉ khi có một
kết quả tool THÀNH CÔNG thì mới chuyển sang PHA 3.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest  # type: ignore
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore
from skills_ref.models import SkillProperties

from agenticskills.common import _resolve_mcp_servers, resolve_skills_dir
from agenticskills.mcp import load_mcp_tools
from agenticskills.langgraph_agents.optimize_agent import (
    TwoPhaseSkillMiddleware,
    TwoPhaseState,
    _split_scratchpad,
    _text_of,
    _user_request,
    parse_selection,
)
from agenticskills.tools import TOOLS

logger = logging.getLogger(__name__)

SELECTOR_MAX_TOKENS = 256  # chọn skill chỉ cần trả về một mảng JSON ngắn

# Nhãn gắn lên lần gọi LLM của PHA 3 (câu trả lời cuối cho người dùng).
# `StreamSafeGraph(stream_tags=[FINAL_ANSWER_TAG])` dùng nó để CHỈ stream câu trả lời
# này ra client — không stream JSON chọn skill của PHA 1, cũng không stream câu nói
# trung gian của PHA 2 ("Tôi đang gọi API...").
FINAL_ANSWER_TAG = "nat_final_answer"

# Dấu hiệu tool chạy LỖI (tools.py format lỗi thành text, không đổi status).
_FAILURE_MARKERS = ("Error (exit", "[Killed:", "Traceback (most recent call last)", "[LỖI MẠNG]",
                    "❌ LỖI THAM SỐ")


# --------------------------------------------------------------------------- #
# LLM cho PHA 1 — dựng bằng LangChain để TẮT ĐƯỢC THINKING
# --------------------------------------------------------------------------- #
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Bỏ khối `<think>...</think>` nếu server vẫn trả về nó.

    Lưới an toàn: nếu `enable_thinking=False` không có tác dụng, khối reasoning thường
    chứa những mảng JSON "nháp" model đang cân nhắc rồi loại bỏ — `parse_selection` sẽ
    vớ đúng cái nháp đó và chọn nhầm skill.
    """
    return _THINK_BLOCK.sub("", text or "").strip()


def build_langchain_llm(
    base_llm: BaseChatModel | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    enable_thinking: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 120.0,
    label: str = "LLM",
) -> BaseChatModel:
    """Dựng một `ChatOpenAI` bằng LangChain, có knob `enable_thinking`.

    Khối `llms:` trong YAML của NAT **không có knob `enable_thinking`**, nên Qwen sinh nguyên
    một khối `<think>...</think>` trước mọi câu trả lời. Với vLLM/SGLang, `chat_template_kwargs`
    được chuyển thẳng vào Qwen chat template -> `enable_thinking=False` bỏ khối đó, vừa sạch đáp
    án vừa nhanh hơn nhiều.

    Cấu hình được **clone từ `base_llm`** (chính LLM trong NAT config) khi nó là `ChatOpenAI`:
    `model` / `base_url` / `api_key`, và cả `temperature` / `max_tokens` nếu không truyền tay.
    Nhờ vậy không phải khai báo lại endpoint ở đâu cả.
    """
    if isinstance(base_llm, ChatOpenAI):
        model = model or base_llm.model_name
        base_url = base_url or base_llm.openai_api_base
        if api_key is None:
            secret = base_llm.openai_api_key
            api_key = secret.get_secret_value() if hasattr(secret, "get_secret_value") else secret
        if temperature is None:
            temperature = base_llm.temperature
        if max_tokens is None:
            max_tokens = base_llm.max_tokens

    model = model or os.getenv("LLM_MODEL")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    api_key = api_key or os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"

    if not model:
        raise ValueError("Không suy ra được model cho LLM LangChain. Truyền `model=` hoặc đặt env "
                         "LLM_MODEL, hoặc tắt cờ *_use_langchain để dùng LLM của NAT config.")

    logger.info("[v2] %s (LangChain): model=%s thinking=%s temperature=%s max_tokens=%s",
                label, model, "ON" if enable_thinking else "OFF", temperature, max_tokens)

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )


def build_selector_llm(
    base_llm: BaseChatModel | None = None,
    *,
    enable_thinking: bool = False,
    max_tokens: int = SELECTOR_MAX_TOKENS,
    **kwargs: Any,
) -> BaseChatModel:
    """LLM cho PHA 1 (chọn skill): thinking TẮT, `temperature=0`, `max_tokens=256`.

    Chọn skill là bài toán phân loại -> chỉ cần trả một mảng JSON ngắn, quyết định ổn định.
    """
    return build_langchain_llm(base_llm, enable_thinking=enable_thinking, temperature=0.0,
                               max_tokens=max_tokens, timeout=60.0, label="PHA 1 selector LLM",
                               **kwargs)


# --------------------------------------------------------------------------- #
# PHA 1.5 — đọc SKILL.md bằng CODE (không tốn vòng LLM nào)
# --------------------------------------------------------------------------- #
def load_skill_docs(selected: list[SkillProperties], skills_dir: Path) -> str:
    """`cat` SKILL.md của các skill đã chọn — bằng Python, không qua LLM/tool.

    Đây chính là chỗ tiết kiệm một vòng ReAct: ReAct chuẩn phải tốn một lần gọi LLM chỉ
    để model "nghĩ ra" rằng nó cần `cat SKILL.md`, rồi thêm một lần nữa để đọc kết quả.
    Skill đã chọn xong rồi thì đọc file là việc của code.
    """
    blocks: list[str] = []
    for skill in selected:
        skill_dir = skills_dir / skill.name
        doc = skill_dir / "SKILL.md"
        try:
            content = doc.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("[v2] không đọc được %s: %s", doc, exc)
            continue
        blocks.append(f"### Skill: {skill.name}\n"
                      f"Thư mục skill (đường dẫn tuyệt đối): {skill_dir}\n\n"
                      f"{content}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
EXEC_PROMPT = """\
Bạn là Elite Enterprise AI Agent.

Skill phù hợp ĐÃ ĐƯỢC CHỌN SẴN và **tài liệu của nó ĐÃ ĐƯỢC ĐỌC SẴN cho bạn** ở ngay
bên dưới. Bạn KHÔNG cần tìm skill, KHÔNG cần `cat` lại bất cứ file nào.

## ⚡ HÀNH ĐỘNG NGAY trong lượt này
Đọc tài liệu bên dưới, rồi **GỌI TOOL NGAY trong chính lượt này** để thực thi.

> ❌ SAI: kết thúc lượt bằng *"Tôi sẽ gọi script..."* / *"Bây giờ tôi cần..."* mà không gọi tool.
> ✅ ĐÚNG: nói một câu ngắn về việc đang làm, VÀ gọi tool trong cùng lượt đó.

Quy tắc:
- Làm ĐÚNG như tài liệu skill hướng dẫn. Lệnh trong tài liệu đã có đường dẫn tuyệt đối —
  copy y nguyên, chỉ thay tham số.
- **KHÔNG `cat` lại SKILL.md** — nó đã nằm ngay dưới đây rồi.
- Gộp mọi việc vào MỘT lần gọi tool nếu được. Mỗi tool call thừa là một vòng LLM chậm.
- Thiếu tham số bắt buộc (VD số điện thoại) → hỏi lại người dùng, KHÔNG đoán.

## Tài liệu skill (đã đọc sẵn — đây là toàn bộ nội dung SKILL.md)

{docs}
"""

NO_SKILL_PROMPT = """\
Bạn là Elite Enterprise AI Agent. Không có skill nào phù hợp với yêu cầu này.

Hãy trả lời trực tiếp, hoặc dùng `bash` / `python_repl` nếu cần thực thi.
Nếu gọi tool, hãy gọi NGAY trong lượt này — đừng chỉ mô tả kế hoạch rồi dừng.
"""

SYNTHESIS_PROMPT = """\
Bạn là trợ lý trả lời người dùng.

Dưới đây là câu hỏi của người dùng và KẾT QUẢ đã lấy được từ công cụ. Hãy trả lời dựa
HOÀN TOÀN trên kết quả đó:
1. **Diễn giải chi tiết** dữ liệu — chỉ ra con số/điểm quan trọng, so sánh, xu hướng.
2. **Kết luận ngắn gọn** trả lời thẳng câu hỏi.

TUYỆT ĐỐI KHÔNG bịa số liệu không có trong kết quả. Nếu một phần dữ liệu bị lỗi/thiếu,
nói rõ phần đó chưa lấy được. Không gọi thêm tool nào nữa.
"""


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class FastSkillState(TwoPhaseState):
    """State: kết quả PHA 1 + nội dung SKILL.md nạp ở PHA 1.5."""

    skill_docs: NotRequired[str]


# --------------------------------------------------------------------------- #
# Phân tích scratchpad
# --------------------------------------------------------------------------- #
def _looks_failed(text: str) -> bool:
    head = text[:400]
    return any(marker in head for marker in _FAILURE_MARKERS)


def _tool_outputs(scratchpad: list[BaseMessage]) -> list[tuple[str, str]]:
    """[(tên tool, output)] theo thứ tự. TOOL INPUT bị bỏ hoàn toàn."""
    results: dict[str, ToolMessage] = {
        m.tool_call_id: m
        for m in scratchpad if isinstance(m, ToolMessage) and m.tool_call_id
    }
    outputs: list[tuple[str, str]] = []
    for message in scratchpad:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            result = results.get(call.get("id") or "")
            if result is not None:
                outputs.append((call["name"], _text_of(result)))
    return outputs


def _is_doc_echo(output: str, docs: str) -> bool:
    """Model `cat` lại SKILL.md dù đã bảo đừng? Đừng tính đó là 'đã thực thi'."""
    if not docs or len(output) < 500:
        return False
    probe = output.strip()[:200]
    return bool(probe) and probe in docs


def build_synthesis_messages(user_question: str, outputs: list[tuple[str, str]]) -> list[BaseMessage]:
    """PHA 3: CHỈ câu hỏi người dùng + kết quả tool. Không gì khác."""
    blocks = [f"## Kết quả từ `{name}`\n```\n{output}\n```" for name, output in outputs]
    return [
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=f"## Câu hỏi của người dùng\n{user_question}\n\n" + "\n\n".join(blocks)),
    ]


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
class FastSkillMiddleware(TwoPhaseSkillMiddleware):
    """PHA 1 (chọn skill) + PHA 1.5 (cat SKILL.md bằng code) trong `before_agent`;
    PHA 2 (gọi tool) và PHA 3 (tổng hợp) trong `wrap_model_call`."""

    state_schema = FastSkillState  # type: ignore

    def __init__(
        self,
        skills_dir: Path,
        selector_llm: BaseChatModel,
        *,
        keep_doc_on_error: bool = True,
        compact_context: bool = True,
    ):
        super().__init__(skills_dir, selector_llm, compact_context=compact_context)
        self.keep_doc_on_error = keep_doc_on_error

    # ---- PHA 1 + 1.5 ------------------------------------------------------ #

    def _finish_selection(self, raw: str, skills: list[SkillProperties]) -> dict[str, Any]:
        selected = parse_selection(strip_think(raw), skills)
        docs = load_skill_docs(selected, self.skills_dir)  # PHA 1.5 — không tốn vòng LLM

        logger.info(
            "[v2] PHA 1: chọn %d/%d skill %s | PHA 1.5: nạp sẵn %s ký tự SKILL.md",
            len(selected),
            len(skills),
            [s.name for s in selected] or "(không có)",
            f"{len(docs):,}",
        )
        return {
            "selected_skills": [str(self.skills_dir / s.name) for s in selected],
            "skill_docs": docs,
            "exec_prompt": EXEC_PROMPT.format(docs=docs) if docs else NO_SKILL_PROMPT,
        }

    # ---- PHA 2 / PHA 3 ---------------------------------------------------- #

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        state = request.state
        exec_prompt = state.get("exec_prompt") or NO_SKILL_PROMPT
        docs = state.get("skill_docs") or ""

        raw = list(request.messages)
        if not self.compact_context:
            return request.override(system_message=SystemMessage(content=exec_prompt))

        history, scratchpad = _split_scratchpad(raw)
        outputs = _tool_outputs(scratchpad)

        # Kết quả THỰC THI = kết quả tool, không tính lần model lỡ cat lại tài liệu.
        executions = [(name, out) for name, out in outputs if not _is_doc_echo(out, docs)]
        succeeded = [(n, o) for n, o in executions if not _looks_failed(o)]

        # --- PHA 3: có kết quả thành công -> tổng hợp với ngữ cảnh TỐI THIỂU ---
        if succeeded and not (self.keep_doc_on_error and executions and _looks_failed(executions[-1][1])):
            messages = build_synthesis_messages(_user_request(history), executions)
            self._log("PHA 3 (tổng hợp)", raw, messages)
            # Gắn NHÃN lên lần gọi LLM này: đây là câu trả lời cho người dùng. Chỉ token
            # của nó mới được stream ra client (xem stream.StreamSafeGraph(stream_tags=...)).
            # Không gắn thì client sẽ thấy cả JSON chọn skill lẫn câu trung gian.
            # tools=[] -> model không thể gọi thêm tool, buộc phải trả lời ngay.
            return request.override(model=request.model.with_config(tags=[FINAL_ANSWER_TAG]),
                                    system_message=messages[0],
                                    messages=messages[1:],
                                    tools=[])

        # --- PHA 2: chưa có kết quả (hoặc tool lỗi, cần tài liệu để sửa) ---
        messages: list[BaseMessage] = list(history)
        if executions:  # tool lỗi -> đưa lại kết quả lỗi (không đưa tool input)
            fails = "\n\n".join(f"## Kết quả `{name}` (LỖI)\n```\n{out}\n```" for name, out in executions)
            messages.append(
                HumanMessage(content=f"{fails}\n\n---\nTool vừa chạy LỖI. Dựa vào tài liệu skill trong "
                             "system prompt, hãy SỬA và GỌI LẠI TOOL ngay trong lượt này."))

        self._log("PHA 2 (gọi tool)", raw, [SystemMessage(content=exec_prompt), *messages])
        return request.override(system_message=SystemMessage(content=exec_prompt), messages=messages)

    def _log(self, phase: str, raw: list[BaseMessage], sent: list[BaseMessage]) -> None:
        before = sum(len(_text_of(m)) for m in raw)
        after = sum(len(_text_of(m)) for m in sent)
        logger.info("[v2] %s — ngữ cảnh gửi model: %s ký tự (state đang giữ %s)", phase, f"{after:,}",
                    f"{before:,}")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_optimized_agent_v2(
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
    keep_doc_on_error: bool = True,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    compact_context: bool = True,
):
    """Agent 3 pha: chọn skill -> nạp sẵn SKILL.md (bằng code) -> gọi tool -> tổng hợp.

    Đường đi lý tưởng chỉ tốn **3 lần gọi LLM**, trong đó lần đầu (chọn skill) tắt thinking.

    Args:
        selector_use_langchain: `True` (mặc định) -> PHA 1 dùng `ChatOpenAI` clone từ LLM
            của NAT config nhưng **tắt thinking**. `False` -> dùng thẳng LLM của config.
        selector_enable_thinking: bật lại thinking cho PHA 1 (mặc định `False`).
        main_llm_disable_thinking: `True` (mặc định) -> LLM CHÍNH (PHA 2 gọi tool + PHA 3 tổng
            hợp) **không** lấy từ NAT config mà được dựng bằng LangChain (clone endpoint từ config)
            với **thinking TẮT**. Đây là điểm quyết định tốc độ khi stream: bỏ khối `<think>` giúp
            đáp án vừa sạch vừa sinh ít token hơn nhiều. `False` -> dùng thẳng LLM của NAT config
            (thinking theo mặc định của model).
        keep_doc_on_error: khi tool chạy LỖI thì KHÔNG nhảy sang tổng hợp — quay lại PHA 2
            với tài liệu skill còn nguyên để model sửa và chạy lại. Mặc định `True`.
        compact_context: `False` => tắt toàn bộ cơ chế nén (chỉ còn PHA 1 + 1.5), để A/B.
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    builder = SyncBuilder.current()
    config_llm = builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # LLM CHÍNH (PHA 2 + PHA 3). Mặc định dựng bằng LangChain với thinking TẮT — clone endpoint
    # từ LLM của NAT config, chỉ ghi đè enable_thinking=False. Không giới hạn max_tokens để đáp án
    # dài không bị cắt (build_langchain_llm clone max_tokens từ config nếu config có đặt).
    llm = (build_langchain_llm(config_llm, enable_thinking=False, label="PHA 2/3 main LLM")
           if main_llm_disable_thinking else config_llm)

    if selector_llm is None:
        base = (builder.get_llm(selector_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
                if selector_llm_name else config_llm)
        selector_llm = (build_selector_llm(base,
                                           enable_thinking=selector_enable_thinking,
                                           max_tokens=selector_max_tokens) if selector_use_langchain else base)

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    tools = [*TOOLS, *load_mcp_tools(servers)]

    middleware = FastSkillMiddleware(
        resolved,
        selector_llm,
        keep_doc_on_error=keep_doc_on_error,
        compact_context=compact_context,
    )

    return create_agent(
        model=llm,
        tools=tools,
        middleware=[middleware],
        checkpointer=memory,
    )
