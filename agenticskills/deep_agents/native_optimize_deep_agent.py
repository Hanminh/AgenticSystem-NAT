"""native_optimize_deep_agent — deep agent tối ưu bằng CHÍNH các knob của deepagents 0.7.0.

Khác `optimize_deep_agent.py` (giữ deepagents làm orchestrator + đẩy skill sang sub-agent
`optimize_agent_v3`), file này GIỮ NGUYÊN kiến trúc deepagents thuần (progressive-disclosure
skills + filesystem + execute) nhưng chỉnh cho **nhanh và ĐÁNG TIN hơn** bằng đúng API mà bản
deepagents **0.7.0** (đã cài trong venv nvidia-nat) cung cấp. Không sửa code cũ.

Đã khảo sát trực tiếp deepagents 0.7.0 (`graph.py`, `profiles/harness/harness_profiles.py`,
`middleware/skills.py`, `middleware/subagents.py`). Stack middleware MẶC ĐỊNH của agent chính
(graph.py:815-846) — đã GỌN hơn 0.6.x (bỏ TodoList & Planning):

    SkillsMiddleware(if skills) -> FilesystemMiddleware(bắt buộc) ->
    SubAgentMiddleware(+tool `task`, nếu có sub-agent — GP tự thêm làm nó luôn có) ->
    SummarizationMiddleware -> PatchToolCallsMiddleware -> [profile extra] ->
    PromptCaching -> [Memory].

Ba nguyên nhân khiến deep agent thuần vừa CHẬM vừa HAY "narrate rồi dừng":

1. **Sampling variance**: `temperature` không ghim -> server vLLM tự lấy default (~0.7). Với
   tool-calling, cùng câu hỏi lúc phát `tool_call` (chạy), lúc chỉ trả văn xuôi "Em sẽ gọi
   API..." rồi KHÔNG kèm tool_call -> ReAct không có nhánh tool -> DỪNG.
2. **Thinking BẬT** (Qwen): sinh khối `<think>` dài trước mỗi vòng, nhân lên nhiều vòng.
3. **Prompt phình + progressive disclosure 2 chặng**: model phải tự quyết gọi `read_file`
   SKILL.md (chặng 1) rồi mới `execute` (chặng 2) — mỗi chặng một lần "tung xúc xắc".

Cách file này xử lý (dùng đúng knob 0.7.0):

* **Model dựng bằng LangChain**: `enable_thinking=False` + `temperature=0` (greedy) -> trị 1 & 2.
  Greedy biến "chập chờn" thành xác định-theo-câu-hỏi và gần như luôn cam kết `tool_call`.
* **HarnessProfile** (đăng ký theo đúng `provider:model`):
    - `GeneralPurposeSubagentProfile(enabled=False)` -> BỎ sub-agent general-purpose + tool
      `task` -> prompt ngắn hơn, model bớt bị "dụ" đi nhánh khác.
    - `excluded_middleware={"SummarizationMiddleware"}` -> bỏ vòng LLM tóm tắt (hội thoại
      ngắn không chạm ngưỡng 85% nên nó chỉ tổ phình stack).
    - `system_prompt_suffix` ép gọi tool ngay (chống narrate), đặt SÁT lịch sử hội thoại.
    - `excluded_tools` (tuỳ chọn) ẩn tool file không cần cho skill "đọc rồi chạy script".
* **system_prompt** riêng: bắt đọc SKILL.md khớp với `limit=1000` rồi CHẠY ngay trong cùng
  chuỗi, cấm kết thúc lượt bằng "Tôi sẽ gọi...".

⚠️  `register_harness_profile` ghi vào REGISTRY TOÀN CỤC của deepagents theo khoá
`provider:model`. Vì mọi deep agent trong tiến trình dùng CHUNG một model (Qwen) sẽ **cùng**
dính profile này. Mặc định `lean_profile=True` (đây là agent "optimize"), nhưng có thể tắt.
Xem tài liệu `my_instruction/DEEPAGENTS_0_7_NATIVE_OPTIMIZE.md`.
"""

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
from agenticskills.langgraph_agents.optimize_agent_v2 import build_langchain_llm
from agenticskills.mcp import load_mcp_tools

load_dotenv()

logger = logging.getLogger(__name__)

# Đã đăng ký profile cho khoá nào rồi (tránh log trùng khi build nhiều agent).
_REGISTERED_PROFILE_KEYS: set[str] = set()

# Prompt gốc: ép đọc-rồi-chạy trong CÙNG chuỗi, chống "narrate rồi dừng".
NATIVE_OPTIMIZE_SYSTEM_PROMPT = """\
Bạn là Elite Enterprise AI Agent có một thư viện SKILL. Ưu tiên: ĐÚNG và NHANH.

Quy trình cho mỗi yêu cầu:
1. Đối chiếu yêu cầu với danh sách skill (tên + mô tả) ở phần "Skills System" bên dưới.
2. Khớp một skill -> **GỌI `read_file` NGAY** trên đường dẫn skill đó với `limit=1000` để đọc
   trọn SKILL.md. ĐỪNG đoán nội dung skill.
3. Làm ĐÚNG như SKILL.md hướng dẫn: lệnh trong đó thường đã có đường dẫn tuyệt đối — copy
   nguyên, chỉ thay tham số, rồi **gọi `execute` NGAY** trong cùng lượt.
4. EFFICIENCY: gộp việc độc lập vào MỘT `execute` (nối bằng `&&` hoặc một script tự chứa).
   Chỉ tách lệnh khi bước sau thật sự cần output bước trước.
5. Trả lời NGẮN GỌN sau khi đã có kết quả tool.

TUYỆT ĐỐI KHÔNG kết thúc lượt bằng câu kiểu "Tôi sẽ gọi...", "Em sẽ chạy..." mà KHÔNG kèm một
lời gọi tool trong CHÍNH lượt đó. Nói một câu ngắn thì được, nhưng phải gọi tool ngay sau đó.
"""

# Nhắc lại ngắn, đặt SÁT hội thoại (suffix của profile) — điểm dễ ảnh hưởng hành vi nhất.
_ANTI_NARRATE_SUFFIX = (
    "Nhắc lại: nếu còn thiếu dữ liệu, hãy GỌI TOOL ngay trong lượt này "
    "(`read_file` để đọc SKILL.md khớp, rồi `execute`). KHÔNG hứa suông rồi dừng.")


def _register_lean_profile(
    model_name: str,
    *,
    disable_general_purpose: bool,
    exclude_summarization: bool,
    excluded_tools: tuple[str, ...],
    profile_key: str | None,
) -> str | None:
    """Đăng ký một HarnessProfile 'gọn' cho model, dùng API deepagents 0.7.0.

    Trả về khoá đã đăng ký (hoặc None nếu bản deepagents không có API profile — degrade êm).
    """
    try:
        from deepagents import (  # 0.7.0+
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            register_harness_profile,
        )
    except ImportError:
        logger.warning("[native_optimize] deepagents không có API HarnessProfile (cần >=0.7) "
                       "-> bỏ qua lean_profile, chỉ tối ưu ở model/prompt.")
        return None

    key = profile_key or f"openai:{model_name}"

    gp = GeneralPurposeSubagentProfile(enabled=False) if disable_general_purpose else None
    excluded_mw = frozenset({"SummarizationMiddleware"}) if exclude_summarization else frozenset()

    profile = HarnessProfile(
        system_prompt_suffix=_ANTI_NARRATE_SUFFIX,
        excluded_middleware=excluded_mw,
        excluded_tools=frozenset(excluded_tools),
        general_purpose_subagent=gp,
    )
    register_harness_profile(key, profile)  # GHI REGISTRY TOÀN CỤC (theo model)

    if key not in _REGISTERED_PROFILE_KEYS:
        _REGISTERED_PROFILE_KEYS.add(key)
        logger.info(
            "[native_optimize] đã đăng ký HarnessProfile cho '%s' | GP subagent=%s | "
            "bỏ Summarization=%s | ẩn tool=%s",
            key,
            "OFF" if disable_general_purpose else "giữ",
            exclude_summarization,
            list(excluded_tools) or "(không)",
        )
    return key


def build_native_optimize_deep_agent(
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
    system_prompt: str | None = None,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    backend=None,
    inherit_env: bool = True,
    backend_env: dict | None = None,
):
    """Dựng deep agent deepagents 0.7.0 thuần, tối ưu tốc độ + độ tin cậy tool-calling.

    Args:
        skills_dir / skills_subdir / env_var: thư mục skill (progressive disclosure của deepagents).
        llm_name: tên LLM trong NAT config.
        disable_thinking: `True` (mặc định) -> dựng LLM bằng LangChain với `enable_thinking=False`
            (clone endpoint từ config). Trị khối `<think>` của Qwen.
        temperature: ghim nhiệt độ. `0.0` (mặc định) = greedy -> tool-calling ỔN ĐỊNH, hết
            "chập chờn". `None` -> để nguyên (server tự quyết — KHÔNG khuyến nghị).
        lean_profile: `True` (mặc định) -> đăng ký HarnessProfile 0.7.0 để cắt middleware/tool
            thừa (xem cảnh báo REGISTRY TOÀN CỤC ở docstring module). `False` -> chỉ tối ưu
            model + system_prompt, KHÔNG đụng registry.
        disable_general_purpose: (khi lean_profile) bỏ sub-agent general-purpose + tool `task`.
        exclude_summarization: (khi lean_profile) loại `SummarizationMiddleware`.
        excluded_tools: (khi lean_profile) ẩn thêm tool (vd `("write_file","edit_file")` nếu skill
            chỉ đọc + chạy script). Mặc định rỗng để an toàn cho skill có ghi file (vd làm slide).
        profile_key: khoá registry. Mặc định `openai:<model_name>`. Đổi nếu muốn tách profile
            khỏi các deep agent khác cùng model (đăng ký model_name khác).
        system_prompt: thay `NATIVE_OPTIMIZE_SYSTEM_PROMPT`.
        memory / mcp_* / backend / inherit_env / backend_env: như `build_deep_agent`.

    Returns:
        Graph deepagents đã compile. Bọc `as_nat_graph(...)` trước khi wire vào langgraph_wrapper.
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    builder = SyncBuilder.current()
    config_llm = builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # --- Model: thinking OFF + temperature ghim (điểm quyết định độ tin cậy) ---------------
    if disable_thinking:
        llm = build_langchain_llm(config_llm,
                                  enable_thinking=False,
                                  temperature=temperature,
                                  label="native deep LLM")
    else:
        llm = config_llm
        if temperature is not None:
            try:
                llm = config_llm.bind(temperature=temperature)
            except Exception:  # pragma: no cover - phòng khi client không hỗ trợ bind
                logger.warning("[native_optimize] không bind được temperature vào config LLM.")

    # tên model để làm khoá profile (clone từ ChatOpenAI của config)
    model_name = getattr(config_llm, "model_name", None) or getattr(llm, "model_name", "unknown")

    # --- HarnessProfile 0.7.0 (tuỳ chọn) ---------------------------------------------------
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

    if backend is None:
        # Local, KHÔNG cô lập: `execute` chạy trên host tại PROJECT_ROOT. inherit_env=True để
        # `execute` dùng đúng virtualenv NAT (khỏi pip install lại mỗi lần).
        backend = LocalShellBackend(
            root_dir=str(PROJECT_ROOT),
            virtual_mode=False,   # 0.7.0 mặc định True -> phải TẮT để read_file đọc được
            inherit_env=inherit_env,  # đường dẫn TUYỆT ĐỐI của SKILL.md
            env=backend_env,
        )

    logger.info(
        "[native_optimize] skills=%s | thinking=%s | temperature=%s | lean_profile=%s",
        resolved,
        "OFF" if disable_thinking else "theo config",
        temperature,
        lean_profile,
    )

    return create_deep_agent(
        model=llm,
        tools=mcp_tools,                      # execute + file tools là built-in qua backend
        system_prompt=system_prompt or NATIVE_OPTIMIZE_SYSTEM_PROMPT,
        skills=[str(resolved)],               # progressive disclosure của deepagents
        backend=backend,
        checkpointer=memory,
    )
