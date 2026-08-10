"""Deep agent TỐI ƯU TỐC ĐỘ: giữ deepagents làm orchestrator, giao phần việc skill cho
`optimize_agent_v3` chạy dưới dạng **sub-agent**.

Vì sao
------
`build_deep_agent` để deepagents tự lo phần skill bằng `SkillsMiddleware` (progressive
disclosure). Với LLM Qwen thinking BẬT, một yêu cầu đơn giản phải trả giá:

    vòng 1  model nghĩ "cần skill nào" -> gọi `execute`/`read_file` để `cat` SKILL.md
    vòng 2  model đọc xong -> mới gọi tool thật
    vòng 3  tổng hợp
    (mỗi vòng đều sinh trọn một khối <think>...</think> rồi vứt đi, và gửi lại
     TOÀN BỘ system prompt của 7 middleware deepagents)

`optimize_agent_v3` làm đúng việc đó nhưng nhanh hơn hẳn:

    PHA 1    chọn skill      — 1 lần gọi LLM, thinking TẮT, tối đa 256 token
    PHA 1.5  `cat` SKILL.md  — bằng CODE, KHÔNG tốn vòng LLM nào
    PHA 2    ReAct           — thinking TẮT, ngữ cảnh đầy đủ (SKILL.md + mọi output trước)

Module này ghép hai thứ: deepagents vẫn là agent chính (todo/filesystem/`execute`/sub-agent
cho các việc *không* thuộc skill nào), còn mọi việc "chạy một skill" thì đẩy sang sub-agent
`skill_runner` = graph của `optimize_agent_v3`.

Cơ chế
------
deepagents có sẵn `CompiledSubAgent` (`middleware/subagents.py:167`) — một spec
`{name, description, runnable}` nhận **graph đã compile sẵn**, không phải spec để nó tự dựng.
Tool `task(description=..., subagent_type=...)` sẽ `await runnable.ainvoke(...)` rồi lấy
**AIMessage cuối cùng** của sub-agent làm nội dung `ToolMessage` trả về cho agent chính.

Hai điểm đã kiểm chứng trong mã nguồn/thực nghiệm, quyết định rằng ghép thế này AN TOÀN:

1. **State không đụng nhau.** Tool `task` bơm nguyên state của agent chính vào sub-agent
   (`subagents.py:666`) và bơm nguyên state trả về ngược lại (`subagents.py:613`). Hai schema
   khác nhau (`skill_docs`/`exec_prompt` của v3 vs `files`/`todos` của deepagents) nhưng
   langgraph **lọc im lặng** mọi key lạ ở CẢ hai chiều — `graph/state.py:_get_updates` chỉ giữ
   `k in output_keys`. Không cần adapter.
2. **Token của sub-agent KHÔNG lọt ra stream người dùng.** Đo thực tế: graph lồng được gọi
   bằng `ainvoke` bên trong một node sẽ chạy vòng Pregel RIÊNG, `stream_mode="messages"` của
   graph cha không thấy token của nó — cha chỉ thấy `ToolMessage` kết quả. Nên
   `as_nat_graph(..., stream_nodes=["model"])` là đủ để người dùng chỉ nhận câu trả lời cuối.

Đánh đổi
--------
* Thêm **một vòng LLM của agent chính**: nó phải viết `description` cho `task(...)` rồi đọc kết
  quả để trả lời. Bù lại cắt được ít nhất 1-2 vòng "tìm + đọc SKILL.md" mà mỗi vòng đều mang
  system prompt deepagents rất dài + khối thinking. Prompt ở dưới ép nó trả lại NGUYÊN VĂN kết
  quả để vòng cuối sinh ít token nhất có thể.
* Sub-agent có **ngữ cảnh cô lập**: nó KHÔNG thấy hội thoại trước đó. Vì vậy `description`
  phải TỰ CHỨA (prompt đã nhấn mạnh điều này).
* Người dùng thấy first-token muộn hơn: trong lúc sub-agent chạy, không có token nào chảy ra.
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
from agenticskills.deep_agents.native_optimize_deep_agent import llm_from_config
from agenticskills.langgraph_agents.optimize_agent_v3 import build_optimized_agent_v3
from agenticskills.mcp import load_mcp_tools

load_dotenv()

logger = logging.getLogger(__name__)

# Tên mặc định của sub-agent nhanh.
SKILL_RUNNER_NAME = "skill_runner"

# Tên sub-agent mà deepagents TỰ THÊM nếu caller không khai báo trùng tên
# (`graph.py:829`). Khai một spec trùng tên này là cách chính thức để chặn nó.
GENERAL_PURPOSE_NAME = "general-purpose"

SKILL_RUNNER_DESCRIPTION = """\
Chuyên gia thực thi SKILL — đường nhanh cho MỌI việc mà thư viện skill xử lý được.

Nó tự chọn skill phù hợp, tự đọc SKILL.md và tự chạy các lệnh/tool cần thiết, rồi trả về câu
trả lời HOÀN CHỈNH. Nhanh hơn nhiều so với việc bạn tự đi tìm và đọc SKILL.md.

DÙNG NÓ khi yêu cầu người dùng khớp với một skill bất kỳ (tra cứu/diễn giải cước, xử lý
pdf/docx/xlsx/pptx, làm slide, thiết kế/web, sinh-sửa ảnh, ...). Chỉ TỰ LÀM khi chắc chắn
không skill nào phụ trách.

`description` bạn gửi phải TỰ CHỨA: nguyên văn yêu cầu của người dùng KÈM mọi dữ kiện cụ thể
(số điện thoại, CCCD, đường dẫn file, khoảng thời gian, ...). Sub-agent KHÔNG nhìn thấy lịch
sử hội thoại."""

ORCHESTRATOR_PROMPT = """\
Bạn là Elite Enterprise AI Agent, đóng vai ĐIỀU PHỐI. Ưu tiên số một là TỐC ĐỘ.

## Nguyên tắc
1. Yêu cầu thuộc phạm vi thư viện skill -> **giao NGAY cho `{runner}`** qua tool `task`:
   `task(description="<nguyên văn yêu cầu + mọi dữ kiện cụ thể>", subagent_type="{runner}")`.
   Đừng tự đi tìm skill, đừng tự `cat` SKILL.md — sub-agent làm việc đó nhanh hơn bạn.
2. `description` phải TỰ CHỨA. Sub-agent không thấy hội thoại: chép đủ số điện thoại, CCCD,
   đường dẫn file, khoảng thời gian, định dạng đầu ra mong muốn.
3. Nhiều nhiệm vụ ĐỘC LẬP -> gọi `task` NHIỀU LẦN trong CÙNG một lượt để chúng chạy song song.
   Nhiệm vụ phụ thuộc nhau mới gọi tuần tự.
4. Kết quả sub-agent trả về đã là câu trả lời hoàn chỉnh: **trả lại cho người dùng gần như
   NGUYÊN VĂN**. KHÔNG tóm tắt lại, KHÔNG diễn giải lại, KHÔNG chạy lại việc nó đã làm.
5. Chỉ tự dùng `execute` / các tool file khi việc đó KHÔNG thuộc skill nào (thao tác vặt, kiểm
   tra nhanh). Gộp việc độc lập vào MỘT lệnh `execute`.
6. Việc một bước thì KHÔNG cần `write_todos`. Chỉ lập kế hoạch khi thật sự nhiều giai đoạn.

> SAI: kết thúc lượt bằng *"Tôi sẽ giao cho sub-agent"* mà không gọi tool.
> ĐÚNG: nói một câu ngắn VÀ gọi `task` trong cùng lượt đó.
"""


def build_optimize_deep_agent(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    subagent_name: str = SKILL_RUNNER_NAME,
    subagent_description: str = SKILL_RUNNER_DESCRIPTION,
    subagent_kwargs: dict[str, Any] | None = None,
    subagent_runnable: Any = None,
    replace_general_purpose: bool = True,
    main_agent_skills: bool = False,
    system_prompt: str | None = None,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    backend=None,
    inherit_env: bool = True,
    backend_env: dict | None = None,
    extra_subagents: list | None = None,
):
    """Dựng deep agent (deepagents) có sub-agent `skill_runner` = graph `optimize_agent_v3`.

    Args:
        skills_dir / skills_subdir / env_var: thư mục skill, phân giải bởi `resolve_skills_dir`.
            Dùng chung cho CẢ sub-agent v3 và (nếu bật) skill catalog của agent chính.
        llm_name: tên LLM trong NAT config, dùng cho cả agent chính lẫn sub-agent. LLM của agent
            CHÍNH dùng TRỰC TIẾP từ config (không dựng lại bằng LangChain). Điều khiển thinking +
            temperature ngay trong YAML `llms.<llm_name>` (đặt `extra_body.chat_template_kwargs.
            enable_thinking: false` + `temperature: 0` để nhanh + tool-calling ổn định).
        subagent_name / subagent_description: định danh sub-agent trong tool `task`. Description
            là thứ DUY NHẤT agent chính dựa vào để quyết định có giao việc hay không — sửa nó
            nếu bộ skill của bạn hẹp hơn mô tả mặc định.
        subagent_kwargs: forward nguyên vẹn sang `build_optimized_agent_v3` (ví dụ
            `{"selector_enable_thinking": True, "compact_context": False}`). `skills_dir` và
            `llm_name` đã được truyền sẵn, không cần lặp lại.
        subagent_runnable: truyền graph đã compile sẵn để dùng thay v3 (A/B với v2/v4, hoặc
            tái dùng một graph đã dựng). Khi có, `subagent_kwargs` bị bỏ qua.
        replace_general_purpose: `True` (mặc định) -> đăng ký THÊM một bí danh trùng tên
            `general-purpose`, cùng trỏ vào graph v3. deepagents chỉ tự thêm sub-agent
            "general-purpose" của nó khi caller CHƯA khai tên đó (`graph.py:829`), nên đây là
            cách chặn: bỏ được cả một stack middleware + tránh việc agent chính lỡ giao việc
            cho sub-agent chậm (thinking BẬT, phải tự đi đọc SKILL.md). Đặt `False` nếu bạn
            thực sự cần một sub-agent đa dụng của deepagents.
        main_agent_skills: `False` (mặc định) -> agent CHÍNH không nạp `SkillsMiddleware`, tức
            không có catalog skill trong prompt. Vừa ngắn prompt vừa ép nó giao việc cho
            sub-agent. `True` -> nạp catalog cho agent chính (nó có thể tự làm, chậm hơn).
        system_prompt: thay `ORCHESTRATOR_PROMPT` mặc định.
        memory: checkpointer của langgraph cho agent chính.
        mcp_servers / mcp_config_path: tool MCP thêm cho agent chính (sub-agent v3 tự nạp MCP
            theo cấu hình riêng của nó trong `subagent_kwargs`).
        backend / inherit_env / backend_env: giống `build_deep_agent` — mặc định
            `LocalShellBackend` chạy TRÊN MÁY (không cô lập) tại `PROJECT_ROOT`.
        extra_subagents: các spec sub-agent khác, nối sau `skill_runner`.

    Returns:
        Graph deepagents đã compile. Bọc bằng `as_nat_graph(..., stream_nodes=["model"])` trước
        khi wire vào `langgraph_wrapper`.
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    builder = SyncBuilder.current()
    # LLM agent CHÍNH lấy TRỰC TIẾP từ config NAT — KHÔNG dựng lại bằng LangChain. thinking &
    # temperature điều khiển trong YAML `llms.<llm_name>` (enable_thinking: false + temperature: 0).
    # `llm_from_config` tắt stream_usage -> tránh stream_options gây lỗi proxy vLLM khi streaming.
    llm = llm_from_config(builder, llm_name)

    # --- Sub-agent nhanh: chính là graph optimize_agent_v3 ---------------------
    if subagent_runnable is None:
        subagent_runnable = build_optimized_agent_v3(skills_dir=resolved,
                                                    llm_name=llm_name,
                                                    **(subagent_kwargs or {}))

    # `CompiledSubAgent` = {name, description, runnable} — deepagents dùng graph NGUYÊN TRẠNG,
    # không bọc thêm middleware nào, nên v3 giữ đúng hành vi 3 pha của nó.
    subagents: list[dict[str, Any]] = [{
        "name": subagent_name,
        "description": subagent_description,
        "runnable": subagent_runnable,
    }]
    if replace_general_purpose and subagent_name != GENERAL_PURPOSE_NAME:
        subagents.append({
            "name": GENERAL_PURPOSE_NAME,
            "description": subagent_description,
            "runnable": subagent_runnable,
        })
    subagents.extend(extra_subagents or [])

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    mcp_tools = load_mcp_tools(servers)

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Biến thể deep-agent cần 'deepagents'. Cài bằng: pip install deepagents"
        ) from exc

    if backend is None:
        # Giống `build_deep_agent`: inherit_env=True để `execute` dùng ĐÚNG virtualenv NAT đang
        # chạy — nếu không, model tốn thời gian `pip install` lại thư viện ở mỗi lần chạy.
        backend = LocalShellBackend(
            root_dir=str(PROJECT_ROOT),
            virtual_mode=False,
            inherit_env=inherit_env,
            env=backend_env,
        )

    logger.info(
        "[optimize_deep] skills=%s | sub-agent=%s (optimize_agent_v3) | LLM agent chính từ config "
        "(thinking/temperature theo YAML) | catalog skill cho agent chính=%s | chặn general-purpose=%s",
        resolved,
        subagent_name,
        main_agent_skills,
        replace_general_purpose,
    )

    return create_deep_agent(
        model=llm,
        tools=mcp_tools,  # `execute` + tool file là built-in qua backend
        system_prompt=(system_prompt
                       or ORCHESTRATOR_PROMPT.format(runner=subagent_name)),
        # skills=None -> KHÔNG nạp SkillsMiddleware cho agent chính (prompt ngắn hơn hẳn).
        skills=[str(resolved)] if main_agent_skills else None,
        backend=backend,
        subagents=subagents,  # type: ignore[arg-type]
        checkpointer=memory,
    )
