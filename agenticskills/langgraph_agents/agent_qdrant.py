"""Skills agent dùng SEMANTIC SEARCH (Qdrant) để nạp skill — bản Qdrant của `agent.py`.

Khác `agent.py` ở ĐÚNG MỘT ĐIỂM: catalog skill đưa vào system prompt.

    agent.py        : list_skills(dir)  -> nạp MÔ TẢ CỦA TẤT CẢ SKILL vào prompt
    agent_qdrant.py : embed(câu hỏi) -> similarity search trên Qdrant
                      -> chỉ nạp TOP-K skill giống nhất vào prompt

Rồi agent vẫn **tự chọn** skill trong số K đó và đọc `SKILL.md` bằng `bash` y như cũ
(pattern progressive disclosure không đổi). Ở đây CHƯA tối ưu ngữ cảnh — vòng ReAct
vẫn giữ nguyên hành vi của `agent.py` (xem `optimize_agent.py` nếu cần phần đó).

Vì sao đáng làm:
  * Prompt không còn phình theo số lượng skill. `anthropic_skills` (16 skill) tốn
    ~9.6k ký tự catalog, gửi lại ở MỌI vòng ReAct; với top_k=3 chỉ còn ~2k.
  * Thêm 100 skill mới vào thư viện cũng không làm prompt dài thêm một chữ.

Luồng
-----
  1. `before_agent`: lấy câu hỏi người dùng -> `SkillIndex.search(query, top_k)`
     -> danh sách (đường dẫn tuyệt đối, description, score).
  2. `wrap_model_call`: dựng system prompt từ đúng K skill đó (dùng lại
     `prompt.build_system_prompt` của `agent.py`, không sửa gì).

Chuẩn bị index trước khi chạy (xem `scripts/skill_index_cli.py`):

    python scripts/skill_index_cli.py build --skills-root skills

`build_qdrant_agent(auto_index=True)` (mặc định) sẽ tự build nếu collection còn rỗng.
"""

import logging
from pathlib import Path
from typing import Any, Callable, NotRequired

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (  # type: ignore
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore

from agenticskills.common import _resolve_mcp_servers, resolve_skills_dir
from agenticskills.mcp import load_mcp_tools
from agenticskills.prompt import build_system_prompt
from agenticskills.skill_index import SkillHit, SkillIndex, build_index, open_index
from agenticskills.tools import TOOLS

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3


class QdrantSkillsState(AgentState):
    """State bổ sung kết quả semantic search."""

    skills_prompt: NotRequired[str]
    selected_skills: NotRequired[list[str]]


def build_hits_prompt(hits: list[SkillHit]) -> str:
    """Catalog từ kết quả search — dùng ĐƯỜNG DẪN TUYỆT ĐỐI lấy thẳng từ Qdrant.

    Cùng định dạng với `prompt.build_skills_prompt` để agent dùng y như cũ.
    """
    if not hits:
        return "(Không tìm thấy skill phù hợp — hãy tự xử lý bằng `bash` / `python_repl`.)"
    return "\n".join(f"- **{hit.path}**: {hit.description}" for hit in hits)


def _user_request(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


class QdrantSkillMiddleware(AgentMiddleware):
    """Semantic search top-K skill một lần mỗi lượt, rồi nạp vào system prompt."""

    state_schema = QdrantSkillsState  # type: ignore

    def __init__(
        self,
        index: SkillIndex,
        *,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float | None = None,
    ):
        self.index = index
        self.top_k = top_k
        self.score_threshold = score_threshold

    def _search(self, state: QdrantSkillsState) -> dict[str, Any]:
        query = _user_request(list(state["messages"]))
        hits = self.index.search(query, top_k=self.top_k, score_threshold=self.score_threshold)
        logger.info(
            "[qdrant] top-%d skill cho %r: %s",
            self.top_k,
            query[:60],
            [f"{h.name} ({h.score:.3f})" for h in hits] or "(không có)",
        )
        return {
            "skills_prompt": build_hits_prompt(hits),
            "selected_skills": [hit.path for hit in hits],
        }

    def before_agent(self, state: QdrantSkillsState, runtime) -> dict[str, Any]:  # type: ignore
        return self._search(state)

    def _inject_skills(self, request: ModelRequest) -> ModelRequest:
        skills_prompt = request.state.get("skills_prompt", "")
        return request.override(system_message=SystemMessage(content=build_system_prompt(skills_prompt)))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._inject_skills(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        return await handler(self._inject_skills(request))


def build_qdrant_agent(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float | None = None,
    collection: str | None = None,
    qdrant_url: str | None = None,
    qdrant_path: Path | str | None = None,
    embedder_provider: str | None = None,
    embed_model: str | None = None,
    index: SkillIndex | None = None,
    auto_index: bool = True,
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
):
    """Như `build_agent`, nhưng skill được nạp qua semantic search trên Qdrant.

    Args (ngoài phần giống `build_agent`):
        top_k: số skill giống nhất được nạp vào prompt (mặc định 3).
        score_threshold: bỏ qua skill có cosine score thấp hơn ngưỡng (None = không lọc).
        collection / qdrant_url / qdrant_path: cấu hình Qdrant (mặc định đọc env —
            xem `skill_index`). Không có `QDRANT_URL` thì dùng Qdrant nhúng on-disk.
        embedder_provider: `"fastembed"` (local, mặc định) hoặc `"openai"`.
        index: truyền sẵn một `SkillIndex` (tiện cho test / tái sử dụng).
        auto_index: nếu collection RỖNG thì tự build từ `skills_dir`. Đặt False nếu
            bạn muốn quản lý index hoàn toàn bằng `scripts/skill_index_cli.py`.
    """
    resolved = resolve_skills_dir(skills_dir, skills_subdir=skills_subdir, env_var=env_var)

    if index is None:
        index = open_index(
            collection=collection,
            qdrant_url=qdrant_url,
            qdrant_path=qdrant_path,
            embedder_provider=embedder_provider,
            embed_model=embed_model,
        )

    if auto_index and index.count() == 0:
        logger.info("[qdrant] collection '%s' rỗng -> tự build từ %s", index.collection, resolved)
        build_index(resolved, index=index)

    llm = SyncBuilder.current().get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    tools = [*TOOLS, *load_mcp_tools(servers)]

    return create_agent(
        model=llm,
        tools=tools,
        middleware=[QdrantSkillMiddleware(index, top_k=top_k, score_threshold=score_threshold)],
        checkpointer=memory,
    )
