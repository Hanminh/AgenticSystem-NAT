"""Generalized skills-agent factory.

This single module replaces the ~8 near-duplicate `*_graph.py` files from the
original project. Every deployment (anthropic / oneai / telecom / xlsx / ...)
is now just a call to `build_agent(...)` with a different skills directory.

Design notes:
- The agent is a single compiled ReAct graph from `create_agent` (model <-> a
  ToolNode of [bash, python_repl]). There is no redundant outer StateGraph.
- `SkillMiddleware` discovers skills ONCE per run (`before_agent`) and injects
  a stable system prompt on every model call (`wrap_model_call`), reusing the
  preformatted catalog rather than rebuilding it each turn.
- The chat model comes from the NAT config's `llms:` block, resolved through
  `SyncBuilder.current().get_llm(...)`.

A variant that builds the model directly with LangChain (and can turn Qwen's
`<think>` block off to cut latency) is kept in `backups/` — see `backups/README.md`.
"""

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
from langchain_core.messages import SystemMessage

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore
from skills_ref.models import SkillProperties
from skills_ref.utils import list_skills

from agenticskills.common import PROJECT_ROOT, SKILLS_ROOT, _resolve_mcp_servers, resolve_skills_dir
from agenticskills.mcp import load_mcp_tools
from agenticskills.prompt import build_skills_prompt, build_system_prompt
from agenticskills.tools import TOOLS

load_dotenv()

# `PROJECT_ROOT`, `SKILLS_ROOT`, `resolve_skills_dir`, `_resolve_mcp_servers` now live in
# `agenticskills.common` (shared with deep_agents). Re-exported above for backward compatibility.

__all__ = [
    "SkillMiddleware",
    "SkillsState",
    "build_agent",
    "resolve_skills_dir",
    "PROJECT_ROOT",
    "SKILLS_ROOT",
]


class SkillsState(AgentState):
    """Agent state extended with the discovered skill catalog and its prompt."""

    skills_metadata: NotRequired[list[SkillProperties]]
    skills_prompt: NotRequired[str]


class SkillMiddleware(AgentMiddleware):
    """Discovers skills once, then injects them into every model call."""

    state_schema = SkillsState  # type: ignore

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir

    def before_agent(self, state: SkillsState, runtime):  # type: ignore
        # Scan disk once and precompute the catalog prompt for the whole run.
        skills = list_skills(self.skills_dir)
        return {
            "skills_metadata": skills,
            "skills_prompt": build_skills_prompt(self.skills_dir, skills),
        }

    def _inject_skills(self, request: ModelRequest) -> ModelRequest:
        # Reuse the prompt built in before_agent; no per-turn reformatting.
        skills_prompt = request.state.get("skills_prompt", "")
        system_message = SystemMessage(content=build_system_prompt(skills_prompt))
        return request.override(system_message=system_message)

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
        # Required when NAT drives the agent asynchronously (ainvoke/astream):
        # the sync-only wrap_model_call raises NotImplementedError in async mode.
        return await handler(self._inject_skills(request))


def build_agent(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
):
    """Build a compiled skills agent (a ReAct graph) for one skills directory.

    The agent always has the built-in `bash` / `python_repl` tools plus the
    `skills/` folder. If any MCP servers are configured, their tools are loaded
    and made available to the model as well.

    Args:
        skills_dir: Explicit path to the skills directory (highest priority).
        skills_subdir: Folder under `<project>/skills/` to use (e.g. "anthropic_skills").
        env_var: Environment variable that may override the skills path.
        llm_name: Name of the LLM to resolve from the NAT builder (default "llm").
        memory: Optional LangGraph checkpointer for persistent conversation state.
        mcp_servers: Inline MCP server mapping (see agenticskills.mcp).
        mcp_config_path: Path to a JSON file with the MCP server mapping. Falls
            back to the MCP_CONFIG_PATH environment variable.
    """
    resolved = resolve_skills_dir(
        skills_dir, skills_subdir=skills_subdir, env_var=env_var
    )
    llm = SyncBuilder.current().get_llm(
        llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    tools = [*TOOLS, *load_mcp_tools(servers)]

    return create_agent(
        model=llm,
        tools=tools,
        middleware=[SkillMiddleware(resolved)],
        checkpointer=memory,
    )
