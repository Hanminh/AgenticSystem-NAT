"""Deep-agent variant using deepagents' NATIVE skills + sandbox execution.

Unlike `build_agent` (which uses a custom SkillMiddleware and our own isolated
`bash` / `python_repl` tools), this factory uses deepagents end to end:

- **Native skills** via `skills=[...]` (deepagents' `SkillsMiddleware`) — the
  same Anthropic agent-skills pattern with progressive disclosure, so we do NOT
  reuse our `skills_ref` / `SkillMiddleware` here.
- **Native sandbox execution** via the built-in `execute` tool, routed through a
  `SandboxBackendProtocol` backend. This IS deepagents' own sandbox mechanism,
  which replaces our custom shell/python tools.
- **Native file tools** (`read_file` / `write_file` / `edit_file` / `ls` /
  `grep` / `glob`) provided by deepagents' filesystem middleware.

Backends (`deepagents.backends`):
- `LocalShellBackend` (default): runs `execute` commands on the LOCAL host.
  Convenient for dev, but per deepagents' own docs it provides **NO isolation**
  — use only with trusted skills/code.
- For real isolation, pass a remote sandbox backend implementing
  `SandboxBackendProtocol`. deepagents ships partner sandboxes: Daytona, Modal,
  Runloop, Vercel, and `LangSmithSandbox`. `execute` then runs inside that
  sandbox instead of on the host.

`deepagents` is an optional dependency; it is only imported inside the factory.
"""

from pathlib import Path

from dotenv import load_dotenv

from nat.builder.framework_enum import LLMFrameworkEnum  # type: ignore
from nat.builder.sync_builder import SyncBuilder  # type: ignore

from agenticskills.common import (
    PROJECT_ROOT,
    _resolve_mcp_servers,
    resolve_skills_dir,
)
from agenticskills.mcp import load_mcp_tools

load_dotenv()

# Extra guidance layered on top of deepagents' built-in skill/execute prompts.
# deepagents already injects how to use the `execute` tool and how skills work,
# so this only adds the persona and our batching/efficiency preference.
DEEP_AGENT_INSTRUCTIONS = """\
You are an Elite Enterprise AI Agent with a library of specialized skills.

For each task:
1. Pick the matching skill from the available skills, then READ its SKILL.md
   (and any referenced docs) before implementing — never guess the rules.
2. Run code, scripts, builds, and verification with the `execute` tool; it runs
   in the sandbox environment.
3. EFFICIENCY: minimise the number of tool calls. Put independent work into a
   SINGLE `execute` call (chain with `&&`, or run one self-contained script,
   e.g. a `python - <<'PY' ... PY` heredoc). Only split calls when a later step
   truly needs the output of an earlier one. Read several files in one call.
4. Respond concisely once the task is verified.
"""


def build_deep_agent(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    subagents: list | None = None,
    backend=None,
    skills_sources: list[str] | None = None,
    inherit_env: bool = True,
    backend_env: dict | None = None,
):
    """Build a compiled deep-agent using deepagents' native skills + sandbox.

    Args mirror `build_agent`, plus:
        subagents: optional list of deepagents SubAgent specs.
        backend: a `SandboxBackendProtocol` backend to run the `execute` tool in.
            Defaults to `LocalShellBackend` rooted at the project (LOCAL, NOT
            isolated). For real isolation pass a remote sandbox backend
            (Daytona / Modal / Runloop / Vercel / LangSmithSandbox / E2B).
        skills_sources: explicit deepagents `skills=` source paths. Use this when
            the skills live somewhere other than the resolved local dir — e.g. a
            path INSIDE a remote sandbox (see `build_e2b_deep_agent`). When None,
            the resolved local skills directory is used.
        inherit_env: for the default `LocalShellBackend`, inherit the parent
            process environment (PATH/VIRTUAL_ENV/PYTHONPATH). Default True so the
            `execute` shell uses the SAME virtualenv NAT runs in — otherwise
            `execute` runs a bare Python that is missing this project's libraries
            and the model wastes time re-`pip install`-ing them every run.
        backend_env: extra/override environment variables for the default backend.
    """
    llm = SyncBuilder.current().get_llm(
        llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    if skills_sources is None:
        resolved = resolve_skills_dir(
            skills_dir, skills_subdir=skills_subdir, env_var=env_var
        )
        skills_sources = [str(resolved)]

    servers = _resolve_mcp_servers(mcp_servers, mcp_config_path)
    mcp_tools = load_mcp_tools(servers)

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "The deep-agent variant needs 'deepagents'. Install it with: "
            "pip install deepagents"
        ) from exc

    if backend is None:
        # Local, non-isolated sandbox: `execute` runs on the host in PROJECT_ROOT.
        # inherit_env=True is important: without it LocalShellBackend runs with an
        # EMPTY environment, so `python`/`pip` resolve outside this virtualenv and
        # every run re-installs missing libraries. Inheriting the env makes the
        # execute shell use the same venv (deps already installed, installs persist).
        backend = LocalShellBackend(
            root_dir=str(PROJECT_ROOT),
            virtual_mode=False,
            inherit_env=inherit_env,
            env=backend_env,
        )

    return create_deep_agent(
        model=llm,
        tools=mcp_tools,  # execute + file tools are built-in via the backend
        system_prompt=DEEP_AGENT_INSTRUCTIONS,
        skills=skills_sources,  # native skill sources
        backend=backend,
        subagents=subagents or [],
        checkpointer=memory,
    )
