"""Deep-agent factory backed by an E2B cloud sandbox.

Uses the official `langchain-e2b` integration so deepagents' `execute` tool runs
inside a real, isolated E2B sandbox (a remote container) instead of on the host.

Because an E2B sandbox is a *separate* filesystem, the local `skills/` folder is
not visible inside it. This factory therefore **provisions** (uploads) the chosen
skills directory into the sandbox and points deepagents' native `skills=` at the
uploaded remote path. Set `provision_skills=False` if your E2B template already
bakes the skills in.

Requirements:
    pip install langchain-e2b e2b deepagents
    export E2B_API_KEY="..."

Notes / caveats:
- A default E2B sandbox has Python but NOT this project's third-party deps
  (pandas, openpyxl, ...). Either let the agent `execute("pip install ...")` at
  runtime, or use a custom E2B template that pre-installs them (and the skills).
- The sandbox is a live billed resource. By default one is created at build time
  and killed at interpreter exit (`auto_kill=True`). Pass your own `sandbox=` to
  manage the lifecycle yourself.
"""

import atexit
from pathlib import Path

from dotenv import load_dotenv

from agenticskills.common import resolve_skills_dir
from agenticskills.deep_agents.deep_agent import build_deep_agent

load_dotenv()

DEFAULT_REMOTE_SKILLS_ROOT = "/home/user/skills"


def _safe_kill(sandbox) -> None:
    try:
        sandbox.kill()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def _provision_skills(backend, local_dir: Path, remote_dir: str) -> None:
    """Upload every file under `local_dir` into the sandbox at `remote_dir`."""
    files: list[tuple[str, bytes]] = []
    base = remote_dir.rstrip("/")
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            files.append((f"{base}/{rel}", path.read_bytes()))
    if files:
        backend.upload_files(files)


def build_e2b_deep_agent(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
    llm_name: str = "llm",
    memory=None,
    mcp_servers: dict | None = None,
    mcp_config_path: Path | str | None = None,
    subagents: list | None = None,
    sandbox=None,
    e2b_template: str | None = None,
    remote_skills_root: str = DEFAULT_REMOTE_SKILLS_ROOT,
    provision_skills: bool = True,
    auto_kill: bool = True,
):
    """Build a deep-agent whose `execute` tool runs inside an E2B sandbox.

    Args mirror `build_deep_agent`, plus:
        sandbox: an existing `e2b.Sandbox`; if None, one is created (and, when
            `auto_kill`, killed at interpreter exit).
        e2b_template: optional E2B template id used when creating the sandbox.
        remote_skills_root: base path inside the sandbox to upload skills to.
        provision_skills: upload the local skills dir into the sandbox (default).
            Set False if the template already contains the skills.
        auto_kill: register an atexit hook to kill a sandbox we created.
    """
    local_dir = resolve_skills_dir(
        skills_dir, skills_subdir=skills_subdir, env_var=env_var
    )

    try:
        from e2b import Sandbox
        from langchain_e2b import E2BSandbox
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "The E2B deep-agent needs 'langchain-e2b' and 'e2b'. Install with: "
            "pip install langchain-e2b e2b  (and set E2B_API_KEY)."
        ) from exc

    if sandbox is None:
        sandbox = (
            Sandbox.create(template=e2b_template)
            if e2b_template
            else Sandbox.create()
        )
        if auto_kill:
            atexit.register(_safe_kill, sandbox)

    backend = E2BSandbox(sandbox=sandbox)

    remote_dir = f"{remote_skills_root.rstrip('/')}/{local_dir.name}"
    if provision_skills:
        _provision_skills(backend, local_dir, remote_dir)

    return build_deep_agent(
        llm_name=llm_name,
        memory=memory,
        mcp_servers=mcp_servers,
        mcp_config_path=mcp_config_path,
        subagents=subagents,
        backend=backend,
        skills_sources=[remote_dir],
    )
