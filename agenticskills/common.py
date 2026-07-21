"""Shared utilities dùng bởi CẢ hai nhóm agent (langgraph_agents và deep_agents).

Tách riêng ở đây để nhóm `deep_agents` không phải import từ `langgraph_agents` chỉ vì cần
mấy hàm phân giải thư mục skill / cấu hình MCP.
"""

import os
from pathlib import Path

from agenticskills.mcp import load_mcp_config

# Project root: AgenticSkills/. Bundled skills live under `<root>/skills/<name>`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"


def resolve_skills_dir(
    skills_dir: Path | str | None = None,
    *,
    skills_subdir: str | None = None,
    env_var: str | None = None,
) -> Path:
    """Resolve the skills directory for a deployment.

    Resolution order (first match wins):
      1. `skills_dir` — an explicit path passed by the caller.
      2. `env_var`    — environment variable holding a path (allows overrides).
      3. `skills_subdir` — a folder bundled under `<project>/skills/`.
    """
    if skills_dir is not None:
        return Path(skills_dir)
    if env_var and os.getenv(env_var):
        return Path(os.getenv(env_var))  # type: ignore[arg-type]
    if skills_subdir:
        return SKILLS_ROOT / skills_subdir
    raise ValueError(
        "No skills directory could be resolved. Pass skills_dir, or set env_var, "
        "or provide skills_subdir."
    )


def _resolve_mcp_servers(
    mcp_servers: dict | None, mcp_config_path: Path | str | None
) -> dict | None:
    """Resolve the MCP server mapping: inline dict > config path > MCP_CONFIG_PATH."""
    if mcp_servers:
        return mcp_servers
    path = mcp_config_path or os.getenv("MCP_CONFIG_PATH")
    if path:
        return load_mcp_config(path)
    return None
