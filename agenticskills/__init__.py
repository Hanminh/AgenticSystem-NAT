"""AgenticSkills: a generalized, skills-based agent on NeMo Agent Toolkit.

Layout
------
    agenticskills/
    ├── common.py          shared: resolve_skills_dir, _resolve_mcp_servers, PROJECT_ROOT
    ├── tools.py           shared: bash / python_repl
    ├── prompt.py          shared: system prompt builders
    ├── mcp.py             shared: MCP tool loading
    ├── skill_index.py     shared: Qdrant semantic skill search
    ├── langgraph_agents/  agents dựng bằng LangChain / LangGraph
    │       agent.py, agent_qdrant.py, stream.py, optimize_agent{,_v2,_v3,_v4}.py
    └── deep_agents/       agents dựng bằng deepagents (framework `deepagents`)
            deep_agent.py, e2b_agent.py

Public API vẫn phẳng và lazy: `from agenticskills import build_agent` hoạt động như trước, dù
module đã chuyển vào sub-package. Submodule cũng import được theo đường mới, ví dụ
`from agenticskills.langgraph_agents.optimize_agent_v4 import build_optimized_agent_v4`.
"""

from typing import TYPE_CHECKING

__all__ = [
    "build_agent",
    "build_deep_agent",
    "build_e2b_deep_agent",
    "resolve_skills_dir",
    "SkillMiddleware",
    "TOOLS",
    "bash",
    "python_repl",
    "load_mcp_tools",
    "load_mcp_config",
    "as_nat_graph",
    "StreamSafeGraph",
    "build_optimized_agent",
    "TwoPhaseSkillMiddleware",
    "build_context",
]

if TYPE_CHECKING:  # for type checkers / IDEs only
    from agenticskills.common import resolve_skills_dir
    from agenticskills.deep_agents.deep_agent import build_deep_agent
    from agenticskills.deep_agents.e2b_agent import build_e2b_deep_agent
    from agenticskills.langgraph_agents.agent import SkillMiddleware, build_agent
    from agenticskills.langgraph_agents.optimize_agent import (
        TwoPhaseSkillMiddleware,
        build_context,
        build_optimized_agent,
    )
    from agenticskills.langgraph_agents.stream import StreamSafeGraph, as_nat_graph
    from agenticskills.mcp import load_mcp_config, load_mcp_tools
    from agenticskills.tools import TOOLS, bash, python_repl

# name -> (module path, attribute) cho lazy import (PEP 562).
_LAZY = {
    "build_agent": ("agenticskills.langgraph_agents.agent", "build_agent"),
    "SkillMiddleware": ("agenticskills.langgraph_agents.agent", "SkillMiddleware"),
    "resolve_skills_dir": ("agenticskills.common", "resolve_skills_dir"),
    "build_deep_agent": ("agenticskills.deep_agents.deep_agent", "build_deep_agent"),
    "build_e2b_deep_agent": ("agenticskills.deep_agents.e2b_agent", "build_e2b_deep_agent"),
    "TOOLS": ("agenticskills.tools", "TOOLS"),
    "bash": ("agenticskills.tools", "bash"),
    "python_repl": ("agenticskills.tools", "python_repl"),
    "load_mcp_tools": ("agenticskills.mcp", "load_mcp_tools"),
    "load_mcp_config": ("agenticskills.mcp", "load_mcp_config"),
    "as_nat_graph": ("agenticskills.langgraph_agents.stream", "as_nat_graph"),
    "StreamSafeGraph": ("agenticskills.langgraph_agents.stream", "StreamSafeGraph"),
    "build_optimized_agent": ("agenticskills.langgraph_agents.optimize_agent", "build_optimized_agent"),
    "TwoPhaseSkillMiddleware": ("agenticskills.langgraph_agents.optimize_agent", "TwoPhaseSkillMiddleware"),
    "build_context": ("agenticskills.langgraph_agents.optimize_agent", "build_context"),
}


def __getattr__(name):  # PEP 562 lazy attribute access
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])
