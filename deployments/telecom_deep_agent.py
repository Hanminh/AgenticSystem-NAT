"""Deep-agent deployment over the Telecom (Viettel) skills.

Same skills as `telecom_agent.py`, but built with the `deepagents` framework
(planning + filesystem + native sandbox `execute` + optional sub-agents) instead
of the plain ReAct graph.

Requires: pip install deepagents
NAT wires this via `deployments/telecom_deep_agent.py:my_agent`.
"""

from agenticskills import as_nat_graph, build_deep_agent

# `as_nat_graph` is what makes `nat serve`'s /chat/stream work: without it the
# wrapper validates every raw LangGraph state update against a schema requiring
# `messages`, and deepagents' todo/file/skill updates have none (see
# agenticskills/stream.py). /generate is unaffected either way.
my_agent = as_nat_graph(
    build_deep_agent(
        skills_subdir="Telecom_Skills",
        env_var="TELECOM_SKILL_DIR",  # optional override
        # mcp_config_path="conf/mcp_servers.json",   # optionally add MCP tools
    ))
