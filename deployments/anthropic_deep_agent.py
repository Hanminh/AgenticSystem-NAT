"""Deep-agent deployment over the Anthropic technical skills.

Same skills as `anthropic_agent.py`, but built with the `deepagents` framework
(planning + filesystem + optional sub-agents) instead of the plain ReAct graph.

Requires: pip install deepagents
NAT wires this via `deployments/anthropic_deep_agent.py:my_agent`.
"""

# from agenticskills import as_nat_graph, build_deep_agent
# # as_nat_graph -> safe token streaming for `nat serve` /chat/stream (see agenticskills/stream.py).
# my_agent = as_nat_graph(
#     build_deep_agent(
#         skills_subdir="anthropic_skills",
#         env_var="ANTHROPIC_SKILLS_DIR",  # optional override
#         # mcp_config_path="conf/mcp_servers.json",   # optionally add MCP tools
#     ))


from agenticskills.deep_agents.native_optimize_deep_agent_v2 import build_native_optimize_deep_agent_v2
from agenticskills.deep_agents.todo_event_stream import as_nat_graph_todo_events
my_agent = as_nat_graph_todo_events(
    build_native_optimize_deep_agent_v2(
        skills_subdir="anthropic_skills",
        env_var="ANTHROPIC_SKILLS_DIR",
        llm_name="llm",
        )
    )
