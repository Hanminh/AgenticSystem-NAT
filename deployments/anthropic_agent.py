"""Anthropic technical-skills deployment.

Skills: algorithmic-art, brand-guidelines, canvas-design, claude-api,
doc-coauthoring, docx, frontend-design, internal-comms, mcp-builder, pdf, pptx,
skill-creator, slack-gif-creator, theme-factory, web-artifacts-builder,
webapp-testing, xlsx.

NAT wires this via `deployments/anthropic_agent.py:my_agent`.
"""

# from agenticskills import as_nat_graph, build_agent

# # as_nat_graph -> safe token streaming for `nat serve` /chat/stream (see agenticskills/stream.py).
# my_agent = as_nat_graph(
#     build_agent(
#         skills_subdir="anthropic_skills",
#         env_var="ANTHROPIC_SKILLS_DIR",  # optional override
#         # --- Optionally connect MCP servers (their tools join bash/python_repl) ---
#         # mcp_config_path="conf/mcp_servers.json",   # or set MCP_CONFIG_PATH
#         # mcp_servers={
#         #     "filesystem": {
#         #         "command": "npx",
#         #         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
#         #         "transport": "stdio",
#         #     },
#         # },
#     ))

from agenticskills import as_nat_graph
from agenticskills.langgraph_agents.optimize_agent_v4 import build_optimized_agent_v4
my_agent = as_nat_graph(build_optimized_agent_v4(skills_subdir="anthropic_skills", llm_name="llm", main_llm_disable_thinking= False),
                        stream_nodes=["model"])


