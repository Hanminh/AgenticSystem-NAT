"""Deep-agent deployment whose `execute` tool runs inside an E2B sandbox.

Same Anthropic skills, but with real isolation: deepagents' `execute` runs in a
remote E2B container. The local skills folder is uploaded into the sandbox at
build time (provision_skills=True by default).

Requires: pip install langchain-e2b e2b deepagents  +  export E2B_API_KEY=...
NAT wires this via `deployments/anthropic_e2b_agent.py:my_agent`.
"""

from agenticskills import as_nat_graph, build_e2b_deep_agent

# as_nat_graph -> safe token streaming for `nat serve` /chat/stream (see agenticskills/stream.py).
my_agent = as_nat_graph(
    build_e2b_deep_agent(
        skills_subdir="anthropic_skills",
        env_var="ANTHROPIC_SKILLS_DIR",  # optional override for the LOCAL source
        # e2b_template="my-template-with-deps",   # optional custom template
        # provision_skills=False,                 # skip upload if template bakes skills
    ))
