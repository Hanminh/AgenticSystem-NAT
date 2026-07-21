"""OneAI deployment.

Skills: generate-image / edit-image, evaluate-resume (and other OneAI skills
under skills/skills_oneai).

NAT wires this via `deployments/oneai_agent.py:my_agent`.
"""

from agenticskills import as_nat_graph, build_agent

# as_nat_graph -> safe token streaming for `nat serve` /chat/stream (see agenticskills/stream.py).
my_agent = as_nat_graph(build_agent(
    skills_subdir="skills_oneai",
    env_var="ONEAI_SKILL_DIR",  # optional override
))
