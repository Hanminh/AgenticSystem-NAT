"""Deployment TEST v3 (ReAct, cat SKILL.md bằng code) — so sánh độ tin cậy với deepagents v2."""
from agenticskills import as_nat_graph
from agenticskills.langgraph_agents.optimize_agent_v3 import build_optimized_agent_v3

my_agent = as_nat_graph(
    build_optimized_agent_v3(
        skills_subdir="Test_Skills",
        env_var="TEST_SKILL_DIR",
        llm_name="llm",
        main_llm_disable_thinking=False,   # NIM không có chat_template_kwargs -> để config LLM nguyên
    ),
    stream_nodes=["model"],
)
