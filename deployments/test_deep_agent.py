"""Deployment TEST (debug tool-calling) — deep agent v2 + skill lookup cục bộ."""
from agenticskills.deep_agents.native_optimize_deep_agent_v2 import build_native_optimize_deep_agent_v2
from agenticskills.deep_agents.todo_event_stream import as_nat_graph_todo_events

my_agent = as_nat_graph_todo_events(
    build_native_optimize_deep_agent_v2(
        skills_subdir="Test_Skills",
        env_var="TEST_SKILL_DIR",
        llm_name="llm",
    )
)
