"""Deployment TEST (debug tool-calling) — deep agent v3 (flat todos) + skill lookup cục bộ."""
from agenticskills.deep_agents.native_optimize_deep_agent_v3 import build_native_optimize_deep_agent_v3
from agenticskills.deep_agents.todo_event_stream import as_nat_graph_todo_events

my_agent = as_nat_graph_todo_events(
    build_native_optimize_deep_agent_v3(
        skills_subdir="Test_Skills",
        env_var="TEST_SKILL_DIR",
        llm_name="llm",
        enable_todos=True,   # flat write_todos -> test cả todos + streaming + fix pickle
    )
)
