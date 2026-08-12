"""Deployment v4 — deep agent LLM dựng bằng LangChain + tắt thinking (/no_think), flat todos.

Dùng để A/B với v3 (telecom_deep_agent.py) khi model còn rò `<think>`. Trỏ config vào:
    workflow:
      _type: langgraph_wrapper
      graph: deployments/telecom_deep_agent_v4.py:my_agent
"""
from agenticskills.deep_agents.native_optimize_deep_agent_v4 import build_native_optimize_deep_agent_v4
from agenticskills.deep_agents.todo_event_stream import as_nat_graph_todo_events

my_agent = as_nat_graph_todo_events(
    build_native_optimize_deep_agent_v4(
        skills_subdir="Telecom_Skills",
        env_var="TELECOM_SKILL_DIR",
        llm_name="llm",
        enable_todos=True,     # flat write_todos -> todos + streaming
        no_think=True,         # /no_think soft-switch Qwen (né việc proxy nuốt chat_template_kwargs)
        temperature=0.0,       # greedy -> tool-calling ổn định
    )
)
