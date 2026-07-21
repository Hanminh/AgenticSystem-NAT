"""Vietnamese telecom deployment (Viettel domain).

Skills: dien-giai-cuoc-skill (tariff explanation), tu-van-goi-skill (plan
advisory), viettel-pptx (Viettel-branded presentations).

NAT wires this via `deployments/telecom_agent.py:my_agent`.
"""

# ------OPTIMIZED AGENT V1------ #
from agenticskills import as_nat_graph, build_agent
# as_nat_graph -> safe token streaming for `nat serve` /chat/stream (see agenticskills/stream.py).
my_agent = as_nat_graph(build_agent(
    skills_subdir="Telecom_Skills",
    env_var="TELECOM_SKILL_DIR",  # optional override
    llm_name="llm",  # tên LLM trong khối `llms:` của config NAT
))


# ------OPTIMIZED AGENT V1------ #
# from agenticskills import as_nat_graph, build_optimized_agent
# my_agent = as_nat_graph(build_optimized_agent(
#     skills_subdir="Telecom_Skills",
#     env_var="TELECOM_SKILL_DIR",
#     llm_name="llm",
# ))


# ------OPTIMIZED AGENT V2------ #
# from agenticskills import as_nat_graph
# from agenticskills.langgraph_agents.optimize_agent_v2 import FINAL_ANSWER_TAG, build_optimized_agent_v2
# # `stream_tags=[FINAL_ANSWER_TAG]` -> /chat/stream CHỈ stream câu trả lời cuối (PHA 3).
# # Không có nó, client sẽ thấy cả JSON chọn skill của PHA 1 (`["dien-giai-cuoc-..."]`)
# # lẫn câu trung gian của PHA 2 ("Tôi đang gọi API...") nối vào câu trả lời.
# my_agent = as_nat_graph(
#     build_optimized_agent_v2(
#         skills_subdir="Telecom_Skills",
#         llm_name="llm",
#         env_var="TELECOM_SKILL_DIR",
#     ),
#     stream_tags=[FINAL_ANSWER_TAG],
# )

# ------OPTIMIZED AGENT V3------ #
# from agenticskills import as_nat_graph
# from agenticskills.langgraph_agents.optimize_agent_v3 import build_optimized_agent_v3

# my_agent = as_nat_graph(
#     build_optimized_agent_v3(skills_subdir="Telecom_Skills", llm_name="llm"),
#     stream_nodes=["model"],   # chỉ stream node ReAct, ẩn JSON chọn skill của PHA 1
# )