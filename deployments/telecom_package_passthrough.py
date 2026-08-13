"""Telecom deployment — deep agent (native v3 + todos) + tool `send_package_details` passthrough.

Khác `telecom_deep_agent.py`:
  * Agent dựng bằng `build_native_optimize_deep_agent_passthrough` (v3 + tool passthrough).
  * Bọc bằng `as_nat_graph_passthrough` -> stream token đáp án + đẩy NAT step `write_todos` LẪN
    `package_details`.

Phục vụ cùng `cardbridge` (openresponses/cardbridge):
  * step write_todos     -> box tiến trình (không hộp thinking bọc quanh).
  * step package_details -> item type RIÊNG (`script_output`) để client render thẻ gói.
  * token                -> câu trả lời cuối (agent vẫn trả lời bằng chữ như hiện tại).

NAT wires this via `deployments/telecom_package_passthrough.py:my_agent`.
"""

from agenticskills.deep_agents.native_optimize_deep_agent_passthrough import (
    build_native_optimize_deep_agent_passthrough,
)
from agenticskills.deep_agents.stream_passthrough import as_nat_graph_passthrough

my_agent = as_nat_graph_passthrough(
    build_native_optimize_deep_agent_passthrough(
        skills_subdir="Telecom_Skills",
        env_var="TELECOM_SKILL_DIR",       # optional override
        llm_name="llm",
        enable_todos=True,                 # flat write_todos -> box tiến trình cho cardbridge
        # package_skill_subdir="tu-van-goi-skill-optimize-v3",  # nơi chứa package_details.py (mặc định)
    ),
    stream_nodes=["model"],                 # chỉ stream token của node ReAct chính
)
