"""Deep-agent TỐI ƯU TỐC ĐỘ — deepagents điều phối, `optimize_agent_v3` làm sub-agent.

Khác `telecom_deep_agent.py` / `anthropic_deep_agent.py` (deepagents tự lo skill bằng
`SkillsMiddleware`) ở chỗ: phần việc "chọn skill -> đọc SKILL.md -> chạy tool" được đẩy sang
sub-agent `skill_runner` = graph `optimize_agent_v3` (chọn skill bằng 1 lần gọi LLM thinking
TẮT, `cat` SKILL.md bằng CODE, rồi ReAct thinking TẮT). Xem
`agenticskills/deep_agents/optimize_deep_agent.py` để biết chi tiết và các đánh đổi.

`stream_nodes=["model"]` là đủ để người dùng chỉ nhận câu trả lời cuối: token bên trong
sub-agent chạy ở một vòng Pregel riêng nên KHÔNG lọt ra stream của graph cha (đã đo).

NAT wire qua `deployments/optimize_deep_agent.py:my_agent`.
"""

from agenticskills import as_nat_graph
from agenticskills.deep_agents.optimize_deep_agent import build_optimize_deep_agent

my_agent = as_nat_graph(
    build_optimize_deep_agent(
        skills_subdir="Telecom_Skills",
        env_var="TELECOM_SKILL_DIR",  # optional override

        # Knob tốc độ (giá trị dưới đây là MẶC ĐỊNH, ghi ra cho dễ A/B):
        # main_llm_disable_thinking=True,   # LLM điều phối: thinking TẮT
        # main_agent_skills=False,          # agent chính KHÔNG mang catalog skill -> prompt ngắn
        # replace_general_purpose=True,     # chặn sub-agent "general-purpose" chậm của deepagents

        # Tinh chỉnh sub-agent v3 (forward thẳng sang build_optimized_agent_v3):
        # subagent_kwargs={"selector_enable_thinking": False, "compact_context": True},

        # mcp_config_path="conf/mcp_servers.json",   # thêm tool MCP cho agent chính
    ),
    stream_nodes=["model"],
)
