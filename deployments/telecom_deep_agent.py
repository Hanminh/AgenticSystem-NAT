"""Telecom (Viettel) skills deployment.

LỖI của bản deepagents (build_deep_agent) — progressive disclosure:
    Lượt 1 model chỉ thấy TÊN + MÔ TẢ skill; phải TỰ quyết định gọi tool để đọc SKILL.md.
    Mọi chỉ dẫn ép "gọi tool ngay / không được narrate" lại nằm BÊN TRONG SKILL.md (chưa
    đọc ở lượt 1). Do dao động sampling, có lúc model chỉ trả "Em sẽ gọi API ..." rồi KHÔNG
    phát tool_call -> graph không có nhánh tool để đi tiếp -> kết thúc workflow giữa chừng.

CÁCH SỬA — dùng optimize_agent_v3 (xây đúng để trị lỗi này):
    PHA 1    chọn skill      — 1 lần gọi LLM RIÊNG (không phụ thuộc "model có chịu đọc skill")
    PHA 1.5  cat SKILL.md    — bằng CODE, không cần model quyết định
    PHA 2    ReAct           — ngay lượt ĐẦU model đã có TOÀN BỘ SKILL.md (kèm "HÀNH ĐỘNG
                               NGAY / ❌ SAI khi chỉ narrate") + system prompt cấm narrate.
    => Loại bỏ điểm hỏng "narrate rồi dừng". thinking TẮT (main_llm_disable_thinking=True).

v3 (ngữ cảnh ĐẦY ĐỦ) hợp cho CẢ 3 telecom skill: dien-giai-cuoc-skill-optimize, tra-cuu-so-skill
(1 lệnh bash) lẫn viettel-pptx (nhiều bước — v3 giữ nguyên toàn bộ output các bước trước). Cả 3
chỉ cần tool `bash`/`python_repl` (v3 có sẵn), không cần filesystem/planning của deepagents.

`as_nat_graph(..., stream_nodes=["model"])`: chỉ stream token của node ReAct chính; token của
LLM chọn skill (PHA 1, chạy trong before_agent) KHÔNG lọt ra cho người dùng.

NAT wires this via `deployments/telecom_deep_agent.py:my_agent`.
"""

from agenticskills import as_nat_graph
# from agenticskills.langgraph_agents.optimize_agent_v3 import build_optimized_agent_v3

# my_agent = as_nat_graph(
#     build_optimized_agent_v3(
#         skills_subdir="Telecom_Skills",
#         env_var="TELECOM_SKILL_DIR",       # optional override
#         llm_name="llm",
#         main_llm_disable_thinking=True,     # LLM chính (ReAct) + selector: thinking TẮT
#         # mcp_config_path="conf/mcp_servers.json",   # optionally add MCP tools
#     ),
#     stream_nodes=["model"],
# )

# --- Bản deepagents cũ (giữ để A/B). Bỏ comment nếu THỰC SỰ cần planning/filesystem/sub-agents
#     của deepagents — nhưng nó có lỗi "narrate rồi dừng" mô tả ở đầu file.
from agenticskills import build_deep_agent
my_agent = as_nat_graph(
    build_deep_agent(skills_subdir="Telecom_Skills", env_var="TELECOM_SKILL_DIR"))
