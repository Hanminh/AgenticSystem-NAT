"""Deployment mẫu: skills nạp bằng SEMANTIC SEARCH trên Qdrant.

Giống `telecom_agent.py` (vẫn là ReAct + bash/python_repl, CHƯA tối ưu ngữ cảnh),
chỉ khác: thay vì nhồi mô tả của TẤT CẢ skill vào system prompt, mỗi câu hỏi được
embed rồi similarity search để lấy TOP-K skill giống nhất — chỉ K skill đó vào prompt.

Trỏ `skills_dir` vào thư mục GỐC (`skills/`) để search được trên toàn bộ thư viện
(Telecom_Skills + anthropic_skills + legal-skills + skills_oneai).

Chuẩn bị index trước (một lần):
    python scripts/skill_index_cli.py build --skills-root skills

NAT wire qua `deployments/telecom_qdrant_agent.py:my_agent`.
"""

from pathlib import Path

from agenticskills import as_nat_graph
from agenticskills.langgraph_agents.agent_qdrant import build_qdrant_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent

my_agent = as_nat_graph(
    build_qdrant_agent(
        skills_dir=PROJECT_ROOT / "skills",  # thư mục GỐC — quét đệ quy mọi nhóm skill
        llm_name="llm",
        top_k=3,  # số skill giống nhất nạp vào prompt
        # score_threshold=0.35,   # bỏ skill quá kém liên quan
        # auto_index=False,       # nếu bạn tự quản index bằng scripts/skill_index_cli.py
    ))
