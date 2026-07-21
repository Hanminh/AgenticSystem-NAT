"""Lightweight tests that do not require the NAT runtime or an LLM.

They validate the parts that must be correct for any deployment:
- skills discovery over each bundled skills directory,
- the skills-catalog / system-prompt builders.

Run:  python -m pytest tests/  (or)  python tests/test_agent.py
"""

import sys
from pathlib import Path

# Make the project root importable when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
import tempfile  # noqa: E402

from skills_ref.utils import list_skills  # noqa: E402
from agenticskills.prompt import build_skills_prompt, build_system_prompt  # noqa: E402
from agenticskills.mcp import load_mcp_config, load_mcp_tools  # noqa: E402

SKILLS_ROOT = PROJECT_ROOT / "skills"
SKILL_DIRS = ["anthropic_skills", "Telecom_Skills", "skills_oneai"]


def test_each_skills_dir_discovers_skills():
    for subdir in SKILL_DIRS:
        skills_dir = SKILLS_ROOT / subdir
        assert skills_dir.exists(), f"missing skills dir: {skills_dir}"
        skills = list_skills(skills_dir)
        assert skills, f"no skills discovered in {skills_dir}"
        for skill in skills:
            assert skill.name and skill.description


def test_prompt_builders_produce_nonempty_prompt():
    skills_dir = SKILLS_ROOT / "anthropic_skills"
    skills = list_skills(skills_dir)
    catalog = build_skills_prompt(skills_dir, skills)
    prompt = build_system_prompt(catalog)
    assert "Available Skills" in prompt
    assert catalog and catalog in prompt


def test_mcp_no_servers_returns_empty():
    assert load_mcp_tools(None) == []
    assert load_mcp_tools({}) == []


def test_mcp_config_parsing_bare_and_wrapped():
    mapping = {"srv": {"url": "http://x/mcp/", "transport": "streamable_http"}}
    with tempfile.TemporaryDirectory() as d:
        bare = Path(d) / "bare.json"
        bare.write_text(json.dumps(mapping))
        assert load_mcp_config(bare) == mapping

        wrapped = Path(d) / "wrapped.json"
        wrapped.write_text(json.dumps({"mcpServers": mapping}))
        assert load_mcp_config(wrapped) == mapping


if __name__ == "__main__":
    test_each_skills_dir_discovers_skills()
    test_prompt_builders_produce_nonempty_prompt()
    test_mcp_no_servers_returns_empty()
    test_mcp_config_parsing_bare_and_wrapped()
    for subdir in SKILL_DIRS:
        n = len(list_skills(SKILLS_ROOT / subdir))
        print(f"[OK] {subdir}: {n} skills discovered")
    print("[OK] mcp: config parsing + empty-servers handling")
    print("All checks passed.")
