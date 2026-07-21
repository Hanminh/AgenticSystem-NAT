from pathlib import Path
from .models import SkillProperties
from .errors import ParseError, ValidationError
from .parser import read_properties
def list_skills(skills_dir: Path) -> list[SkillProperties]:
    """List all skills from a single skills directory (internal helper).

    Scans the skills directory for subdirectories containing SKILL.md files,
    parses YAML frontmatter, and returns skill metadata.

    Skills are organized as:
    skills/
    ├── skill-name/
    │   ├── SKILL.md        # Required: instructions with YAML frontmatter
    │   ├── script.py       # Optional: supporting files
    │   └── config.json     # Optional: supporting files

    Args:
        skills_dir: Path to the skills directory.
        source: Source of the skills ('user' or 'project').

    Returns:
        List of skill metadata dictionaries with name, description, path, and source.
    """
    # Check if skills directory exists
    if not skills_dir.exists():
        return []

    skills: list[SkillProperties] = []

    # Iterate through subdirectories
    for skill_dir in sorted(skills_dir.iterdir()):
        # Only real skill directories count — skip stray files and hidden entries.
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue

        # A single malformed/empty skill dir (e.g. missing SKILL.md) must NOT
        # crash discovery — that would take down the whole agent at before_agent.
        try:
            props = read_properties(skill_dir)
        except (ParseError, ValidationError) as exc:
            print(f"[skills] Skipping '{skill_dir.name}': {exc}")
            continue

        if props:
            skills.append(props)

    return skills