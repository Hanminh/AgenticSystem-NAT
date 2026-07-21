# AgenticSkills

A skills-based AI agent built on **NeMo Agent Toolkit (NAT)** + LangGraph.
The agent reads a skill's `SKILL.md` to learn how to do a task, then executes it
with `bash` and `python_repl`.

This project is a cleaned-up, generalized rewrite of an earlier prototype that
had ~8 near-duplicate graph files. All of that logic now lives in **one
parametrized factory**, `agenticskills.build_agent(...)`; each deployment is a
three-line file that only picks a skills directory.

## Layout

```
AgenticSkills/
├── agenticskills/            # the single generalized agent package
│   ├── agent.py              # SkillMiddleware + build_agent() factory (LangGraph)
│   ├── deep_agent.py         # build_deep_agent() factory (deepagents framework)
│   ├── prompt.py             # system prompt + skill-catalog builders
│   ├── tools.py              # bash (chained-command safe) + python_repl
│   └── mcp.py                # optional MCP-server tool loading
├── skills_ref/               # skill discovery/validation framework
│   ├── utils.py              # list_skills()
│   ├── parser.py             # SKILL.md frontmatter parsing
│   └── validator.py, models.py, prompt.py, errors.py
├── deployments/              # thin entrypoints NAT wires as `my_agent`
│   ├── anthropic_agent.py    # -> skills/anthropic_skills
│   ├── telecom_agent.py      # -> skills/Telecom_Skills
│   └── oneai_agent.py        # -> skills/skills_oneai
├── conf/                     # NAT configs
│   ├── config.yaml           # supervisor + anthropic/telecom agents
│   ├── conf_oneai.yaml       # + oneai agent + postgres synthesis
│   ├── conf_oneai_mem0.yaml  # + mem0 long-term memory
│   └── conf_redis.yaml       # Redis-memory supervisor variant
├── skills/                   # bundled skills (anthropic / telecom / oneai)
├── tests/test_agent.py       # runtime-free discovery + prompt tests
├── .env                      # API keys + optional *_SKILL_DIR overrides
└── requirements.txt
```

## Architecture

`build_agent()` returns a single **compiled ReAct graph** (`create_agent`): the
model calls `bash` / `python_repl` in a loop until done. There is no redundant
outer graph.

`SkillMiddleware`:
- **`before_agent`** — scans the skills directory once via `list_skills()` and
  precomputes the skill catalog for the whole run.
- **`wrap_model_call`** — injects a stable system prompt (persona + catalog +
  the IDENTIFY→LEARN→EXECUTE→VERIFY→RESPOND loop) on every model call, reusing
  the precomputed catalog rather than rebuilding it each turn.

**Latency:** the prompt carries an *Efficiency Protocol* that pushes the model to
**batch** tool calls (chain shell commands with `&&`, `cat` several docs at once,
one Python block, parallel independent calls). Fewer sequential LLM round-trips
is the main speed lever on a large model. The `bash` tool applies its safety
blocklist to **each** sub-command of a chained string.

**Execution isolation (safe under FastAPI/async serving):** both `bash` and
`python_repl` run their work in a **separate OS process**, never via in-process
`exec()`. Each call has a timeout and is killed by process group, exposes a true
async path (so concurrent requests don't block the event loop, hold the GIL, or
exhaust the thread pool), and shares no interpreter state between calls. As a
result the Python tool is **not** a persistent REPL — write one self-contained
script per call and persist reusable data to a file. Tuning via env vars:
`BASH_TOOL_TIMEOUT`, `PYTHON_TOOL_TIMEOUT`, `TOOL_MAX_OUTPUT_CHARS`,
`TOOL_MEM_LIMIT_MB`. For fully untrusted code, run behind an OS-level sandbox
(container / gVisor / nsjail) in addition.

## Running

Run from the project root (so `agenticskills`, `skills_ref`, and `deployments`
are importable):

```bash
cd AgenticSkills
nat run --config_file conf/config.yaml --input "<your message>"
```

Redis-memory variant:

```bash
nat run --config_file conf/conf_redis.yaml --input "<your message>"
```

Runtime-free sanity check (no NAT / LLM needed):

```bash
python tests/test_agent.py
```

## Adding a deployment

```python
# deployments/my_agent.py
from agenticskills import build_agent
my_agent = build_agent(skills_subdir="my_skills")   # folder under skills/
```

Then reference `deployments/my_agent.py:my_agent` from a NAT config. Skills
directory resolution order: explicit `skills_dir=` → `env_var=` → `skills_subdir`
under `skills/`.

## Deep-agent variant (deepagents + native sandbox)

Besides the plain LangGraph ReAct agent (`build_agent`), the same skills workflow
can be built with the **deepagents** framework via `build_deep_agent`. This
variant uses deepagents **end to end** — including its own sandbox mechanism —
rather than our custom tools:

- **Native skills** via deepagents' `skills=[...]` (`SkillsMiddleware`), the same
  progressive-disclosure pattern. (`build_agent`'s custom `SkillMiddleware` /
  `skills_ref` is *not* used here.)
- **Native sandboxed execution** via deepagents' built-in **`execute`** tool,
  routed through a `SandboxBackendProtocol` backend — instead of our
  `bash` / `python_repl`.
- **Native file tools** (`read_file` / `write_file` / `edit_file` / `ls` /
  `grep` / `glob`), plus planning (`write_todos`) and sub-agents (`task`).

```bash
pip install deepagents
nat run --config_file conf/config_deep.yaml --input "<your message>"
```

```python
# deployments/anthropic_deep_agent.py
from agenticskills import build_deep_agent
my_agent = build_deep_agent(skills_subdir="anthropic_skills")
```

**Sandbox backend.** By default `build_deep_agent` uses deepagents'
`LocalShellBackend` — the `execute` tool runs on the **local host**. That is
convenient for dev but, per deepagents' own docs, provides **NO isolation**; use
it only with trusted skills/code.

> **Missing-libraries / slow `pip install` every run?** `LocalShellBackend`
> defaults to `inherit_env=False`, which runs `execute` with an EMPTY environment
> — `python`/`pip` then resolve outside your virtualenv, so the shell can't see
> installed packages and the model keeps re-installing them. `build_deep_agent`
> sets **`inherit_env=True`** by default so `execute` uses the SAME venv NAT runs
> in (activate it before `nat serve`). For a permanent fix, **pre-install the
> skills' dependencies into that venv once** (e.g. `pip install openpyxl
> python-docx python-pptx pypdf pillow pandas ...`) so the model never needs to
> install anything. Pass `backend_env=` to add/override specific vars. For real isolation, pass a remote sandbox
backend that implements `SandboxBackendProtocol` (deepagents partners: **Daytona,
Modal, Runloop, Vercel**, or `LangSmithSandbox`); `execute` then runs inside that
sandbox:

```python
from langchain_daytona import DaytonaBackend      # pip install langchain-daytona
my_agent = build_deep_agent(skills_subdir="anthropic_skills", backend=DaytonaBackend(...))
```

### E2B sandbox (`build_e2b_deep_agent`)

For an isolated **E2B** cloud sandbox there's a dedicated factory that also
handles the remote-filesystem detail: since the sandbox is a separate container,
the local `skills/` folder is **uploaded into the sandbox** at build time and
deepagents' `skills=` is pointed at the uploaded path.

```bash
pip install langchain-e2b e2b deepagents
export E2B_API_KEY="..."
nat run --config_file conf/config_e2b.yaml --input "<your message>"
```

```python
# deployments/anthropic_e2b_agent.py
from agenticskills import build_e2b_deep_agent
my_agent = build_e2b_deep_agent(skills_subdir="anthropic_skills")
```

Options: pass your own `sandbox=e2b.Sandbox(...)` to control lifecycle,
`e2b_template=` to use a custom image, `provision_skills=False` if the template
already contains the skills, and `remote_skills_root=` for the upload location.
By default a sandbox is created at build time and killed at interpreter exit.
Note: a default E2B image has Python but not this project's third-party deps —
either let the agent `execute("pip install ...")` or bake them into a template.

`build_deep_agent` accepts the same arguments as `build_agent` (skills dir, MCP
options, memory) plus `subagents=` and `backend=`. Use the deep agent for
multi-step tasks that benefit from planning / sub-agents / real sandboxing; use
the plain `build_agent` for lighter, lower-latency runs.

## MCP servers (optional)

Beyond the `skills/` folder, the agent can use tools from **MCP servers**. Their
tools are loaded via `langchain-mcp-adapters` and appear to the model alongside
`bash` / `python_repl`.

Install the optional dependency, then point a deployment at your servers:

```bash
pip install langchain-mcp-adapters
```

```python
# deployments/anthropic_agent.py
my_agent = build_agent(
    skills_subdir="anthropic_skills",
    mcp_config_path="conf/mcp_servers.json",   # or set MCP_CONFIG_PATH
)
```

Or inline:

```python
my_agent = build_agent(
    skills_subdir="anthropic_skills",
    mcp_servers={
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "transport": "stdio",
        },
    },
)
```

See `conf/mcp_servers.example.json` for the file format (a bare mapping or one
wrapped under `"mcpServers"` both work). With no MCP config, nothing is imported
and behaviour is unchanged.

## Adding a skill

Create `skills/<group>/<skill-name>/SKILL.md` with valid YAML frontmatter
(`name` matching the directory, a descriptive `description`). It is auto-discovered
by `list_skills()` — no registration needed. Validate with:

```python
from skills_ref.validator import validate
from pathlib import Path
print(validate(Path("skills/anthropic_skills/xlsx")))  # [] == valid
```

> **Secrets:** `.env` contains real API keys and is git-ignored. Rotate keys
> before sharing this project.
