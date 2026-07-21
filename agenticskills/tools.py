"""Shared agent tools: a safety-guarded `bash` and an isolated `python_repl`.

Both tools run their work in a **separate OS process**, never via in-process
`exec()`. This matters when the agent is served behind FastAPI/async:

- LLM-generated code cannot block or corrupt the server process. A crash,
  `sys.exit()`, segfault or OOM is contained in the child, not the API worker.
- Every call has a **timeout** and is killed by process group, so a runaway
  `while True` cannot pin a thread forever.
- Both tools expose a true **async** path (`asyncio` subprocess), so serving
  many concurrent requests does not exhaust the thread pool or hold the GIL.
- No shared interpreter state between calls, so concurrent requests cannot leak
  data into each other's namespace.

Trade-off: the Python tool is therefore **not** a persistent REPL — each call is
a fresh process. Write one self-contained script per call and persist anything
you need to reuse to a file (see the tool description).

Tuning knobs (environment variables):
- ``BASH_TOOL_TIMEOUT`` / ``PYTHON_TOOL_TIMEOUT``  seconds per call (default 120)
- ``TOOL_MAX_OUTPUT_CHARS``  cap on returned text (default 20000)
- ``TOOL_MEM_LIMIT_MB``  per-process address-space limit; 0 = disabled (default).
  Note: some libraries (e.g. numpy) reserve large virtual memory, so a tight
  limit can cause spurious MemoryErrors — leave at 0 unless you need it.
"""

import asyncio
import os
import signal
import subprocess
import sys

from langchain_core.tools import StructuredTool

try:  # POSIX-only; used for per-process resource limits
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore

# Command names refused for safety. Checked against EVERY sub-command of a
# chained string, not just the first token — see `_split_shell`.
_BLOCKED_COMMANDS = frozenset(
    {
        "rm", "mv", "cp", "dd", "shutdown", "reboot", "init",
        "systemctl", "service", "kill", "pkill", "chmod", "chown",
    }
)

_BASH_TIMEOUT = int(os.getenv("BASH_TOOL_TIMEOUT", "120"))
_PYTHON_TIMEOUT = int(os.getenv("PYTHON_TOOL_TIMEOUT", "120"))
_MAX_OUTPUT = int(os.getenv("TOOL_MAX_OUTPUT_CHARS", "20000"))
_MEM_LIMIT_MB = int(os.getenv("TOOL_MEM_LIMIT_MB", "0"))  # 0 = disabled


def _split_shell(command: str) -> list[str]:
    """Split a (possibly chained) shell string into its component commands.

    Splits on &&, ||, ; and | so the safety blocklist can be applied to each
    sub-command rather than only the first token of the whole string.
    """
    normalized = (
        command.replace("&&", "\n")
        .replace("||", "\n")
        .replace(";", "\n")
        .replace("|", "\n")
    )
    return [seg.strip() for seg in normalized.split("\n") if seg.strip()]


def _check_blocked(command: str) -> str | None:
    for segment in _split_shell(command):
        first = segment.split()[0] if segment.split() else ""
        if first in _BLOCKED_COMMANDS:
            return f'Command "{first}" is not allowed for safety reasons.'
    return None


def _preexec():
    """Child-side setup: apply memory limit (if configured)."""
    if resource is not None and _MEM_LIMIT_MB > 0:
        nbytes = _MEM_LIMIT_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))


def _format(returncode: int, out: str, err: str) -> str:
    body = out or ""
    if returncode != 0 and err:
        body = f"Error (exit {returncode}): {err}\n{body}"
    body = body or err or ""
    if len(body) > _MAX_OUTPUT:
        body = body[:_MAX_OUTPUT] + f"\n...[output truncated to {_MAX_OUTPUT} chars]"
    return body


def _kill_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run_subprocess(argv: list[str], timeout: int) -> str:
    """Synchronous, isolated run in a new session (killable as a group)."""
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # own process group -> kill children on timeout
        preexec_fn=_preexec if os.name == "posix" else None,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        out, err = proc.communicate()
        return _format(-1, out or "", f"[Killed: exceeded {timeout}s timeout]")
    return _format(proc.returncode, out, err)


async def _arun_subprocess(argv: list[str], timeout: int) -> str:
    """Async, isolated run — does not block the event loop or hold the GIL."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        preexec_fn=_preexec if os.name == "posix" else None,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        _kill_group(proc.pid)
        try:
            await proc.communicate()
        except Exception:
            pass
        return _format(-1, "", f"[Killed: exceeded {timeout}s timeout]")
    return _format(
        proc.returncode or 0,
        out_b.decode(errors="replace"),
        err_b.decode(errors="replace"),
    )


# --------------------------------------------------------------------------- #
# bash
# --------------------------------------------------------------------------- #
_BASH_DESCRIPTION = (
    "Execute one or MORE bash commands and return the combined output.\n"
    "Prefer batching: chain related commands into a SINGLE call to avoid extra "
    "reasoning round-trips. Examples:\n"
    "  - Read several skill docs at once:  `cat SKILL.md REFERENCE.md FORMS.md`\n"
    "  - Chain ordered steps:              `mkdir -p out && python build.py && ls -la out`\n"
    "  - Independent probes together:      `ls skills/ ; cat skills/xlsx/SKILL.md`\n"
    "Use this to read skill files (cat) or run verification scripts (python)."
)


def _bash_sync(command: str) -> str:
    blocked = _check_blocked(command)
    if blocked:
        return blocked
    return _run_subprocess(["bash", "-c", command], _BASH_TIMEOUT)


async def _bash_async(command: str) -> str:
    blocked = _check_blocked(command)
    if blocked:
        return blocked
    return await _arun_subprocess(["bash", "-c", command], _BASH_TIMEOUT)


bash = StructuredTool.from_function(
    func=_bash_sync,
    coroutine=_bash_async,
    name="bash",
    description=_BASH_DESCRIPTION,
)


# --------------------------------------------------------------------------- #
# python_repl (subprocess-isolated; NOT a persistent REPL)
# --------------------------------------------------------------------------- #
_PYTHON_DESCRIPTION = (
    "Run Python code (pandas, openpyxl, etc.) for processing files. Each call "
    "runs in a FRESH, ISOLATED process (same virtualenv, current working "
    "directory) — state does NOT persist between calls.\n"
    "BATCH AGGRESSIVELY — write ONE self-contained script that does as much as "
    "possible: imports, reading inputs, ALL transformations, writing outputs, "
    "and verification prints all in the SAME call. Combine every step whose code "
    "does NOT need the printed result of an earlier step.\n"
    "Only make a SEPARATE call when the next code genuinely needs to see the "
    "previous output. To carry data across calls, write it to a file and read it "
    "back — module-level variables do NOT survive.\n"
    "End the script with print()s of what confirms success."
)


def _python_sync(code: str) -> str:
    return _run_subprocess([sys.executable, "-c", code], _PYTHON_TIMEOUT)


async def _python_async(code: str) -> str:
    return await _arun_subprocess([sys.executable, "-c", code], _PYTHON_TIMEOUT)


python_repl = StructuredTool.from_function(
    func=_python_sync,
    coroutine=_python_async,
    name="python_repl",
    description=_PYTHON_DESCRIPTION,
)


# The canonical tool list shared by every deployment.
TOOLS = [bash, python_repl]
