"""Optional MCP (Model Context Protocol) tool loading.

Lets the agent use tools exposed by one or more MCP servers *in addition to* the
built-in `bash` / `python_repl` and the `skills/` folder. Uses
`langchain-mcp-adapters` to turn MCP tools into LangChain tools that plug
straight into `create_agent`.

MCP support is fully optional:
- No servers configured  -> nothing is imported, behaviour is unchanged.
- Servers configured but `langchain-mcp-adapters` not installed -> a clear
  ImportError telling you what to install.

Server config format (passed to MultiServerMCPClient), e.g.:

    {
      "math":    {"command": "python", "args": ["math_server.py"], "transport": "stdio"},
      "weather": {"url": "http://localhost:8000/mcp/", "transport": "streamable_http"}
    }

It can be given inline, or in a JSON file whose top level is that mapping or is
wrapped under an "mcpServers" key (the common convention).
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Any


def _run_coro(coro):
    """Run a coroutine to completion whether or not a loop is already running.

    `build_agent` may be imported by NAT either from a plain sync context or from
    within a running event loop; both must work.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no loop running: safe to drive one

    # A loop is already running in this thread — run the coroutine in a separate
    # thread with its own loop and wait for it.
    box: dict[str, Any] = {}

    def _runner():
        box["value"] = asyncio.run(coro)

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    return box["value"]


def load_mcp_config(config_path: Path | str) -> dict[str, Any]:
    """Load an MCP server mapping from a JSON file.

    Accepts either a bare `{server_name: {...}}` mapping or one wrapped as
    `{"mcpServers": {server_name: {...}}}`.
    """
    data = json.loads(Path(config_path).read_text())
    if isinstance(data, dict) and "mcpServers" in data:
        return data["mcpServers"]
    return data


def load_mcp_tools(servers: dict[str, Any] | None) -> list:
    """Return LangChain tools for the given MCP server mapping (or [] if none)."""
    if not servers:
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "MCP servers were configured but 'langchain-mcp-adapters' is not "
            "installed. Install it with: pip install langchain-mcp-adapters"
        ) from exc

    client = MultiServerMCPClient(servers)
    return _run_coro(client.get_tools())
