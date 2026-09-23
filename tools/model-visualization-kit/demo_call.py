"""Protocol smoke test for the local MCP server."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "mcp_server.py"


def request(request_id: int, method: str, params: dict | None = None) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}, ensure_ascii=False)


def main() -> None:
    payload = "\n".join([
        request(1, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "demo", "version": "0.1"}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        request(2, "tools/list"),
        request(3, "tools/call", {"name": "render_model_graph", "arguments": {"model_path": str(ROOT / "examples" / "resnet50.json")}}),
    ]) + "\n"
    completed = subprocess.run([sys.executable, str(SERVER)], input=payload, text=True, capture_output=True, check=True)
    responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert len(responses) == 3, responses
    assert len(responses[1]["result"]["tools"]) == 4
    artifact = responses[2]["result"]["structuredContent"]["artifacts"][0]["path"]
    assert Path(artifact).exists(), artifact
    print(json.dumps({"ok": True, "tool_count": 4, "artifact": artifact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
