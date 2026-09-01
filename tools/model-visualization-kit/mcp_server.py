#!/usr/bin/env python3
"""Minimal dependency-free MCP server for local Codex integration."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ASSET_MANIFEST = ROOT / "assets" / "manifest.json"
ARTIFACTS = ROOT / "artifacts"


def reply(request_id: Any, result: Any = None, error: dict[str, Any] | None = None) -> None:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {"name": "search_visual_assets", "description": "Search visualization assets by keyword, category, or framework.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "category": {"type": "string"}, "framework": {"type": "string"}}}},
        {"name": "render_model_graph", "description": "Render a model JSON into an HTML graph and return a structural summary.", "inputSchema": {"type": "object", "properties": {"model_path": {"type": "string"}, "output_name": {"type": "string"}}, "required": ["model_path"]}},
        {"name": "inspect_operator", "description": "Inspect an operator contract and provide static visualization and validation guidance.", "inputSchema": {"type": "object", "properties": {"operator": {"type": "string"}, "framework": {"type": "string"}, "input_shapes": {"type": "array"}, "dtype": {"type": "string"}}, "required": ["operator"]}},
        {"name": "generate_operator_scaffold", "description": "Generate a reviewable operator development scaffold with tests and visualization metadata.", "inputSchema": {"type": "object", "properties": {"operator_name": {"type": "string"}, "framework": {"type": "string"}, "output_dir": {"type": "string"}}, "required": ["operator_name"]}},
    ]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_user_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def search_assets(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "")).lower()
    category = str(arguments.get("category", "")).lower()
    framework = str(arguments.get("framework", "")).lower()
    assets = load_json(ASSET_MANIFEST)["assets"]
    matches = []
    for asset in assets:
        haystack = json.dumps(asset, ensure_ascii=False).lower()
        if query and query not in haystack:
            continue
        if category and category not in [asset.get("category", "").lower()]:
            continue
        if framework and framework not in [item.lower() for item in asset.get("frameworks", [])]:
            continue
        matches.append(asset)
    return {"count": len(matches), "assets": matches}


def render_model(arguments: dict[str, Any]) -> dict[str, Any]:
    model_path = resolve_user_path(str(arguments["model_path"]))
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")
    model = load_json(model_path)
    nodes = model.get("nodes", [])
    output_name = str(arguments.get("output_name", model_path.stem))
    ARTIFACTS.mkdir(exist_ok=True)
    output = ARTIFACTS / f"{output_name}.html"
    cards = []
    for index, node in enumerate(nodes):
        shape = " × ".join(str(item) for item in node.get("shape", [])) or "unknown"
        cards.append(f'<article class="node"><span class="index">{index + 1:02d}</span><h2>{html.escape(str(node.get("op", "Unknown")))}</h2><p>{html.escape(str(node.get("id", "")))}</p><code>{html.escape(shape)} · {html.escape(str(node.get("dtype", "unknown")))}</code></article>')
    body = "<div class=\"graph\">" + "<div class=\"arrow\">↓</div>".join(cards) + "</div>"
    page = f"""<!doctype html><meta charset='utf-8'><title>{html.escape(str(model.get('name', output_name)))}</title>
<style>body{{font:15px system-ui;background:#10141d;color:#eaf0f6;padding:32px}}.graph{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}.node{{background:#1c2635;border:1px solid #4e6b8c;border-radius:12px;padding:16px;min-width:160px;box-shadow:0 6px 18px #0004}}h2{{margin:0 0 8px;color:#8dd5ff}}p{{color:#9dacbb}}code{{color:#ffdc8a}}.arrow{{color:#78b8d9;font-size:28px}}</style><h1>{html.escape(str(model.get('name', output_name)))}</h1><p>Framework: {html.escape(str(model.get('framework', 'unknown')))} · Nodes: {len(nodes)}</p>{body}"""
    output.write_text(page, encoding="utf-8")
    return {"summary": {"model": model.get("name", output_name), "framework": model.get("framework"), "node_count": len(nodes), "input_count": len(model.get("inputs", []))}, "artifacts": [{"type": "html", "path": str(output)}], "nodes": nodes}


def inspect_operator(arguments: dict[str, Any]) -> dict[str, Any]:
    operator = str(arguments["operator"])
    dtype = str(arguments.get("dtype", "unknown"))
    shapes = arguments.get("input_shapes", [])
    known_risks = []
    if dtype in {"float16", "bfloat16", "bf16", "fp16"}:
        known_risks.append("混合精度算子应检查累加精度、cast 位置和容差。")
    if operator.lower() in {"multiheadattention", "softmax", "layernorm", "rmsnorm"}:
        known_risks.append("该类算子通常包含归约或非线性路径，建议优先查看 Tensor 精度流和边界 shape。")
    return {"operator": operator, "framework": arguments.get("framework", "unknown"), "contract": {"input_shapes": shapes, "dtype": dtype}, "static_inference": {"recommended_assets": ["operator-card", "tensor-contract"], "risks": known_risks or ["需要结合实现源码和真实编译结果进一步确认。"]}, "evidence_boundary": "这是基于输入参数的静态分析，不代表真实设备性能或编译通过。"}


def generate_scaffold(arguments: dict[str, Any]) -> dict[str, Any]:
    name = str(arguments["operator_name"])
    safe_name = "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_") or "custom_operator"
    output_dir = resolve_user_path(str(arguments.get("output_dir", ARTIFACTS / safe_name)))
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"operator.py": f"\"\"{name} scaffold generated by Model Visualization Kit.\"\"\n\ndef {safe_name}(x):\n    raise NotImplementedError(\"Implement {name} after reviewing the contract.\")\n", "test_operator.py": f"def test_{safe_name}_placeholder():\n    # Replace with shape, dtype, boundary, and numerical golden tests.\n    assert True\n", "visualization.json": json.dumps({"operator": name, "assets": ["operator-card", "tensor-contract"], "checks": ["shape", "dtype", "layout", "precision", "boundary"]}, ensure_ascii=False, indent=2)}
    for filename, content in files.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    return {"operator": name, "framework": arguments.get("framework", "unknown"), "output_dir": str(output_dir), "files": [str(output_dir / filename) for filename in files], "next_steps": ["补充输入输出契约", "实现正确性 Golden", "运行最小编译检查", "再进行真实性能 profiling"]}


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "search_visual_assets":
        return search_assets(arguments)
    if name == "render_model_graph":
        return render_model(arguments)
    if name == "inspect_operator":
        return inspect_operator(arguments)
    if name == "generate_operator_scaffold":
        return generate_scaffold(arguments)
    raise ValueError(f"unknown tool: {name}")


def handle(request: dict[str, Any]) -> None:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})
    if method == "initialize":
        reply(request_id, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "model-visualization-kit", "version": "0.1.0"}})
    elif method == "notifications/initialized":
        return
    elif method == "tools/list":
        reply(request_id, {"tools": tool_definitions()})
    elif method == "tools/call":
        try:
            result = call_tool(str(params.get("name")), params.get("arguments", {}))
            reply(request_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}], "structuredContent": result})
        except Exception as exc:  # JSON-RPC boundary: return an actionable tool error.
            reply(request_id, error={"code": -32000, "message": str(exc)})
    else:
        reply(request_id, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            handle(json.loads(line))
        except json.JSONDecodeError as exc:
            reply(None, error={"code": -32700, "message": f"invalid JSON: {exc}"})


if __name__ == "__main__":
    main()
