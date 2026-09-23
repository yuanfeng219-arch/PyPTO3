#!/usr/bin/env python3
"""Compile the two PyPTO controlFlow files and emit LLVM-FLOW-compatible data."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUBS = ROOT / "llvm-flow-api" / "app" / "pypto_stubs"
DEFAULT_SOURCE_DIR = Path("/Users/yin/gitcode/output_deepseek/kernel_aicpu")
DEFAULT_DEV = DEFAULT_SOURCE_DIR / "controlFlow_dev_13801420179101682564.cpp"
DEFAULT_HOST = DEFAULT_SOURCE_DIR / "controlFlow_host_13801420179101682564.cpp"
DEFAULT_OUTPUT = ROOT / "llvm-flow-frontend" / "src" / "data" / "pyptoCfg.json"

LABEL_RE = re.compile(r"^([A-Za-z$._][\w$.-]*|\d+):")
COND_BR_RE = re.compile(r"\bbr i1 .+?, label %([\w$.-]+), label %([\w$.-]+)")
BR_RE = re.compile(r"\bbr label %([\w$.-]+)")


@dataclass
class Block:
    name: str
    lines: list[str]


def debug_locations(ir: str) -> dict[str, int]:
    """Map LLVM metadata ids to real source lines."""
    locations: dict[str, int] = {}
    for metadata_id, line in re.findall(
        r"!(\d+) = !DILocation\(line: (\d+),", ir
    ):
        locations[metadata_id] = int(line)
    return locations


def instruction_opcode(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith(("#dbg_", ";")):
        return None
    if " = " in text:
        text = text.split(" = ", 1)[1]
    tokens = text.split()
    if not tokens:
        return None
    if tokens[0] in {"tail", "musttail", "notail"} and len(tokens) > 1:
        return tokens[1]
    return tokens[0]


def compile_ir(source: Path, output: Path) -> str:
    command = [
        "/usr/bin/clang++",
        "-std=c++17",
        "-Dsection(x)=unused",
        "-O1",
        "-g",
        "-emit-llvm",
        "-S",
        "-I",
        str(STUBS),
        str(source),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output.read_text()


def control_flow_function(ir: str) -> tuple[str, list[str]]:
    lines = ir.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("define ") and "ControlFlowEntry" in line
    )
    header = lines[start]
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line == "}":
            break
        body.append(line)
    return header, body


def parse_blocks(ir: str) -> list[Block]:
    header, body = control_flow_function(ir)
    numeric_args = [int(value) for value in re.findall(r"%([0-9]+)", header)]
    entry_name = str(max(numeric_args, default=-1) + 1)
    blocks = [Block(entry_name, [])]
    for line in body:
        match = LABEL_RE.match(line)
        if match:
            blocks.append(Block(match.group(1), []))
        elif line.strip():
            blocks[-1].lines.append(line.strip())
    return blocks


def graph_json(name: str, blocks: list[Block], ir: str) -> dict:
    index_by_name = {block.name: index for index, block in enumerate(blocks)}
    locations = debug_locations(ir)
    objects = []
    for index, block in enumerate(blocks):
        label_lines = [f"%{block.name}:", *block.lines]
        label = "{" + "\\l".join(label_lines) + "\\l}"
        source_lines = sorted(
            {
                locations[metadata_id]
                for line in block.lines
                for metadata_id in re.findall(r"!dbg !(\d+)", line)
                if metadata_id in locations
            }
        )
        opcodes = [
            opcode
            for line in block.lines
            if (opcode := instruction_opcode(line)) is not None
        ]
        objects.append(
            {
                "_gvid": index,
                "name": f"Node{index}",
                "label": label,
                "shape": "record",
                "metadata": {
                    "block_id": f"%{block.name}",
                    "source_lines": source_lines,
                    "instruction_count": len(opcodes),
                    "opcodes": sorted(set(opcodes)),
                    "terminator": block.lines[-1] if block.lines else "",
                    "predecessors": [],
                    "successors": [],
                },
            }
        )

    edges = []
    for block in blocks:
        if not block.lines:
            continue
        terminator = block.lines[-1]
        conditional = COND_BR_RE.search(terminator)
        targets: list[tuple[str, str | None]]
        if conditional:
            targets = [(conditional.group(1), "T"), (conditional.group(2), "F")]
        else:
            branch = BR_RE.search(terminator)
            targets = [(branch.group(1), None)] if branch else []
        for target, label in targets:
            if target not in index_by_name:
                raise ValueError(f"branch target %{target} has no Basic Block")
            edge = {
                "_gvid": len(edges),
                "tail": index_by_name[block.name],
                "head": index_by_name[target],
            }
            if label:
                edge["label"] = label
            edges.append(edge)
            objects[index_by_name[block.name]]["metadata"]["successors"].append(
                {"block_id": f"%{target}", "label": label}
            )
            objects[index_by_name[target]]["metadata"]["predecessors"].append(
                f"%{block.name}"
            )

    return {
        "name": name,
        "directed": True,
        "strict": False,
        "label": "",
        "_subgraph_cnt": 0,
        "objects": objects,
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--host", type=Path, default=DEFAULT_HOST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="llvm-flow-pypto-") as temp_dir:
        temp = Path(temp_dir)
        dev_ir = compile_ir(args.dev, temp / "dev.ll")
        host_ir = compile_ir(args.host, temp / "host.ll")

    dev_blocks = parse_blocks(dev_ir)
    host_blocks = parse_blocks(host_ir)
    dev_graph = graph_json("PyPTO ControlFlowEntry — dev", dev_blocks, dev_ir)
    host_graph = graph_json("PyPTO ControlFlowEntry — host", host_blocks, host_ir)

    # The two supplied generated sources are structurally identical. Pairing by
    # real LLVM block label therefore gives exact matches while retaining the
    # same parallel arrays consumed by LLVM-FLOW's llvm-block UI.
    host_names = {block.name for block in host_blocks}
    matched = [f"%{block.name}" for block in dev_blocks if block.name in host_names]
    payload = {
        "before_json": dev_graph,
        "before_output": matched,
        "after_json": host_graph,
        "after_output": matched,
        "beforeg_data": dev_ir,
        "afterg_data": host_ir,
        "source_data": {
            "dev": args.dev.read_text(),
            "host": args.host.read_text(),
            "dev_path": str(args.dev),
            "host_path": str(args.host),
        },
        "file_pass": "Apple Clang -O1 · real LLVM IR · dev vs host",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False))

    conditional_edges = sum(edge.get("label") == "T" for edge in dev_graph["edges"])
    located_blocks = sum(
        bool(node["metadata"]["source_lines"]) for node in dev_graph["objects"]
    )
    print(
        f"generated {args.output}: "
        f"dev={len(dev_blocks)} blocks/{len(dev_graph['edges'])} edges, "
        f"host={len(host_blocks)} blocks/{len(host_graph['edges'])} edges, "
        f"conditional branches={conditional_edges}, "
        f"source-located={located_blocks}/{len(dev_blocks)}"
    )


if __name__ == "__main__":
    main()
