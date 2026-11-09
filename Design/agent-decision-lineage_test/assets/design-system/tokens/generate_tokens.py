#!/usr/bin/env python3
"""Generate token snapshots from the CSS token source files."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


TOKEN_DIR = Path(__file__).resolve().parent
SOURCES = [
    ("foundation", TOKEN_DIR / "foundation.css"),
    ("semantic", TOKEN_DIR / "semantic.css"),
    ("components", TOKEN_DIR / "components.css"),
]


BLOCK_RE = re.compile(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", re.S)
VAR_RE = re.compile(r"--(?P<name>[\w-]+)\s*:\s*(?P<value>.*?);", re.S)


def block_group(selectors: str) -> str | None:
    selectors = selectors.strip()
    if ":root" not in selectors:
        return None
    if "data-theme='light'" in selectors or 'data-theme="light"' in selectors:
        return "light"
    if "data-theme='glass'" in selectors or 'data-theme="glass"' in selectors:
        return "glass"
    if "data-theme='dark'" in selectors or 'data-theme="dark"' in selectors:
        return "dark"
    return "base"


def parse_css_tokens(path: Path) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    css = path.read_text(encoding="utf-8")
    for match in BLOCK_RE.finditer(css):
        group = block_group(match.group("selectors"))
        if not group:
            continue
        values = groups.setdefault(group, {})
        for var_match in VAR_RE.finditer(match.group("body")):
            value = " ".join(var_match.group("value").strip().split())
            values[var_match.group("name")] = value
    return groups


def merge_theme(source: dict[str, dict[str, dict[str, str]]], mode: str) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source_name, _path in SOURCES:
        groups = source[source_name]
        merged.update(groups.get("base", {}))
        merged.update(groups.get("dark", {}))
        if mode == "light":
            merged.update(groups.get("light", {}))
        if mode == "glass":
            merged.update(groups.get("glass", {}))
    return merged


def main() -> None:
    source = {name: parse_css_tokens(path) for name, path in SOURCES}
    snapshot = {
        "$meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "generator": "tokens/generate_tokens.py",
            "sourceOfTruth": [f"tokens/{path.name}" for _name, path in SOURCES],
            "editRule": "Do not edit tokens.json or tokens.js by hand. Regenerate from CSS token sources.",
        },
        "source": source,
        "themes": {
            "dark": merge_theme(source, "dark"),
            "light": merge_theme(source, "light"),
            "glass": merge_theme(source, "glass"),
        },
    }

    json_path = TOKEN_DIR / "tokens.json"
    js_path = TOKEN_DIR / "tokens.js"
    json_text = json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    js_path.write_text(
        "const PTO_DESIGN_TOKENS = "
        + json.dumps(snapshot, ensure_ascii=False, indent=2)
        + ";\n\n"
        + "if (typeof window !== 'undefined') window.PTO_DESIGN_TOKENS = PTO_DESIGN_TOKENS;\n"
        + "if (typeof module !== 'undefined') module.exports = PTO_DESIGN_TOKENS;\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path.relative_to(TOKEN_DIR.parent)}")
    print(f"Wrote {js_path.relative_to(TOKEN_DIR.parent)}")


if __name__ == "__main__":
    main()
