from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGINS = ROOT / "plugins"
DEVELOPER_PLUGIN = PLUGINS / "pypto-developer"
USER_PLUGIN = PLUGINS / "pypto-user"
SKILLS = DEVELOPER_PLUGIN / "skills"
USER_SKILLS = USER_PLUGIN / "skills"
LINK_RE = re.compile(r"(?<!!)\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)")


def skill_dirs(skills_root: Path = SKILLS) -> list[Path]:
    if not skills_root.is_dir():
        return []
    return sorted(path for path in skills_root.iterdir() if path.is_dir())


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise AssertionError(f"{path} has no YAML frontmatter")
    block = text.split("---\n", 2)[1]
    result: dict[str, str] = {}
    for line in block.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            raise AssertionError(f"{path} has invalid frontmatter line: {line}")
        result[key.strip()] = value.strip()
    return result


def markdown_links(path: Path) -> list[Path]:
    links = []
    for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
        if "://" not in target and not target.startswith("mailto:"):
            links.append((path.parent / target).resolve())
    return links
