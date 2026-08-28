from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from tests.skill_assertions import DEVELOPER_PLUGIN, ROOT, USER_PLUGIN

PLUGIN_NAMES = ("pypto-developer", "pypto-user")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class PluginStructureTests(unittest.TestCase):
    def test_dual_manifests_match_plugin_identity_and_version(self) -> None:
        for name, plugin in (
            ("pypto-developer", DEVELOPER_PLUGIN),
            ("pypto-user", USER_PLUGIN),
        ):
            with self.subTest(plugin=name):
                codex = load_json(plugin / ".codex-plugin/plugin.json")
                claude = load_json(plugin / ".claude-plugin/plugin.json")
                self.assertEqual(name, codex["name"])
                self.assertEqual(name, claude["name"])
                self.assertEqual(codex["version"], claude["version"])
                self.assertEqual("./skills/", codex["skills"])
                self.assertTrue((plugin / "skills").is_dir())

    def test_marketplaces_publish_the_same_plugins(self) -> None:
        codex = load_json(ROOT / ".agents/plugins/marketplace.json")
        claude = load_json(ROOT / ".claude-plugin/marketplace.json")

        self.assertEqual("pypto-skills", codex["name"])
        self.assertEqual("pypto-skills", claude["name"])
        self.assertEqual(
            set(PLUGIN_NAMES),
            {entry["name"] for entry in codex["plugins"]},
        )
        self.assertEqual(
            set(PLUGIN_NAMES),
            {entry["name"] for entry in claude["plugins"]},
        )

    def test_marketplace_sources_are_plugin_local(self) -> None:
        codex = load_json(ROOT / ".agents/plugins/marketplace.json")
        claude = load_json(ROOT / ".claude-plugin/marketplace.json")

        for entry in codex["plugins"]:
            source = entry["source"]
            self.assertEqual("local", source["source"])
            self.assertEqual(f"./plugins/{entry['name']}", source["path"])

        for entry in claude["plugins"]:
            self.assertEqual(f"./plugins/{entry['name']}", entry["source"])

    def test_legacy_bundle_paths_are_compatibility_symlinks(self) -> None:
        self.assertEqual(
            Path("plugins/pypto-developer/skills"),
            (ROOT / "skills").readlink(),
        )
        self.assertEqual(
            Path("plugins/pypto-developer/lib"),
            (ROOT / "lib").readlink(),
        )


if __name__ == "__main__":
    unittest.main()
