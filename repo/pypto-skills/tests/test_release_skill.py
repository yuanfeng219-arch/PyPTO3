from __future__ import annotations

import re
import unittest

from tests.skill_assertions import ROOT

SKILL = ROOT / ".claude/skills/release/SKILL.md"
BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def baseline_block() -> str:
    blocks = BASH_BLOCK_RE.findall(SKILL.read_text(encoding="utf-8"))
    return next((block for block in blocks if "BASELINE" in block), "")


class ReleaseSkillTests(unittest.TestCase):
    """The release skill is repository-local and never ships in a plugin."""

    def test_release_skill_is_outside_every_published_plugin(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        for plugin in ("pypto-developer", "pypto-user"):
            with self.subTest(plugin=plugin):
                self.assertFalse(
                    (ROOT / "plugins" / plugin / "skills/release").exists()
                )

    def test_baseline_is_the_commit_that_set_the_current_version(self) -> None:
        block = baseline_block()
        self.assertNotEqual("", block)
        # The last commit that merely touched the manifest is the wrong
        # baseline: a later description edit would hide unreleased changes.
        self.assertNotIn("git log -1 --format=%H --", block)
        self.assertIn(
            "HISTORY=$(git log --format='%H' -- \"$MANIFEST\") || exit 1", block
        )
        self.assertIn('[ "$AT_COMMIT" = "$CURRENT" ] || break', block)

    def test_baseline_walk_fails_closed(self) -> None:
        block = baseline_block()
        self.assertNotEqual("", block)
        # A `while` on the right of a pipe runs in a subshell, so its failure
        # would be masked by the exit status of the last pipeline stage.
        self.assertNotRegex(block, r"\|\s*while\b")
        self.assertNotRegex(block, r"done\s*\|\s*tail")
        self.assertIn("|| exit 1", block)
        self.assertIn('[ -n "$BASELINE" ] || exit 1', block)


if __name__ == "__main__":
    unittest.main()
