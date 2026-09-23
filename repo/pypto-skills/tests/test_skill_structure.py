from __future__ import annotations

import unittest

from tests.skill_assertions import (
    SKILLS,
    USER_SKILLS,
    frontmatter,
    markdown_links,
    skill_dirs,
)

EXPECTED_SKILLS: tuple[str, ...] = (
    "auto-pr",
    "clean-branches",
    "create-issue",
    "fix-issue",
    "fix-pr",
    "git-commit",
    "github-pr",
)

EXPECTED_USER_SKILLS: tuple[str, ...] = (
    "critical-path-analysis",
    "dependency-redundancy",
    "generate-ir-trace",
    "incore-profiling",
    "setup-and-run",
)


class SkillStructureTests(unittest.TestCase):
    def test_expected_skill_directories_exist(self) -> None:
        available = {path.name for path in skill_dirs()}

        for name in EXPECTED_SKILLS:
            with self.subTest(skill=name):
                self.assertIn(name, available)

    def test_expected_skills_have_skill_markdown(self) -> None:
        for name in EXPECTED_SKILLS:
            with self.subTest(skill=name):
                self.assertTrue((SKILLS / name / "SKILL.md").is_file())

    def test_expected_skills_have_valid_frontmatter(self) -> None:
        for name in EXPECTED_SKILLS:
            with self.subTest(skill=name):
                skill_markdown = SKILLS / name / "SKILL.md"
                self.assertTrue(skill_markdown.is_file())
                metadata = frontmatter(skill_markdown)
                self.assertEqual({"name", "description"}, set(metadata))
                self.assertEqual(name, metadata["name"])
                self.assertTrue(metadata["description"].startswith("Use when"))

    def test_expected_skills_have_openai_metadata(self) -> None:
        for name in EXPECTED_SKILLS:
            with self.subTest(skill=name):
                self.assertTrue((SKILLS / name / "agents" / "openai.yaml").is_file())

    def test_expected_skills_have_resolvable_local_markdown_links(self) -> None:
        for name in EXPECTED_SKILLS:
            with self.subTest(skill=name):
                skill_markdown = SKILLS / name / "SKILL.md"
                self.assertTrue(skill_markdown.is_file())
                for target in markdown_links(skill_markdown):
                    self.assertTrue(target.exists())

    def test_expected_user_skill_directories_exist(self) -> None:
        available = {path.name for path in skill_dirs(USER_SKILLS)}

        for name in EXPECTED_USER_SKILLS:
            with self.subTest(skill=name):
                self.assertIn(name, available)

    def test_expected_user_skills_have_valid_structure(self) -> None:
        for name in EXPECTED_USER_SKILLS:
            with self.subTest(skill=name):
                skill_markdown = USER_SKILLS / name / "SKILL.md"
                self.assertTrue(skill_markdown.is_file())
                metadata = frontmatter(skill_markdown)
                self.assertEqual({"name", "description"}, set(metadata))
                self.assertEqual(name, metadata["name"])
                self.assertTrue(metadata["description"].startswith("Use when"))
                self.assertTrue(
                    (USER_SKILLS / name / "agents" / "openai.yaml").is_file()
                )
                for target in markdown_links(skill_markdown):
                    self.assertTrue(target.exists())

    def test_all_skills_bound_local_system_test_scope(self) -> None:
        skill_markdowns = [SKILLS / name / "SKILL.md" for name in EXPECTED_SKILLS]
        skill_markdowns.extend(
            USER_SKILLS / name / "SKILL.md" for name in EXPECTED_USER_SKILLS
        )

        for skill_markdown in skill_markdowns:
            with self.subTest(skill=skill_markdown.parent.name):
                text = skill_markdown.read_text(encoding="utf-8")
                self.assertIn("Never run the full system-test suite locally", text)
                self.assertIn(
                    "directly relevant to the changed or requested scope", text
                )
                self.assertIn("use CI for the full", text)


if __name__ == "__main__":
    unittest.main()
