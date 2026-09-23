from __future__ import annotations

import unittest

from tests.skill_assertions import ROOT


class CIToolConfigurationTests(unittest.TestCase):
    def test_ci_dependencies_are_exactly_pinned(self) -> None:
        requirements = (ROOT / "requirements-ci.txt").read_text(encoding="utf-8")
        self.assertEqual(
            ["pyright==1.1.410", "ruff==0.16.0"],
            requirements.splitlines(),
        )

    def test_python_tools_target_the_supported_baseline(self) -> None:
        configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('target-version = "py310"', configuration)
        self.assertIn('select = ["E4", "E7", "E9", "F", "I"]', configuration)
        self.assertIn('pythonVersion = "3.10"', configuration)
        self.assertIn('typeCheckingMode = "basic"', configuration)
        self.assertIn('include = ["tests"]', configuration)


class CIWorkflowConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    def test_workflow_has_required_triggers_and_read_only_permission(self) -> None:
        for required in (
            "pull_request:",
            "push:",
            "branches: [main]",
            "workflow_dispatch:",
            "permissions:",
            "contents: read",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.workflow)

    def test_actions_are_pinned_and_credentials_are_not_persisted(self) -> None:
        self.assertIn(
            "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd",
            self.workflow,
        )
        self.assertIn(
            "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405",
            self.workflow,
        )
        self.assertEqual(2, self.workflow.count("persist-credentials: false"))

    def test_runtime_uses_supported_baseline_without_isolation_runtime(self) -> None:
        for required in (
            "name: Tests (Python 3.10)",
            'python-version: "3.10"',
            "python -m unittest discover -s tests -v",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.workflow)
        for absent in ("bubblewrap", "PYPTO_SKILLS_REQUIRE_BWRAP"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, self.workflow)
        self.assertNotIn("matrix:", self.workflow)
        self.assertNotIn('python-version: "3.14"', self.workflow)

    def test_workflow_runs_all_quality_checks(self) -> None:
        for required in (
            "ruff check tests",
            "ruff format --check tests",
            "pyright",
            "git ls-files -z -- '*.sh'",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.workflow)
        self.assertNotIn("continue-on-error", self.workflow)


if __name__ == "__main__":
    unittest.main()
