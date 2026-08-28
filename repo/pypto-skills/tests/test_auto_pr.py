from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.skill_assertions import ROOT

HELPER = ROOT / "skills/auto-pr/scripts/auto-pr-loop.sh"
SKILL = ROOT / "skills/auto-pr/SKILL.md"


class AutoPrLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.temp_path = Path(temporary_directory.name)
        self.ledger = self.temp_path / "attempts.tsv"
        self.ledger.write_text("", encoding="utf-8")

    def run_helper(
        self, *arguments: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        self.assertTrue(HELPER.is_file(), f"missing loop helper: {HELPER}")
        result = subprocess.run(
            [str(HELPER), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            self.fail(f"auto-pr-loop failed:\n{result.stdout}{result.stderr}")
        return result

    def classify(self, kind: str) -> subprocess.CompletedProcess[str]:
        return self.run_helper("classify", kind)

    def guard(
        self,
        iteration: int,
        finding_key: str,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self.run_helper(
            "guard",
            str(iteration),
            finding_key,
            str(self.ledger),
            check=check,
        )

    def test_objective_ci_is_fixed_and_architecture_is_deferred(self) -> None:
        self.assertEqual("fix", self.classify("ci-objective").stdout.strip())
        self.assertEqual("fix", self.classify("correctness").stdout.strip())
        self.assertEqual("fix", self.classify("style-policy").stdout.strip())
        self.assertEqual("defer", self.classify("architecture").stdout.strip())
        self.assertEqual("defer", self.classify("product").stdout.strip())
        self.assertEqual("defer", self.classify("judgment").stdout.strip())

    def test_informational_is_ignored_and_unknown_is_deferred(self) -> None:
        self.assertEqual("ignore", self.classify("informational").stdout.strip())
        self.assertEqual("ignore", self.classify("resolved").stdout.strip())
        self.assertEqual("defer", self.classify("unknown-kind").stdout.strip())

    def test_same_finding_stops_before_a_third_attempt(self) -> None:
        self.assertEqual(0, self.guard(1, "unit-tests:stale-value").returncode)
        self.assertEqual(0, self.guard(2, "unit-tests:stale-value").returncode)
        result = self.guard(3, "unit-tests:stale-value", check=False)
        self.assertEqual(20, result.returncode)
        self.assertIn("attempted twice", result.stderr)
        self.assertEqual(
            "unit-tests:stale-value\t1\nunit-tests:stale-value\t2\n",
            self.ledger.read_text(encoding="utf-8"),
        )
        self.assertFalse(Path(f"{self.ledger}.lock").exists())

    def test_concurrent_guards_serialize_and_clean_the_lock(self) -> None:
        processes = [
            subprocess.Popen(
                [str(HELPER), "guard", "1", "same-finding", str(self.ledger)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(4)
        ]
        results = [
            process.communicate() + (process.returncode,) for process in processes
        ]

        self.assertEqual([0, 0, 20, 20], sorted(result[2] for result in results))
        self.assertEqual(
            "same-finding\t1\nsame-finding\t2\n",
            self.ledger.read_text(encoding="utf-8"),
        )
        self.assertFalse(Path(f"{self.ledger}.lock").exists())

    def test_ninth_iteration_is_rejected_without_ledger_change(self) -> None:
        before = self.ledger.read_text(encoding="utf-8")
        result = self.guard(9, "new-finding", check=False)
        self.assertEqual(21, result.returncode)
        self.assertEqual(before, self.ledger.read_text(encoding="utf-8"))

    def test_invalid_iteration_and_keys_leave_ledger_unchanged(self) -> None:
        for iteration in ("0", "9", "eight", "1.0", "9" * 100):
            with self.subTest(iteration=iteration):
                before = self.ledger.read_text(encoding="utf-8")
                result = self.run_helper(
                    "guard",
                    iteration,
                    "new-finding",
                    str(self.ledger),
                    check=False,
                )
                self.assertEqual(21, result.returncode)
                self.assertEqual(before, self.ledger.read_text(encoding="utf-8"))

        for finding_key in ("", "line\nbreak", "tab\tbreak"):
            with self.subTest(finding_key=repr(finding_key)):
                before = self.ledger.read_text(encoding="utf-8")
                result = self.guard(1, finding_key, check=False)
                self.assertNotEqual(0, result.returncode)
                self.assertEqual(before, self.ledger.read_text(encoding="utf-8"))


class AutoPrCompositionTests(unittest.TestCase):
    def test_skill_composes_existing_workflows_and_repository_policy(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        for link in (
            "../git-commit/SKILL.md",
            "../github-pr/SKILL.md",
            "../fix-pr/SKILL.md",
            "../../lib/repository/policy.md",
        ):
            with self.subTest(link=link):
                self.assertIn(link, text)

    def test_skill_bounds_and_orders_each_repair_iteration(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        scope = text.find("current pull request")
        classify = text.find("Classify before editing")
        repair = text.find("one repair iteration")
        verification = text.find("Rerun repository-required verification")
        self.assertGreaterEqual(scope, 0)
        self.assertGreater(classify, scope)
        self.assertGreater(repair, classify)
        self.assertGreater(verification, repair)
        self.assertIn("at most twice", text)
        self.assertIn("at most eight iterations", text)
        self.assertRegex(text, r"Leave every deferred thread\s+unresolved")

    def test_skill_never_invents_missing_pull_request_identity(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        self.assertRegex(text, r"Never\s+invent a missing identity field")
        self.assertIn("unknown identity", text)

    def test_missing_identity_is_terminal_before_orchestration(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        terminal = text.find("Missing PR identity is terminal in every mode")
        ledger = text.find("Create a fresh task-private attempt ledger")
        orchestration = text.find("## Orchestrate the bounded loop")
        self.assertGreaterEqual(terminal, 0)
        self.assertGreater(ledger, terminal)
        self.assertGreater(orchestration, ledger)
        self.assertRegex(
            text,
            r"must not enter classification,\s+guard, or repair orchestration",
        )

    def test_publication_delegates_route_validation_to_github_pr(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        publish_start = text.find("## Publish and bind one PR")
        publish_end = text.find("## Orchestrate the bounded loop")
        self.assertGreaterEqual(publish_start, 0)
        self.assertGreater(publish_end, publish_start)
        publish = text[publish_start:publish_end]
        self.assertIn("Delegate publication directly to `github-pr`", publish)
        self.assertRegex(publish, r"Do not invoke `git-commit`\s+directly")
        self.assertNotRegex(publish, r"Delegate[^\n]*to `git-commit`")

    def test_auto_pr_supplies_narrow_composed_authorization(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        for evidence in (
            "exact validated current-PR identity",
            "unchanged numbered inventory entry",
            "stable finding ID",
            "normalized allowed kind",
            "successful guard iteration and attempt evidence",
            "standing authorization from the explicit `auto-pr` invocation",
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, text)
        self.assertIn("for only those findings", text)

    def test_publication_is_unattended_and_reports_what_it_derived(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        publish = text[
            text.find("## Publish and bind one PR") : text.find(
                "## Orchestrate the bounded loop"
            )
        ]
        self.assertIn("Publication runs unattended", publish)
        self.assertRegex(publish, r"Require a reviewer-ready description")
        self.assertRegex(publish, r"recompose it once")
        self.assertRegex(
            text,
            r"the head branch and commit subjects\s+`github-pr` published",
        )

    def test_deferred_judgment_takes_precedence_over_green_checks(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        self.assertIn("Deferred judgment takes precedence over green success", text)
        self.assertIn("even when every required check is green", text)

    def test_skill_does_not_copy_composed_mechanics(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing required skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        for copied_mechanic in (
            "gh pr create",
            "resolveReviewThread",
            "git push",
            "reviewThreads",
            "fetch-comments.md",
            "reply-and-resolve.md",
        ):
            with self.subTest(copied_mechanic=copied_mechanic):
                self.assertNotIn(copied_mechanic, text)


if __name__ == "__main__":
    unittest.main()
