from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.skill_assertions import ROOT

HELPER = ROOT / "lib/repository/scripts/stage-owned.sh"
POLICY = ROOT / "lib/repository/policy.md"
SKILL = ROOT / "skills/git-commit/SKILL.md"


def run(
    *args: str | Path,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


class GitCommitFileTests(unittest.TestCase):
    def test_repository_policy_defines_precedence_and_stop_rules(self) -> None:
        self.assertTrue(POLICY.is_file(), f"missing repository policy: {POLICY}")
        if not POLICY.is_file():
            return

        text = POLICY.read_text(encoding="utf-8")
        self.assertIn("REPO_ROOT=$(git rev-parse --show-toplevel)", text)
        precedence = (
            "User instructions",
            "Applicable repository instructions",
            "Documented workflow and configuration",
            "Unambiguous local history",
        )
        positions = [text.find(level) for level in precedence]
        self.assertTrue(all(position >= 0 for position in positions))
        self.assertEqual(positions, sorted(positions))
        self.assertIn("every changed path", text)
        self.assertIn("nested instruction", text)
        self.assertIn("Stop and ask the user", text)
        self.assertIn("Never replace missing policy", text)

    def test_git_commit_skill_uses_portable_safe_workflow(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing git-commit skill: {SKILL}")
        if not SKILL.is_file():
            return

        text = SKILL.read_text(encoding="utf-8")
        self.assertIn("../../lib/repository/policy.md", text)
        self.assertIn("../../lib/repository/scripts/stage-owned.sh", text)
        self.assertIn("every changed path", text)
        self.assertIn("Never use broad staging", text)
        self.assertIn("repository-selected verification", text)
        self.assertIn("repository policy and unambiguous history", text)
        self.assertIn("repository-root-relative paths", text)
        self.assertIn("verification results", text.lower())
        self.assertIn("complete commit message", text)
        self.assertIn("git show", text)

        verification = text.find("## Run repository-selected verification")
        commit = text.find("## Commit and verify")
        self.assertGreaterEqual(verification, 0)
        self.assertGreater(commit, verification)
        self.assertLessEqual(len(text.splitlines()), 200)


class StageOwnedBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        if not HELPER.is_file():
            self.skipTest(f"missing staging helper: {HELPER}")

        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.work = Path(temporary_directory.name) / "work"
        run("git", "init", "--initial-branch=trunk", self.work, cwd=Path("/tmp"))
        self.git("config", "user.email", "portable@example.com")
        self.git("config", "user.name", "Portable Tests")

        (self.work / "changed.txt").write_text("before\n", encoding="utf-8")
        (self.work / "notes.txt").write_text("before\n", encoding="utf-8")
        self.git("add", "changed.txt", "notes.txt")
        self.git("commit", "-m", "initial")

        (self.work / "changed.txt").write_text("after\n", encoding="utf-8")
        (self.work / "notes.txt").write_text("user edit\n", encoding="utf-8")
        (self.work / "scratch.txt").write_text("scratch\n", encoding="utf-8")

    def git(
        self,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return run("git", *args, cwd=self.work, check=check)

    def helper(
        self,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return run(HELPER, *args, cwd=self.work, check=check)

    def test_stage_helper_stages_only_explicit_owned_paths(self) -> None:
        result = self.helper("changed.txt")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            "changed.txt",
            self.git("diff", "--cached", "--name-only").stdout.strip(),
        )
        self.assertIn("notes.txt", self.git("diff", "--name-only").stdout.splitlines())
        self.assertIn(
            "?? scratch.txt", self.git("status", "--porcelain").stdout.splitlines()
        )

    def test_stage_helper_accepts_an_explicit_directory_symlink(self) -> None:
        target = self.work / "owned-target"
        target.mkdir()
        (target / "content.txt").write_text("target\n", encoding="utf-8")
        (self.work / "owned-link").symlink_to("owned-target", target_is_directory=True)

        result = self.helper("owned-link")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            "120000",
            self.git("ls-files", "--stage", "owned-link").stdout.split()[0],
        )

    def test_stage_helper_still_rejects_a_real_directory(self) -> None:
        (self.work / "owned-directory").mkdir()
        (self.work / "owned-directory" / "content.txt").write_text(
            "content\n", encoding="utf-8"
        )

        result = self.helper("owned-directory", check=False)

        self.assertEqual(2, result.returncode)
        self.assertIn("directory path is not allowed", result.stderr)

    def test_stage_helper_accepts_an_explicit_tracked_submodule(self) -> None:
        source = self.work.parent / "submodule-source"
        run("git", "init", "--initial-branch=trunk", source, cwd=self.work.parent)
        run("git", "config", "user.email", "portable@example.com", cwd=source)
        run("git", "config", "user.name", "Portable Tests", cwd=source)
        (source / "version.txt").write_text("one\n", encoding="utf-8")
        run("git", "add", "version.txt", cwd=source)
        run("git", "commit", "-m", "initial", cwd=source)

        self.git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(source),
            "vendor",
        )
        self.git("commit", "-m", "add submodule")

        (source / "version.txt").write_text("two\n", encoding="utf-8")
        run("git", "commit", "-am", "update", cwd=source)
        run("git", "fetch", cwd=self.work / "vendor")
        run("git", "checkout", "FETCH_HEAD", cwd=self.work / "vendor")

        result = self.helper("vendor")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            run("git", "rev-parse", "HEAD", cwd=source).stdout.strip(),
            self.git("rev-parse", ":vendor").stdout.strip(),
        )

    def test_stage_helper_rejects_an_unrelated_pre_staged_path(self) -> None:
        self.git("add", "notes.txt")
        result = self.helper("changed.txt", check=False)
        self.assertNotEqual(0, result.returncode)
        self.assertIn(
            "already-staged path is outside the authorized set", result.stderr
        )

    def test_stage_helper_uses_documented_failure_codes(self) -> None:
        invalid_paths = (
            (),
            ("/absolute.txt",),
            ("../outside.txt",),
            (".",),
            (":(glob)*",),
        )
        for arguments in invalid_paths:
            with self.subTest(arguments=arguments):
                self.assertEqual(2, self.helper(*arguments, check=False).returncode)

        self.assertEqual(
            4,
            self.helper("missing.txt", check=False).returncode,
        )

    def test_stage_helper_rejects_short_form_pathspec_magic_before_staging(
        self,
    ) -> None:
        for unsafe_pathspec in (":/", ":!notes.txt", ":^notes.txt", ":"):
            with self.subTest(pathspec=unsafe_pathspec):
                result = self.helper(unsafe_pathspec, check=False)
                staged_names = self.git("diff", "--cached", "--name-only").stdout
                self.git("reset", "--quiet", "HEAD")

                self.assertEqual(2, result.returncode, result.stderr)
                self.assertEqual("", staged_names)


if __name__ == "__main__":
    unittest.main()
