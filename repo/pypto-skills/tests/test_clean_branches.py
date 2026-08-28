from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.skill_assertions import ROOT

HELPER = ROOT / "skills/clean-branches/scripts/clean-branches.sh"


def run(
    *args: str | Path,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


class CleanBranchesHelperPresenceTests(unittest.TestCase):
    def test_behavioral_helper_exists(self) -> None:
        self.assertTrue(HELPER.is_file(), f"missing behavioral helper: {HELPER}")


class CleanBranchesBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        if not HELPER.is_file():
            self.skipTest(f"missing behavioral helper: {HELPER}")

        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.upstream = self.root / "upstream.git"
        self.origin = self.root / "origin.git"
        self.work = self.root / "work"

        run(
            "git",
            "init",
            "--bare",
            "--initial-branch=trunk",
            self.upstream,
        )
        run(
            "git",
            "init",
            "--bare",
            "--initial-branch=trunk",
            self.origin,
        )
        run("git", "init", "--initial-branch=trunk", self.work)
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Clean Branches Test")
        self.git("commit", "--allow-empty", "-m", "root")
        self.git("remote", "add", "upstream", self.upstream)
        self.git("remote", "add", "origin", self.origin)

        self.git("checkout", "-b", "regular")
        self.git("commit", "--allow-empty", "-m", "regular")
        self.regular_oid = self.rev_parse("regular")
        self.git("checkout", "trunk")
        self.git("merge", "--no-ff", "regular", "-m", "merge regular")

        self.git("checkout", "-b", "squash-exact")
        self.git("commit", "--allow-empty", "-m", "squash")
        self.squash_oid = self.rev_parse("squash-exact")
        self.git("checkout", "trunk")
        self.git("merge", "--squash", "squash-exact")
        self.git("commit", "--allow-empty", "-m", "squash merge")

        self.git("branch", "squash-reused", self.squash_oid)
        self.git("checkout", "squash-reused")
        self.git("commit", "--allow-empty", "-m", "post-merge reuse")

        for branch in (
            "base-only",
            "linked-worktree",
            "local-atomic",
            "local-safe",
            "local-race",
            "remote-safe",
            "remote-race",
            "remote-atomic",
        ):
            self.git("branch", branch, self.regular_oid)

        self.git("checkout", "-b", "feature/current", "trunk")
        self.git("commit", "--allow-empty", "-m", "current work")

        self.git("push", "upstream", "trunk", "base-only")
        self.git(
            "push",
            "origin",
            "trunk",
            "regular",
            "squash-exact",
            "squash-reused",
            "feature/current",
            "remote-safe",
            "remote-race",
            "remote-atomic",
        )

    def git(
        self,
        *args: str | Path,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return run("git", *args, cwd=self.work, check=check)

    def helper(
        self,
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return run(
            HELPER,
            *args,
            cwd=self.work,
            check=check,
            env=env,
        )

    def rev_parse(self, ref: str) -> str:
        return self.git("rev-parse", ref).stdout.strip()

    def bare_ref(self, repository: Path, ref: str) -> str:
        return run(
            "git",
            f"--git-dir={repository}",
            "rev-parse",
            ref,
        ).stdout.strip()

    def assert_ref_missing(self, repository: Path, ref: str) -> None:
        result = run(
            "git",
            f"--git-dir={repository}",
            "show-ref",
            "--verify",
            "--quiet",
            ref,
            check=False,
        )
        self.assertNotEqual(0, result.returncode)

    def test_classifies_dynamic_trunk_and_branch_reuse(self) -> None:
        cases = (
            (
                (
                    "classify",
                    "trunk",
                    "refs/heads/trunk",
                    "trunk",
                    "trunk",
                ),
                "protected-default",
            ),
            (
                (
                    "classify",
                    "feature/current",
                    "refs/heads/feature/current",
                    "trunk",
                    "trunk",
                ),
                "protected-current",
            ),
            (
                (
                    "classify",
                    "regular",
                    "refs/heads/regular",
                    "trunk",
                    "trunk",
                ),
                "normal-merge",
            ),
            (
                (
                    "classify",
                    "squash-exact",
                    "refs/heads/squash-exact",
                    "trunk",
                    "trunk",
                    self.squash_oid,
                ),
                "squash-merge",
            ),
            (
                (
                    "classify",
                    "squash-reused",
                    "refs/heads/squash-reused",
                    "trunk",
                    "trunk",
                    self.squash_oid,
                ),
                "reused-or-unfinished",
            ),
        )

        for arguments, expected in cases:
            with self.subTest(branch=arguments[1]):
                result = self.helper(*arguments)
                self.assertEqual(expected, result.stdout.strip())

    def test_classifies_local_and_remote_refs_with_independent_tips(self) -> None:
        remote_oid = self.rev_parse("feature/current")
        self.git("branch", "split-tip", self.regular_oid)
        self.git(
            "push",
            "origin",
            f"{remote_oid}:refs/heads/split-tip",
        )
        self.git("fetch", "origin")

        local_result = self.helper(
            "classify",
            "split-tip",
            "refs/heads/split-tip",
            "trunk",
            "trunk",
            remote_oid,
        )
        remote_result = self.helper(
            "classify",
            "split-tip",
            "refs/remotes/origin/split-tip",
            "trunk",
            "trunk",
            remote_oid,
        )

        self.assertEqual("normal-merge", local_result.stdout.strip())
        self.assertEqual("squash-merge", remote_result.stdout.strip())

    def test_classifies_remote_only_ref(self) -> None:
        remote_oid = self.rev_parse("feature/current")
        self.git(
            "push",
            "origin",
            f"{remote_oid}:refs/heads/remote-only",
        )
        self.git("fetch", "origin")

        result = self.helper(
            "classify",
            "remote-only",
            "refs/remotes/origin/remote-only",
            "trunk",
            "trunk",
            remote_oid,
        )

        self.assertEqual("squash-merge", result.stdout.strip())

    def test_classify_rejects_non_full_or_mismatched_branch_ref(self) -> None:
        invalid_refs = (
            "regular",
            "refs/heads/squash-exact",
            "refs/tags/regular",
        )

        for branch_ref in invalid_refs:
            with self.subTest(branch_ref=branch_ref):
                result = self.helper(
                    "classify",
                    "regular",
                    branch_ref,
                    "trunk",
                    "trunk",
                    check=False,
                )
                self.assertNotEqual(0, result.returncode)

    def test_classify_fails_closed_when_merge_base_errors(self) -> None:
        real_git = shutil.which("git")
        if real_git is None:
            self.fail("git executable is required")
        wrapper_directory = self.root / "merge-base-error-bin"
        wrapper_directory.mkdir()
        git_wrapper = wrapper_directory / "git"
        git_wrapper.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "merge-base" ]; then\n'
            "  exit 42\n"
            "fi\n"
            f'exec {shlex.quote(real_git)} "$@"\n',
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{wrapper_directory}{os.pathsep}{environment['PATH']}"

        result = self.helper(
            "classify",
            "squash-exact",
            "refs/heads/squash-exact",
            "trunk",
            "trunk",
            self.squash_oid,
            check=False,
            env=environment,
        )

        self.assertNotEqual(0, result.returncode)

    def test_local_delete_protects_current_and_default_branches(self) -> None:
        protected = (
            ("trunk", self.rev_parse("trunk")),
            ("feature/current", self.rev_parse("feature/current")),
        )

        for branch, expected_oid in protected:
            with self.subTest(branch=branch):
                result = self.helper(
                    "delete-local",
                    branch,
                    expected_oid,
                    "trunk",
                    check=False,
                )
                self.assertNotEqual(0, result.returncode)
                self.assertEqual(
                    expected_oid,
                    self.rev_parse(f"refs/heads/{branch}"),
                )

    def test_local_delete_refuses_tip_changed_after_approval(self) -> None:
        approved_oid = self.rev_parse("local-race")
        self.git("checkout", "local-race")
        self.git("commit", "--allow-empty", "-m", "advance after approval")
        advanced_oid = self.rev_parse("local-race")
        self.git("checkout", "feature/current")

        result = self.helper(
            "delete-local",
            "local-race",
            approved_oid,
            "trunk",
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("changed since approval", result.stderr)
        self.assertEqual(advanced_oid, self.rev_parse("refs/heads/local-race"))

    def test_local_delete_removes_exact_approved_tip(self) -> None:
        approved_oid = self.rev_parse("local-safe")
        self.git("config", "branch.local-safe.test-marker", "keep-until-delete")

        self.helper("delete-local", "local-safe", approved_oid, "trunk")

        result = self.git(
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/local-safe",
            check=False,
        )
        self.assertNotEqual(0, result.returncode)
        config_result = self.git(
            "config",
            "--get",
            "branch.local-safe.test-marker",
            check=False,
        )
        self.assertNotEqual(0, config_result.returncode)

    def test_local_delete_refuses_branch_checked_out_in_linked_worktree(
        self,
    ) -> None:
        expected_oid = self.rev_parse("linked-worktree")
        linked_worktree = self.root / "linked-worktree"
        self.git("worktree", "add", linked_worktree, "linked-worktree")

        result = self.helper(
            "delete-local",
            "linked-worktree",
            expected_oid,
            "trunk",
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertEqual(
            expected_oid,
            self.rev_parse("refs/heads/linked-worktree"),
        )

    def test_local_delete_fails_closed_when_worktree_listing_errors(self) -> None:
        branch = "worktree-list-error"
        expected_oid = self.regular_oid
        self.git("branch", branch, expected_oid)
        self.git("config", f"branch.{branch}.test-marker", "preserve-on-error")

        real_git = shutil.which("git")
        if real_git is None:
            self.fail("git executable is required")
        wrapper_directory = self.root / "worktree-error-bin"
        wrapper_directory.mkdir()
        git_wrapper = wrapper_directory / "git"
        git_wrapper.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "worktree" ] && [ "$2" = "list" ]; then\n'
            "  exit 42\n"
            "fi\n"
            f'exec {shlex.quote(real_git)} "$@"\n',
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{wrapper_directory}{os.pathsep}{environment['PATH']}"

        result = self.helper(
            "delete-local",
            branch,
            expected_oid,
            "trunk",
            check=False,
            env=environment,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertEqual(
            expected_oid,
            self.rev_parse(f"refs/heads/{branch}"),
        )
        self.assertEqual(
            "preserve-on-error",
            self.git(
                "config",
                "--get",
                f"branch.{branch}.test-marker",
            ).stdout.strip(),
        )

    def test_local_delete_fails_closed_when_worktree_search_errors(self) -> None:
        branch = "worktree-search-error"
        expected_oid = self.regular_oid
        self.git("branch", branch, expected_oid)
        self.git("config", f"branch.{branch}.test-marker", "preserve-on-error")

        wrapper_directory = self.root / "worktree-search-error-bin"
        wrapper_directory.mkdir()
        grep_wrapper = wrapper_directory / "grep"
        grep_wrapper.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
        grep_wrapper.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{wrapper_directory}{os.pathsep}{environment['PATH']}"

        result = self.helper(
            "delete-local",
            branch,
            expected_oid,
            "trunk",
            check=False,
            env=environment,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertEqual(
            expected_oid,
            self.rev_parse(f"refs/heads/{branch}"),
        )
        self.assertEqual(
            "preserve-on-error",
            self.git(
                "config",
                "--get",
                f"branch.{branch}.test-marker",
            ).stdout.strip(),
        )

    def test_local_atomic_delete_rejects_advance_after_last_read(self) -> None:
        approved_oid = self.rev_parse("local-atomic")
        self.git("config", "branch.local-atomic.test-marker", "preserve-on-race")
        self.git("checkout", "-b", "local-atomic-new", "trunk")
        self.git("commit", "--allow-empty", "-m", "atomic local advance")
        advanced_oid = self.rev_parse("local-atomic-new")
        self.git("checkout", "feature/current")

        real_git = shutil.which("git")
        if real_git is None:
            self.fail("git executable is required")
        wrapper_directory = self.root / "local-race-bin"
        wrapper_directory.mkdir()
        git_wrapper = wrapper_directory / "git"
        git_wrapper.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "update-ref" ] && [ "$2" = "-d" ]; then\n'
            f"  {shlex.quote(real_git)} -C {shlex.quote(str(self.work))} "
            f"update-ref refs/heads/local-atomic {advanced_oid} "
            f"{approved_oid}\n"
            "fi\n"
            f'exec {shlex.quote(real_git)} "$@"\n',
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{wrapper_directory}{os.pathsep}{environment['PATH']}"

        result = self.helper(
            "delete-local",
            "local-atomic",
            approved_oid,
            "trunk",
            check=False,
            env=environment,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertEqual(
            advanced_oid,
            self.rev_parse("refs/heads/local-atomic"),
        )
        self.assertEqual(
            "preserve-on-race",
            self.git(
                "config",
                "--get",
                "branch.local-atomic.test-marker",
            ).stdout.strip(),
        )

    def test_remote_delete_refuses_base_remote(self) -> None:
        expected_oid = self.bare_ref(
            self.upstream,
            "refs/heads/base-only",
        )

        result = self.helper(
            "delete-remote",
            "base-only",
            expected_oid,
            "trunk",
            "upstream",
            "upstream",
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("base remote", result.stderr)
        self.assertEqual(
            expected_oid,
            self.bare_ref(self.upstream, "refs/heads/base-only"),
        )

    def test_remote_delete_refuses_tip_advanced_after_approval(self) -> None:
        approved_oid = self.bare_ref(self.origin, "refs/heads/remote-race")
        self.git("checkout", "remote-race")
        self.git("commit", "--allow-empty", "-m", "remote advance")
        advanced_oid = self.rev_parse("remote-race")
        self.git("push", "origin", "remote-race")
        self.git("checkout", "feature/current")

        result = self.helper(
            "delete-remote",
            "remote-race",
            approved_oid,
            "trunk",
            "origin",
            "upstream",
            check=False,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("changed since approval", result.stderr)
        self.assertEqual(
            advanced_oid,
            self.bare_ref(self.origin, "refs/heads/remote-race"),
        )

    def test_remote_delete_lease_rejects_advance_after_last_read(self) -> None:
        approved_oid = self.bare_ref(self.origin, "refs/heads/remote-atomic")
        self.git("checkout", "-b", "remote-atomic-new", "trunk")
        self.git("commit", "--allow-empty", "-m", "atomic remote advance")
        advanced_oid = self.rev_parse("remote-atomic-new")
        self.git("push", "origin", "remote-atomic-new:staged-new")
        self.git("checkout", "feature/current")

        real_git = shutil.which("git")
        if real_git is None:
            self.fail("git executable is required")
        wrapper_directory = self.root / "race-bin"
        wrapper_directory.mkdir()
        git_wrapper = wrapper_directory / "git"
        git_wrapper.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "push" ]; then\n'
            f"  {shlex.quote(real_git)} "
            f"--git-dir={shlex.quote(str(self.origin))} update-ref "
            f"refs/heads/remote-atomic {advanced_oid} {approved_oid}\n"
            "fi\n"
            f'exec {shlex.quote(real_git)} "$@"\n',
            encoding="utf-8",
        )
        git_wrapper.chmod(0o755)
        environment = os.environ.copy()
        environment["PATH"] = f"{wrapper_directory}{os.pathsep}{environment['PATH']}"

        result = self.helper(
            "delete-remote",
            "remote-atomic",
            approved_oid,
            "trunk",
            "origin",
            "upstream",
            check=False,
            env=environment,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertEqual(
            advanced_oid,
            self.bare_ref(self.origin, "refs/heads/remote-atomic"),
        )

    def test_remote_delete_removes_exact_approved_tip(self) -> None:
        approved_oid = self.bare_ref(self.origin, "refs/heads/remote-safe")

        self.helper(
            "delete-remote",
            "remote-safe",
            approved_oid,
            "trunk",
            "origin",
            "upstream",
        )

        self.assert_ref_missing(self.origin, "refs/heads/remote-safe")


if __name__ == "__main__":
    unittest.main()
