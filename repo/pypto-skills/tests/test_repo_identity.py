from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, TypedDict, cast

from tests.skill_assertions import ROOT

IDENTITY_HELPER = ROOT / "lib/github/scripts/repo-identity.sh"
HOST = "ghe.example.test"


class GHCall(TypedDict):
    argv: list[str]


def repository(
    full_name: str,
    *,
    fork: bool = False,
    can_push: bool = False,
    parent: str | None = None,
    default_branch: str = "trunk",
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "full_name": full_name,
        "fork": fork,
        "default_branch": default_branch,
        "html_url": f"https://{HOST}/{full_name}",
        "permissions": {"push": can_push},
    }
    if parent is not None:
        response["parent"] = {
            "full_name": parent,
            "html_url": f"https://{HOST}/{parent}",
        }
    return response


class RepoIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.temp_path = Path(temporary_directory.name)
        self.work = self.temp_path / "work"
        self.work.mkdir()
        subprocess.run(
            ["git", "init", "--initial-branch=trunk"],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )

        self.transcript_path = self.temp_path / "gh-transcript.jsonl"
        self.fixture_path = self.temp_path / "gh-fixtures.json"
        self.bin_path = self.temp_path / "bin"
        self.bin_path.mkdir()
        self.write_fake_gh()
        real_git = shutil.which("git")
        self.assertIsNotNone(real_git)
        assert real_git is not None
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{self.bin_path}{os.pathsep}{self.environment['PATH']}",
                "GH_TRANSCRIPT": str(self.transcript_path),
                "GH_FIXTURES": str(self.fixture_path),
                "TEST_REAL_GIT": real_git,
            }
        )

    def write_fake_gh(self) -> None:
        fake_gh = self.bin_path / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys

argv = sys.argv[1:]
with open(os.environ["GH_TRANSCRIPT"], "a", encoding="utf-8") as transcript:
    transcript.write(json.dumps({"argv": argv}) + "\\n")

with open(os.environ["GH_FIXTURES"], encoding="utf-8") as fixtures_file:
    fixtures = json.load(fixtures_file)

if not argv or argv[0] != "api":
    sys.stderr.write(f"unexpected gh invocation: {argv!r}\\n")
    raise SystemExit(91)

positional = []
index = 1
jq_filter = ""
while index < len(argv):
    if argv[index] in ("--hostname", "--jq"):
        if argv[index] == "--jq":
            jq_filter = argv[index + 1]
        index += 2
        continue
    positional.append(argv[index])
    index += 1

if len(positional) != 1:
    sys.stderr.write(f"unexpected gh api arguments: {argv!r}\\n")
    raise SystemExit(92)
endpoint = positional[0]

if endpoint == "user":
    login = fixtures.get("user")
    if login is None:
        sys.stderr.write("gh: authentication required (HTTP 401)\\n")
        raise SystemExit(1)
    print(login if jq_filter == ".login" else json.dumps({"login": login}))
    raise SystemExit(0)

if not endpoint.startswith("repos/"):
    sys.stderr.write(f"unexpected api endpoint: {endpoint}\\n")
    raise SystemExit(93)

response = fixtures["repos"].get(endpoint[len("repos/"):])
if response is None:
    sys.stderr.write("gh: repository not found (HTTP 404)\\n")
    raise SystemExit(1)
print(json.dumps(response))
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)

    def add_remote(self, name: str, repository_path: str) -> None:
        subprocess.run(
            ["git", "remote", "add", name, f"https://{HOST}/{repository_path}.git"],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )

    def write_fixtures(
        self, repositories: list[dict[str, Any]], *, user: str | None = None
    ) -> None:
        fixtures: dict[str, Any] = {
            "repos": {entry["full_name"]: entry for entry in repositories}
        }
        if user is not None:
            fixtures["user"] = user
        self.fixture_path.write_text(json.dumps(fixtures), encoding="utf-8")

    def resolve(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        self.assertTrue(
            IDENTITY_HELPER.is_file(), f"missing identity helper: {IDENTITY_HELPER}"
        )
        return subprocess.run(
            [str(IDENTITY_HELPER), "resolve", *arguments],
            cwd=self.work,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def transcript(self) -> list[GHCall]:
        if not self.transcript_path.exists():
            return []
        return [
            cast(GHCall, json.loads(line))
            for line in self.transcript_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_fork_checkout_pushes_to_the_fork_beside_an_upstream_remote(self) -> None:
        self.add_remote("origin", "contributor/widget")
        self.add_remote("upstream", "acme/widget")
        self.write_fixtures(
            [
                repository(
                    "contributor/widget",
                    fork=True,
                    can_push=True,
                    parent="acme/widget",
                    default_branch="work",
                ),
                repository("acme/widget", can_push=True),
            ],
            user="contributor",
        )

        result = self.resolve("--require-push")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {
                "github_host": HOST,
                "local_repo": "contributor/widget",
                "base_repo": "acme/widget",
                "is_fork": True,
                "local_can_push": True,
                "default_branch": "trunk",
            },
            json.loads(result.stdout),
        )
        calls = self.transcript()
        self.assertTrue(calls)
        self.assertTrue(all(call["argv"][:2] != ["repo", "view"] for call in calls))
        self.assertTrue(all("--hostname" in call["argv"] for call in calls), calls)

    def test_upstream_only_checkout_resolves_to_the_upstream(self) -> None:
        self.add_remote("origin", "acme/widget")
        self.write_fixtures([repository("acme/widget", can_push=True)])

        result = self.resolve("--require-push")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {
                "github_host": HOST,
                "local_repo": "acme/widget",
                "base_repo": "acme/widget",
                "is_fork": False,
                "local_can_push": True,
                "default_branch": "trunk",
            },
            json.loads(result.stdout),
        )

    def test_upstream_without_push_permission_is_an_explicit_error(self) -> None:
        self.add_remote("origin", "acme/widget")
        self.write_fixtures([repository("acme/widget", can_push=False)])

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("no writable repository", result.stderr)
        self.assertIn("acme/widget", result.stderr)
        self.assertIn("fork remotes: none", result.stderr)
        self.assertEqual("", result.stdout)

    def test_unwritable_fork_remote_never_becomes_the_push_target(self) -> None:
        self.add_remote("origin", "acme/widget")
        self.add_remote("author", "author/widget")
        self.write_fixtures(
            [
                repository("acme/widget", can_push=True),
                repository(
                    "author/widget", fork=True, can_push=False, parent="acme/widget"
                ),
            ]
        )

        result = self.resolve("--require-push")

        self.assertEqual(0, result.returncode, result.stderr)
        identity = json.loads(result.stdout)
        self.assertEqual("acme/widget", identity["local_repo"])
        self.assertFalse(identity["is_fork"])

    def test_no_writable_repository_names_every_rejected_fork(self) -> None:
        self.add_remote("origin", "acme/widget")
        self.add_remote("author", "author/widget")
        self.write_fixtures(
            [
                repository("acme/widget", can_push=False),
                repository(
                    "author/widget", fork=True, can_push=False, parent="acme/widget"
                ),
            ]
        )

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("author/widget", result.stderr)
        self.assertIn("no writable repository", result.stderr)

    def test_two_writable_forks_resolve_through_the_authenticated_account(self) -> None:
        self.add_remote("origin", "contributor/widget")
        self.add_remote("teammate", "teammate/widget")
        self.add_remote("upstream", "acme/widget")
        self.write_fixtures(
            [
                repository(
                    "contributor/widget", fork=True, can_push=True, parent="acme/widget"
                ),
                repository(
                    "teammate/widget", fork=True, can_push=True, parent="acme/widget"
                ),
                repository("acme/widget", can_push=False),
            ],
            user="Contributor",
        )

        result = self.resolve("--require-push")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("contributor/widget", json.loads(result.stdout)["local_repo"])

    def test_sole_writable_fork_owned_by_another_account_never_wins(self) -> None:
        # Write access to somebody else's fork is not ownership: the account
        # falls back to the writable base rather than publishing to that fork.
        self.add_remote("origin", "acme/widget")
        self.add_remote("teammate", "teammate/widget")
        self.write_fixtures(
            [
                repository("acme/widget", can_push=True),
                repository(
                    "teammate/widget", fork=True, can_push=True, parent="acme/widget"
                ),
            ],
            user="contributor",
        )

        result = self.resolve("--require-push")

        self.assertEqual(0, result.returncode, result.stderr)
        identity = json.loads(result.stdout)
        self.assertEqual("acme/widget", identity["local_repo"])
        self.assertFalse(identity["is_fork"])

    def test_sole_unowned_writable_fork_without_writable_base_stops(self) -> None:
        self.add_remote("origin", "acme/widget")
        self.add_remote("teammate", "teammate/widget")
        self.write_fixtures(
            [
                repository("acme/widget", can_push=False),
                repository(
                    "teammate/widget", fork=True, can_push=True, parent="acme/widget"
                ),
            ],
            user="contributor",
        )

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("no writable repository", result.stderr)
        self.assertIn("writable but not owned by contributor", result.stderr)
        self.assertIn("teammate/widget", result.stderr)
        self.assertEqual("", result.stdout)

    def test_two_owned_writable_forks_stop_instead_of_guessing(self) -> None:
        self.add_remote("origin", "contributor/widget")
        self.add_remote("mirror", "contributor/widget-mirror")
        self.write_fixtures(
            [
                repository(
                    "contributor/widget", fork=True, can_push=True, parent="acme/widget"
                ),
                repository(
                    "contributor/widget-mirror",
                    fork=True,
                    can_push=True,
                    parent="acme/widget",
                ),
                repository("acme/widget", can_push=False),
            ],
            user="contributor",
        )

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("multiple writable fork repositories", result.stderr)
        self.assertIn("contributor/widget-mirror", result.stderr)

    def test_writable_forks_owned_by_nobody_present_stop(self) -> None:
        self.add_remote("origin", "contributor/widget")
        self.add_remote("teammate", "teammate/widget")
        self.write_fixtures(
            [
                repository(
                    "contributor/widget", fork=True, can_push=True, parent="acme/widget"
                ),
                repository(
                    "teammate/widget", fork=True, can_push=True, parent="acme/widget"
                ),
                repository("acme/widget", can_push=False),
            ],
            user="someone-else",
        )

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("no writable repository", result.stderr)
        self.assertIn("writable but not owned by someone-else", result.stderr)
        self.assertIn("contributor/widget", result.stderr)
        self.assertIn("teammate/widget", result.stderr)

    def test_read_only_resolution_keeps_the_fork_without_permission_evidence(
        self,
    ) -> None:
        self.add_remote("origin", "contributor/widget")
        self.add_remote("upstream", "acme/widget")
        self.write_fixtures(
            [
                repository(
                    "contributor/widget",
                    fork=True,
                    can_push=False,
                    parent="acme/widget",
                ),
                repository("acme/widget", can_push=False),
            ]
        )

        result = self.resolve()

        self.assertEqual(0, result.returncode, result.stderr)
        identity = json.loads(result.stdout)
        self.assertEqual("contributor/widget", identity["local_repo"])
        self.assertEqual("acme/widget", identity["base_repo"])
        self.assertFalse(identity["local_can_push"])
        self.assertTrue(
            all(call["argv"][-1] != "user" for call in self.transcript()),
            self.transcript(),
        )

    def test_remotes_spanning_unrelated_repositories_stop(self) -> None:
        self.add_remote("origin", "acme/widget")
        self.add_remote("other", "acme/gadget")
        self.write_fixtures(
            [
                repository("acme/widget", can_push=True),
                repository("acme/gadget", can_push=True),
            ]
        )

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("unrelated repositories", result.stderr)

    def test_remotes_spanning_unrelated_hosts_stop(self) -> None:
        self.add_remote("origin", "acme/widget")
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "mirror",
                "https://other.example.test/acme/widget",
            ],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )
        self.write_fixtures([repository("acme/widget", can_push=True)])

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("unrelated hosts", result.stderr)

    def test_checkout_without_remotes_stops(self) -> None:
        self.write_fixtures([])

        result = self.resolve("--require-push")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("no remotes", result.stderr)

    def test_unknown_subcommand_and_flag_are_usage_errors(self) -> None:
        self.add_remote("origin", "acme/widget")
        self.write_fixtures([repository("acme/widget", can_push=True)])

        unknown_flag = self.resolve("--force")
        self.assertEqual(2, unknown_flag.returncode)
        self.assertIn("Usage:", unknown_flag.stderr)

        unknown_command = subprocess.run(
            [str(IDENTITY_HELPER), "publish"],
            cwd=self.work,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, unknown_command.returncode)
        self.assertIn("Usage:", unknown_command.stderr)


if __name__ == "__main__":
    unittest.main()
