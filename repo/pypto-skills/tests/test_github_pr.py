from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.skill_assertions import ROOT

HELPER = ROOT / "lib/github/scripts/pr-context.sh"


def pull_request(
    number: int = 17,
    *,
    state: str = "open",
    head: str = "topic",
    title: str = "Portable PR",
) -> dict[str, object]:
    return {
        "number": number,
        "state": state,
        "head": {"ref": head},
        "title": title,
    }


class GitHubPrContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temp_path = Path(self.temporary_directory.name)
        self.bin_path = self.temp_path / "bin"
        self.bin_path.mkdir()
        self.gh_log = self.temp_path / "gh-args.json"
        fake_gh = self.bin_path / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys

arguments = sys.argv[1:]
with open(os.environ["GH_LOG"], "w", encoding="utf-8") as stream:
    json.dump(arguments, stream)
if "--slurp" in arguments and ("--jq" in arguments or "--template" in arguments):
    sys.stderr.write(
        "the `--slurp` option is not supported with `--jq` or `--template`\\n"
    )
    raise SystemExit(2)
response = os.environ.get("FAKE_GH_RESPONSE", "[]")
if "--slurp" in arguments:
    try:
        response = json.dumps([json.loads(response)])
    except json.JSONDecodeError:
        pass
sys.stdout.write(response)
raise SystemExit(int(os.environ.get("FAKE_GH_EXIT", "0")))
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)

    def run_helper(
        self,
        *arguments: str,
        response: str = "[]",
        gh_exit: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        if not HELPER.is_file():
            self.fail(f"missing production helper: {HELPER}")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin_path}:{environment['PATH']}",
                "GH_LOG": str(self.gh_log),
                "FAKE_GH_RESPONSE": response,
                "FAKE_GH_EXIT": str(gh_exit),
            }
        )
        return subprocess.run(
            [str(HELPER), *arguments],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_lookup_forwards_enterprise_host_repo_and_fork_head_exactly(
        self,
    ) -> None:
        response = json.dumps([pull_request()])

        result = self.run_helper(
            "lookup",
            "ghe.example.com",
            "acme/widget",
            "acme-org:topic",
            response=response,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {
                "match_count": 1,
                "pr": {
                    "headRefName": "topic",
                    "number": 17,
                    "state": "OPEN",
                    "title": "Portable PR",
                },
                "route": "update",
            },
            json.loads(result.stdout),
        )
        self.assertEqual(
            [
                "api",
                "--hostname",
                "ghe.example.com",
                "--method",
                "GET",
                "repos/acme/widget/pulls",
                "-f",
                "state=open",
                "-f",
                "head=acme-org:topic",
                "-f",
                "per_page=100",
                "--paginate",
                "--slurp",
            ],
            json.loads(self.gh_log.read_text(encoding="utf-8")),
        )

    def test_lookup_allows_zero_only_for_create_routing(self) -> None:
        result = self.run_helper(
            "lookup",
            "ghe.example.com",
            "acme/widget",
            "acme-org:topic",
            "--allow-none",
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {"match_count": 0, "pr": None, "route": "create"},
            json.loads(result.stdout),
        )

    def test_lookup_fails_closed_on_zero_without_allow_none(self) -> None:
        result = self.run_helper(
            "lookup",
            "ghe.example.com",
            "acme/widget",
            "acme-org:topic",
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("no open pull request", result.stderr)

    def test_lookup_fails_closed_on_multiple_matches(self) -> None:
        response = json.dumps([pull_request(17), pull_request(18)])

        result = self.run_helper(
            "lookup",
            "ghe.example.com",
            "acme/widget",
            "acme-org:topic",
            "--allow-none",
            response=response,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("multiple open pull requests", result.stderr)

    def test_lookup_fails_closed_on_malformed_responses(self) -> None:
        malformed_responses = (
            "not-json",
            "{}",
            json.dumps([{"number": "17", "state": "open"}]),
            json.dumps([{"number": 17, "state": "open", "head": {"ref": ""}}]),
        )
        for response in malformed_responses:
            with self.subTest(response=response):
                result = self.run_helper(
                    "lookup",
                    "ghe.example.com",
                    "acme/widget",
                    "acme-org:topic",
                    "--allow-none",
                    response=response,
                )
                self.assertNotEqual(0, result.returncode)
                self.assertIn("malformed pull-request response", result.stderr)

    def test_lookup_fails_closed_when_gh_fails(self) -> None:
        result = self.run_helper(
            "lookup",
            "ghe.example.com",
            "acme/widget",
            "acme-org:topic",
            "--allow-none",
            gh_exit=17,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("pull-request lookup failed", result.stderr)

    def test_existing_match_routes_to_update_never_create(self) -> None:
        response = json.dumps([pull_request(91)])

        result = self.run_helper(
            "lookup",
            "github.example.net",
            "base/project",
            "fork-org:work",
            "--allow-none",
            response=response,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        route = json.loads(result.stdout)
        self.assertEqual("update", route["route"])
        self.assertEqual(91, route["pr"]["number"])

    def test_owner_and_fork_mismatch_stop_before_commit(self) -> None:
        if not HELPER.is_file():
            self.fail(f"missing production helper: {HELPER}")
        for role in ("owner", "fork"):
            with self.subTest(role=role):
                commit_marker = self.temp_path / f"{role}-commit-ran"
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        '"$1" guard-branch "$2" "$3" "$4" "$5" "$6" || '
                        'exit 23; touch "$7"',
                        "_",
                        str(HELPER),
                        role,
                        "local-work",
                        "pr-head",
                        "acme/widget",
                        "acme/widget",
                        str(commit_marker),
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(23, result.returncode)
                self.assertIn("PR head is pr-head", result.stderr)
                self.assertFalse(commit_marker.exists())

    def test_owner_and_fork_repository_mismatch_stop_before_commit(self) -> None:
        if not HELPER.is_file():
            self.fail(f"missing production helper: {HELPER}")
        for role in ("owner", "fork"):
            with self.subTest(role=role):
                commit_marker = self.temp_path / f"{role}-repo-commit-ran"
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        '"$1" guard-branch "$2" "$3" "$4" "$5" "$6" || '
                        'exit 23; touch "$7"',
                        "_",
                        str(HELPER),
                        role,
                        "topic",
                        "topic",
                        "acme/local-widget",
                        "acme/other-widget",
                        str(commit_marker),
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(23, result.returncode)
                self.assertIn("head repository", result.stderr)
                self.assertFalse(commit_marker.exists())

    def test_maintainer_mismatch_remains_available_for_safe_checkout(self) -> None:
        result = self.run_helper(
            "guard-branch",
            "maintainer",
            "local-work",
            "contributor-head",
            "base/widget",
            "contributor/widget",
        )

        self.assertEqual(0, result.returncode, result.stderr)

    def test_create_org_fork_uses_host_pinned_post_with_head_repo(self) -> None:
        response = json.dumps(
            {"html_url": "https://ghe.example.com/acme/widget/pull/42"}
        )
        result = self.run_helper(
            "create",
            "ghe.example.com",
            "acme/widget",
            "acme/widget-fork",
            "topic",
            "trunk",
            "Feature title",
            "Summary line\nSecond line",
            response=response,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            "https://ghe.example.com/acme/widget/pull/42\n",
            result.stdout,
        )
        arguments = json.loads(self.gh_log.read_text(encoding="utf-8"))
        self.assertEqual(
            [
                "api",
                "--hostname",
                "ghe.example.com",
                "--method",
                "POST",
                "repos/acme/widget/pulls",
                "-f",
                "title=Feature title",
                "-f",
                "body=Summary line\nSecond line",
                "-f",
                "head=acme:topic",
                "-f",
                "base=trunk",
                "-f",
                "head_repo=widget-fork",
            ],
            arguments,
        )
        self.assertNotEqual(["pr", "create"], arguments[:2])

    def test_create_owner_pr_omits_head_repo(self) -> None:
        response = json.dumps({"html_url": "https://github.com/acme/widget/pull/7"})
        result = self.run_helper(
            "create",
            "github.com",
            "acme/widget",
            "acme/widget",
            "topic",
            "main",
            "Owner title",
            "Owner body",
            response=response,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("https://github.com/acme/widget/pull/7\n", result.stdout)
        arguments = json.loads(self.gh_log.read_text(encoding="utf-8"))
        self.assertIn("head=topic", arguments)
        self.assertNotIn("-f=head_repo", arguments)
        self.assertFalse(any(item.startswith("head_repo=") for item in arguments))

    def test_create_cross_owner_fork_omits_head_repo(self) -> None:
        response = json.dumps({"html_url": "https://github.com/acme/widget/pull/8"})
        result = self.run_helper(
            "create",
            "github.com",
            "acme/widget",
            "contributor/widget",
            "topic",
            "main",
            "Fork title",
            "Fork body",
            response=response,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        arguments = json.loads(self.gh_log.read_text(encoding="utf-8"))
        self.assertIn("head=contributor:topic", arguments)
        self.assertFalse(any(item.startswith("head_repo=") for item in arguments))

    def test_create_fails_closed_on_malformed_responses(self) -> None:
        malformed_responses = (
            "not-json",
            "{}",
            json.dumps({"html_url": ""}),
            json.dumps({"html_url": 42}),
        )
        for response in malformed_responses:
            with self.subTest(response=response):
                result = self.run_helper(
                    "create",
                    "ghe.example.com",
                    "acme/widget",
                    "acme/widget-fork",
                    "topic",
                    "trunk",
                    "Feature title",
                    "Feature body",
                    response=response,
                )
                self.assertNotEqual(0, result.returncode)
                self.assertIn("malformed pull-request creation response", result.stderr)

    def test_create_fails_closed_when_gh_fails(self) -> None:
        result = self.run_helper(
            "create",
            "ghe.example.com",
            "acme/widget",
            "acme/widget-fork",
            "topic",
            "trunk",
            "Feature title",
            "Feature body",
            gh_exit=17,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("pull-request creation failed", result.stderr)

    def test_pr_number_must_be_strictly_numeric(self) -> None:
        for value in ("", "0", "00", "000", "01", "0012", "12x", "-1", "1.5", "1 2"):
            with self.subTest(value=value):
                result = self.run_helper("validate-number", value)
                self.assertNotEqual(0, result.returncode)

        for value in ("1", "9", "10", "123"):
            with self.subTest(value=value):
                result = self.run_helper("validate-number", value)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual(f"{value}\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
