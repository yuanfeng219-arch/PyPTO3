from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict, cast

from tests.skill_assertions import ROOT

CONTEXT_HELPER = ROOT / "lib/github/scripts/issue-context.sh"
CREATE_HELPER = ROOT / "skills/create-issue/scripts/issue-create.sh"
SKILL = ROOT / "skills/create-issue/SKILL.md"


class GHCall(TypedDict):
    argv: list[str]
    host: str
    body: str
    body_file: str


class CreateIssueBehaviorTests(unittest.TestCase):
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
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "enterprise",
                "ssh://git@ghe.example.test/acme/widget.git",
            ],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )

        self.transcript_path = self.temp_path / "gh-transcript.jsonl"
        self.bin_path = self.temp_path / "bin"
        self.bin_path.mkdir()
        self.write_fake_gh()
        self.write_fake_git()
        self.write_fake_cat()
        real_git = shutil.which("git")
        real_cat = shutil.which("cat")
        self.assertIsNotNone(real_git)
        self.assertIsNotNone(real_cat)
        assert real_git is not None
        assert real_cat is not None
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{self.bin_path}{os.pathsep}{self.environment['PATH']}",
                "GH_TRANSCRIPT": str(self.transcript_path),
                "TEST_REAL_GIT": real_git,
                "TEST_REAL_CAT": real_cat,
            }
        )
        self.body_file = self.temp_path / "issue.md"
        self.body_file.write_text(
            """### Component

allocator

### Reproduction

Run the allocator regression case.

### Expected

The allocation succeeds.

### Actual

The allocation fails.

Related: #18
""",
            encoding="utf-8",
        )
        self.title = "[Regression] allocator fails after reuse"

    def write_fake_gh(self) -> None:
        fake_gh = self.bin_path / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
import signal
import sys

argv = sys.argv[1:]
body = ""
body_file = ""
if argv[:2] == ["issue", "create"]:
    body_file = argv[argv.index("--body-file") + 1]
    with open(body_file, encoding="utf-8") as approved_body:
        body = approved_body.read()
with open(os.environ["GH_TRANSCRIPT"], "a", encoding="utf-8") as transcript:
    transcript.write(json.dumps({
        "argv": argv,
        "host": os.environ.get("GH_HOST", ""),
        "body": body,
        "body_file": body_file,
    }) + "\\n")

if argv[:2] == ["issue", "list"]:
    print(json.dumps([
        {
            "number": 7,
            "title": "Open allocator cleanup",
            "body": "Related cleanup",
            "state": "OPEN",
            "labels": [{"name": "maintenance"}],
            "url": "https://ghe.example.test/acme/widget/issues/7",
        },
        {
            "number": 18,
            "title": "Closed allocator regression",
            "body": "Different root cause",
            "state": "CLOSED",
            "labels": [{"name": "bug"}],
            "url": "https://ghe.example.test/acme/widget/issues/18",
        },
    ]))
elif argv and argv[0] == "api":
    endpoint = argv[-1]
    if endpoint.endswith("contents/.github/ISSUE_TEMPLATE"):
        if os.environ.get("GH_TEMPLATE_MISSING") == "1":
            sys.stderr.write("gh: Not Found (HTTP 404)\\n")
            raise SystemExit(1)
        if os.environ.get("GH_TEMPLATE_FAILURE") == "1":
            sys.stderr.write("gh: authentication failed (HTTP 401)\\n")
            raise SystemExit(1)
        print(json.dumps([
            {"name": "regression.yml", "path": ".github/ISSUE_TEMPLATE/regression.yml", "type": "file"},
            {"name": "legacy.md", "path": ".github/ISSUE_TEMPLATE/legacy.md", "type": "file"},
            {"name": "config.yml", "path": ".github/ISSUE_TEMPLATE/config.yml", "type": "file"},
        ]))
    elif endpoint == "repos/contributor/widget":
        print(json.dumps({
            "full_name": "contributor/widget",
            "fork": True,
            "default_branch": "work",
            "html_url": "https://ghe.example.test/contributor/widget",
            "parent": {
                "full_name": "acme/widget",
                "html_url": "https://ghe.example.test/acme/widget",
            },
        }))
    elif endpoint == "repos/acme/widget":
        if os.environ.get("GH_REPO_FAILURE") == "1":
            sys.stderr.write("gh: repository not found (HTTP 404)\\n")
            raise SystemExit(1)
        html_url = "https://ghe.example.test/acme/widget"
        if os.environ.get("GH_CONTRADICTORY_URL") == "1":
            html_url = "https://ghe.example.test/attacker/widget"
        print(json.dumps({
            "full_name": "acme/widget",
            "fork": False,
            "default_branch": "trunk",
            "html_url": html_url,
        }))
    else:
        sys.stderr.write(f"unexpected api endpoint: {endpoint}\\n")
        raise SystemExit(92)
elif argv[:2] == ["issue", "create"]:
    create_stdout = os.environ.get(
        "GH_CREATE_STDOUT", "https://ghe.example.test/acme/widget/issues/42\\n"
    )
    if os.environ.get("BLOCK_GH_AFTER_RESPONSE_READY"):
        sys.stdout.write(create_stdout)
        sys.stdout.flush()
        with open(os.environ["BLOCK_GH_AFTER_RESPONSE_READY"], "w", encoding="utf-8") as ready:
            ready.write("ready")
            ready.flush()
            signal.pause()
    sys.stdout.write(create_stdout)
    sys.stderr.write(os.environ.get("GH_CREATE_STDERR", ""))
    raise SystemExit(int(os.environ.get("GH_CREATE_EXIT", "0")))
else:
    sys.stderr.write(f"unexpected gh invocation: {argv!r}\\n")
    raise SystemExit(91)
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)

    def write_fake_git(self) -> None:
        fake_git = self.bin_path / "git"
        fake_git.write_text(
            """#!/usr/bin/env python3
import os
import sys

real_git = os.environ["TEST_REAL_GIT"]
os.execv(real_git, [real_git, *sys.argv[1:]])
""",
            encoding="utf-8",
        )
        fake_git.chmod(0o755)

    def write_fake_cat(self) -> None:
        fake_cat = self.bin_path / "cat"
        fake_cat.write_text(
            """#!/usr/bin/env python3
import os
import signal
import subprocess
import sys

real_cat = os.environ["TEST_REAL_CAT"]
argv = sys.argv[1:]
is_response_read = any("issue-create-response.stdout." in argument for argument in argv)
is_snapshot_copy = any(argument.endswith("issue.md") for argument in argv)
is_snapshot_read = any(
    "/issue-create." in argument and "issue-create-response." not in argument
    for argument in argv
)
if is_snapshot_read and os.environ.get("FAIL_CAT_SNAPSHOT_READ"):
    sys.stderr.write("cat: simulated snapshot read failure\\n")
    raise SystemExit(1)
if is_snapshot_copy and os.environ.get("BLOCK_CAT_SNAPSHOT_READY"):
    with open(os.environ["BLOCK_CAT_SNAPSHOT_READY"], "w", encoding="utf-8") as ready:
        ready.write("ready")
        ready.flush()
        signal.pause()
if is_snapshot_copy and os.environ.get("RACE_BODY_FILE"):
    result = subprocess.run([real_cat, *argv], check=False, capture_output=True)
    sys.stdout.buffer.write(result.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(result.stderr)
    with open(os.environ["RACE_BODY_FILE"], "w", encoding="utf-8") as body:
        body.write(os.environ["RACE_REPLACEMENT"])
    raise SystemExit(result.returncode)
if is_response_read and os.environ.get("BLOCK_CAT_AFTER_RESPONSE_READY"):
    result = subprocess.run(
        [real_cat, *argv],
        check=False,
        capture_output=True,
    )
    sys.stdout.buffer.write(result.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(result.stderr)
    sys.stderr.buffer.flush()
    with open(os.environ["BLOCK_CAT_AFTER_RESPONSE_READY"], "w", encoding="utf-8") as ready:
        ready.write("ready")
        ready.flush()
    with open(os.environ["BLOCK_CAT_RELEASE"], encoding="utf-8") as release:
        release.read(2)
    raise SystemExit(result.returncode)
os.execv(real_cat, [real_cat, *argv])
""",
            encoding="utf-8",
        )
        fake_cat.chmod(0o755)

    def transcript(self) -> list[GHCall]:
        if not self.transcript_path.exists():
            return []
        return [
            cast(GHCall, json.loads(line))
            for line in self.transcript_path.read_text(encoding="utf-8").splitlines()
        ]

    def context(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        self.assertTrue(
            CONTEXT_HELPER.is_file(), f"missing context helper: {CONTEXT_HELPER}"
        )
        return subprocess.run(
            [str(CONTEXT_HELPER), *arguments],
            cwd=self.work,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def issue_create(
        self,
        mode: str,
        body_file: Path,
        *labels: str,
        title: str | None = None,
        host: str = "ghe.example.test",
        repo: str = "acme/widget",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.assertTrue(
            CREATE_HELPER.is_file(), f"missing create helper: {CREATE_HELPER}"
        )
        arguments = [
            str(CREATE_HELPER),
            mode,
            host,
            repo,
            self.title if title is None else title,
            str(body_file),
        ]
        arguments.extend(labels)
        result = subprocess.run(
            arguments,
            cwd=self.work,
            env=self.environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            self.fail(f"issue-create failed:\n{result.stdout}{result.stderr}")
        return result

    def test_repository_context_bootstraps_from_git_without_ambient_repo(self) -> None:
        result = self.context("repository")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {
                "github_host": "ghe.example.test",
                "local_repo": "acme/widget",
                "issue_repo": "acme/widget",
                "default_branch": "trunk",
            },
            json.loads(result.stdout),
        )
        calls = self.transcript()
        self.assertTrue(calls)
        self.assertTrue(all(call["argv"][:2] != ["repo", "view"] for call in calls))
        self.assertTrue(all(call["host"] == "" for call in calls))
        self.assertTrue(all("--hostname" in call["argv"] for call in calls), calls)

    def test_context_reads_pin_enterprise_host_and_repository(self) -> None:
        result = self.context(
            "search", "ghe.example.test", "acme/widget", "allocator regression"
        )
        self.assertEqual(0, result.returncode, result.stderr)
        calls = self.transcript()
        self.assertTrue(
            any(
                "--repo" in call["argv"] and "acme/widget" in call["argv"]
                for call in calls
            )
        )
        self.assertTrue(all(call["host"] == "ghe.example.test" for call in calls))
        search_call = calls[0]["argv"]
        self.assertIn("--state", search_call)
        self.assertIn("all", search_call)
        self.assertIn("--limit", search_call)
        self.assertIn("1000", search_call)
        states = {issue["state"] for issue in json.loads(result.stdout)}
        self.assertEqual({"OPEN", "CLOSED"}, states)

    def test_repository_context_selects_fork_parent_with_both_remotes(self) -> None:
        subprocess.run(
            ["git", "remote", "remove", "enterprise"],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "fork",
                "ssh://git@ghe.example.test/contributor/widget.git",
            ],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "parent",
                "https://ghe.example.test/acme/widget.git",
            ],
            cwd=self.work,
            check=True,
            capture_output=True,
            text=True,
        )

        result = self.context("repository")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {
                "github_host": "ghe.example.test",
                "local_repo": "contributor/widget",
                "issue_repo": "acme/widget",
                "default_branch": "trunk",
            },
            json.loads(result.stdout),
        )

    def test_repository_context_rejects_full_name_url_repository_conflict(self) -> None:
        self.environment["GH_CONTRADICTORY_URL"] = "1"

        result = self.context("repository")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("does not match", result.stderr)

    def test_template_discovery_is_host_pinned_and_lists_forms_and_legacy(self) -> None:
        result = self.context("templates", "ghe.example.test", "acme/widget")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            {"regression.yml", "legacy.md"},
            {entry["name"] for entry in json.loads(result.stdout)},
        )
        self.assertEqual(
            [
                "api",
                "--hostname",
                "ghe.example.test",
                "--paginate",
                "repos/acme/widget/contents/.github/ISSUE_TEMPLATE",
            ],
            self.transcript()[0]["argv"],
        )

    def test_missing_template_directory_is_empty_but_other_failures_are_fatal(
        self,
    ) -> None:
        missing_environment = self.environment.copy()
        missing_environment["GH_TEMPLATE_MISSING"] = "1"
        missing = subprocess.run(
            [
                str(CONTEXT_HELPER),
                "templates",
                "ghe.example.test",
                "acme/widget",
            ],
            cwd=self.work,
            env=missing_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, missing.returncode, missing.stderr)
        self.assertEqual([], json.loads(missing.stdout))

        failure_environment = self.environment.copy()
        failure_environment["GH_TEMPLATE_FAILURE"] = "1"
        failed = subprocess.run(
            [
                str(CONTEXT_HELPER),
                "templates",
                "ghe.example.test",
                "acme/widget",
            ],
            cwd=self.work,
            env=failure_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, failed.returncode)

        repository_failure_environment = missing_environment.copy()
        repository_failure_environment["GH_REPO_FAILURE"] = "1"
        missing_repository = subprocess.run(
            [
                str(CONTEXT_HELPER),
                "templates",
                "ghe.example.test",
                "acme/widget",
            ],
            cwd=self.work,
            env=repository_failure_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, missing_repository.returncode)

    def test_create_is_the_only_route_and_takes_no_approval_token(self) -> None:
        helper_source = CREATE_HELPER.read_text(encoding="utf-8")
        self.assertNotIn("hash-object", helper_source)
        self.assertNotIn("TOKEN", helper_source)
        self.assertNotIn("ISSUE_CREATE:", helper_source)

        preview = self.issue_create("preview", self.body_file, "bug", check=False)
        self.assertEqual(2, preview.returncode)
        self.assertIn("issue-create.sh create", preview.stderr)
        self.assertEqual([], self.transcript())

    def recorded_body(self, record: str) -> str:
        marker = re.search(r"Body \((\d+) bytes\):\n", record)
        self.assertIsNotNone(marker, record)
        assert marker is not None
        end = record.find("\nISSUE_CREATE_PUBLISHING_END", marker.end())
        self.assertGreaterEqual(end, marker.end(), record)
        body = record[marker.end() : end]
        self.assertEqual(
            int(marker.group(1)),
            len(body.encode("utf-8")),
            "declared byte count must match the recorded body",
        )
        return body

    def test_create_records_the_published_payload_before_the_write(self) -> None:
        approved_body = self.body_file.read_text(encoding="utf-8")

        result = self.issue_create("create", self.body_file, "bug", "regression")

        record = result.stderr
        start = record.find("ISSUE_CREATE_PUBLISHING\n")
        end = record.find("ISSUE_CREATE_PUBLISHING_END")
        self.assertGreaterEqual(start, 0, record)
        self.assertGreater(end, start, record)
        block = record[start:end]
        self.assertIn("Host: ghe.example.test", block)
        self.assertIn("Repository: acme/widget", block)
        self.assertIn(f"Title: {self.title}", block)
        self.assertIn("Labels (2):\n- bug\n- regression", block)
        self.assertEqual(approved_body, self.recorded_body(record))

    def test_record_preserves_trailing_body_bytes_exactly(self) -> None:
        for body in ("no trailing newline", "one\n", "several\n\n\n", "\n"):
            with self.subTest(body=body):
                self.transcript_path.unlink(missing_ok=True)
                self.body_file.write_text(body, encoding="utf-8")

                result = self.issue_create("create", self.body_file, "bug")

                self.assertEqual(body, self.recorded_body(result.stderr))
                calls = self.transcript()
                self.assertEqual(1, len(calls))
                self.assertEqual(body, calls[0]["body"])

    def test_recorded_payload_is_the_snapshot_not_the_mutated_source(self) -> None:
        approved_body = self.body_file.read_text(encoding="utf-8")
        replacement = "MUTATED AFTER CREATE SNAPSHOT\n"
        self.environment["RACE_BODY_FILE"] = str(self.body_file)
        self.environment["RACE_REPLACEMENT"] = replacement

        result = self.issue_create("create", self.body_file, "bug")

        self.assertIn(approved_body, result.stderr)
        self.assertNotIn(replacement, result.stderr)
        calls = self.transcript()
        self.assertEqual(1, len(calls))
        self.assertEqual(approved_body, calls[0]["body"])

    def test_unrecordable_payload_stops_before_the_github_call(self) -> None:
        self.environment["FAIL_CAT_SNAPSHOT_READ"] = "1"

        result = self.issue_create("create", self.body_file, "bug", check=False)

        self.assertNotEqual(0, result.returncode)
        self.assertIn("unable to record the payload being published", result.stderr)
        self.assertIn("ISSUE_CREATE_OUTCOME:confirmed_not_created", result.stderr)
        self.assertEqual([], self.transcript())

    def test_create_rejects_a_comma_label_the_cli_would_split(self) -> None:
        for labels in (("bug, regression",), ("bug", "help wanted,triage")):
            with self.subTest(labels=labels):
                result = self.issue_create(
                    "create", self.body_file, *labels, check=False
                )

                self.assertNotEqual(0, result.returncode)
                self.assertIn("contains a comma", result.stderr)
                self.assertIn(
                    "ISSUE_CREATE_OUTCOME:confirmed_not_created", result.stderr
                )
                self.assertNotIn("ISSUE_CREATE_PUBLISHING", result.stderr)
        self.assertEqual([], self.transcript())

    def test_create_rejects_control_characters_in_title_and_labels(self) -> None:
        cases = (
            {"title": "unsafe\ntitle", "labels": ("bug",)},
            {"title": self.title, "labels": ("bug\ttriage",)},
            {"title": self.title, "labels": ("unsafe\x1blabel",)},
            {"title": self.title, "labels": ("unsafe\x7flabel",)},
        )

        for case in cases:
            with self.subTest(case=case):
                result = self.issue_create(
                    "create",
                    self.body_file,
                    *case["labels"],
                    title=case["title"],
                    check=False,
                )
                self.assertNotEqual(0, result.returncode)
                self.assertIn(
                    "ISSUE_CREATE_OUTCOME:confirmed_not_created", result.stderr
                )
        self.assertEqual([], self.transcript())

    def test_create_rejects_an_unreadable_or_empty_body_before_gh_write(self) -> None:
        empty_body = self.temp_path / "empty.md"
        empty_body.write_text("", encoding="utf-8")

        for body_file in (empty_body, self.temp_path / "absent.md"):
            with self.subTest(body_file=body_file.name):
                result = self.issue_create("create", body_file, "bug", check=False)
                self.assertNotEqual(0, result.returncode)
                self.assertIn(
                    "ISSUE_CREATE_OUTCOME:confirmed_not_created", result.stderr
                )
        self.assertEqual([], self.transcript())

    def response_capture(self, stderr: str, stream: str) -> Path:
        match = re.search(rf"ISSUE_CREATE_RESPONSE_{stream.upper()}:([^\n]+)", stderr)
        self.assertIsNotNone(match, stderr)
        assert match is not None
        return Path(match.group(1))

    def interrupted_create(
        self, coordination_variable: str
    ) -> subprocess.CompletedProcess[str]:
        ready_path = self.temp_path / f"{coordination_variable}.fifo"
        os.mkfifo(ready_path, mode=0o600)
        environment = self.environment.copy()
        environment[coordination_variable] = str(ready_path)
        process = subprocess.Popen(
            [
                str(CREATE_HELPER),
                "create",
                "ghe.example.test",
                "acme/widget",
                self.title,
                str(self.body_file),
                "bug",
            ],
            cwd=self.work,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        with ready_path.open(encoding="utf-8") as ready:
            self.assertEqual("ready", ready.read(5))
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        return subprocess.CompletedProcess(
            process.args, process.returncode, stdout, stderr
        )

    def test_signal_before_mutation_is_confirmed_not_created(self) -> None:
        result = self.interrupted_create("BLOCK_CAT_SNAPSHOT_READY")

        self.assertNotEqual(0, result.returncode)
        self.assertIn("ISSUE_CREATE_OUTCOME:confirmed_not_created", result.stderr)
        self.assertNotIn("ISSUE_CREATE_OUTCOME:unknown", result.stderr)
        self.assertEqual([], self.transcript())

    def test_signal_after_response_before_output_is_unknown(self) -> None:
        result = self.interrupted_create("BLOCK_GH_AFTER_RESPONSE_READY")

        self.assertEqual(31, result.returncode)
        self.assertEqual("", result.stdout)
        self.assertIn("ISSUE_CREATE_OUTCOME:unknown", result.stderr)
        self.assertIn(
            "ISSUE_CREATE_VERIFY_TARGET:ghe.example.test/acme/widget", result.stderr
        )
        self.assertIn("ISSUE_CREATE_RETRY:blocked", result.stderr)
        stdout_capture = self.response_capture(result.stderr, "stdout")
        self.assertEqual(
            "https://ghe.example.test/acme/widget/issues/42\n",
            stdout_capture.read_text(encoding="utf-8"),
        )
        self.assertEqual(0o600, stdout_capture.stat().st_mode & 0o777)

    def test_broken_success_stdout_reports_unknown_on_saved_diagnostic_fd(
        self,
    ) -> None:
        ready_path = self.temp_path / "cat-ready.fifo"
        release_path = self.temp_path / "cat-release.fifo"
        os.mkfifo(ready_path, mode=0o600)
        os.mkfifo(release_path, mode=0o600)
        environment = self.environment.copy()
        environment.update(
            {
                "BLOCK_CAT_AFTER_RESPONSE_READY": str(ready_path),
                "BLOCK_CAT_RELEASE": str(release_path),
            }
        )
        process = subprocess.Popen(
            [
                str(CREATE_HELPER),
                "create",
                "ghe.example.test",
                "acme/widget",
                self.title,
                str(self.body_file),
                "bug",
            ],
            cwd=self.work,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        with ready_path.open(encoding="utf-8") as ready:
            self.assertEqual("ready", ready.read(5))
        self.assertEqual(1, len(self.transcript()))
        process.stdout.close()
        with release_path.open("w", encoding="utf-8") as release:
            release.write("go")
            release.flush()
        stderr = process.stderr.read()
        process.stderr.close()
        returncode = process.wait(timeout=5)

        self.assertNotEqual(0, returncode)
        self.assertIn("ISSUE_CREATE_OUTCOME:unknown", stderr)
        self.assertIn("ISSUE_CREATE_VERIFY_TARGET:ghe.example.test/acme/widget", stderr)
        self.assertIn("ISSUE_CREATE_RETRY:blocked", stderr)
        stdout_capture = self.response_capture(stderr, "stdout")
        self.assertEqual(
            "https://ghe.example.test/acme/widget/issues/42\n",
            stdout_capture.read_text(encoding="utf-8"),
        )
        self.assertEqual(0o600, stdout_capture.stat().st_mode & 0o777)

    def test_success_with_invalid_url_reports_created_but_unvalidated(self) -> None:
        unsafe_response = "created\x1b-but-url-missing\n"
        self.environment["GH_CREATE_STDOUT"] = unsafe_response

        result = self.issue_create("create", self.body_file, "bug", check=False)

        self.assertEqual(30, result.returncode)
        self.assertIn(
            "ISSUE_CREATE_OUTCOME:created_response_unvalidated", result.stderr
        )
        self.assertNotIn("\x1b", result.stderr)
        stdout_capture = self.response_capture(result.stderr, "stdout")
        self.assertEqual(unsafe_response, stdout_capture.read_text(encoding="utf-8"))
        self.assertEqual(0o600, stdout_capture.stat().st_mode & 0o777)

    def test_extra_url_path_is_created_but_unvalidated(self) -> None:
        self.environment["GH_CREATE_STDOUT"] = (
            "https://ghe.example.test/acme/widget/issues/not-an-issue/42\n"
        )

        result = self.issue_create("create", self.body_file, "bug", check=False)

        self.assertEqual(30, result.returncode)
        self.assertIn(
            "ISSUE_CREATE_OUTCOME:created_response_unvalidated", result.stderr
        )

    def test_only_exact_canonical_issue_url_shape_is_accepted(self) -> None:
        invalid_urls = (
            "https://ghe.example.test/acme/widget/issues/42/extra",
            "https://ghe.example.test/acme/widget/issues/42?view=full",
            "https://ghe.example.test/acme/widget/issues/42#discussion",
            "https://user@ghe.example.test/acme/widget/issues/42",
            "https://ghe.example.test:443/acme/widget/issues/42",
            "https://ghe.example.test/acme/widget.evil/issues/42",
        )

        for invalid_url in invalid_urls:
            with self.subTest(invalid_url=invalid_url):
                self.environment["GH_CREATE_STDOUT"] = f"{invalid_url}\n"
                result = self.issue_create("create", self.body_file, "bug", check=False)
                self.assertEqual(30, result.returncode)
                self.assertIn(
                    "ISSUE_CREATE_OUTCOME:created_response_unvalidated",
                    result.stderr,
                )

    def test_canonical_url_treats_validated_host_and_repo_as_literals(self) -> None:
        host = "ghe-1.example.test"
        repo = "acme.team/widget.repo"
        self.environment["GH_CREATE_STDOUT"] = f"https://{host}/{repo}/issues/42\n"

        result = self.issue_create(
            "create",
            self.body_file,
            "bug",
            host=host,
            repo=repo,
        )

        self.assertEqual(f"https://{host}/{repo}/issues/42", result.stdout.strip())
        call = self.transcript()[0]
        self.assertEqual(host, call["host"])
        self.assertIn(repo, call["argv"])

    def test_nonzero_gh_exit_reports_unknown_outcome_and_preserves_response(
        self,
    ) -> None:
        self.environment.update(
            {
                "GH_CREATE_STDOUT": "",
                "GH_CREATE_STDERR": "request \x1b timed out after send\n",
                "GH_CREATE_EXIT": "70",
            }
        )

        result = self.issue_create("create", self.body_file, "bug", check=False)

        self.assertEqual(31, result.returncode)
        self.assertIn("ISSUE_CREATE_OUTCOME:unknown", result.stderr)
        self.assertIn(
            "ISSUE_CREATE_VERIFY_TARGET:ghe.example.test/acme/widget", result.stderr
        )
        self.assertIn("ISSUE_CREATE_RETRY:blocked", result.stderr)
        self.assertNotIn("\x1b", result.stderr)
        stderr_capture = self.response_capture(result.stderr, "stderr")
        self.assertEqual(
            "request \x1b timed out after send\n",
            stderr_capture.read_text(encoding="utf-8"),
        )
        self.assertEqual(0o600, stderr_capture.stat().st_mode & 0o777)

    def test_successful_create_uses_exact_confirmed_payload_without_project_call(
        self,
    ) -> None:
        approved_body = self.body_file.read_text(encoding="utf-8")
        result = self.issue_create("create", self.body_file, "bug", "regression")

        self.assertEqual(
            "https://ghe.example.test/acme/widget/issues/42", result.stdout.strip()
        )
        calls = self.transcript()
        self.assertEqual(1, len(calls))
        call = calls[0]
        snapshot_path = call["body_file"]
        self.assertEqual(
            [
                "issue",
                "create",
                "--repo",
                "acme/widget",
                "--title",
                self.title,
                "--body-file",
                snapshot_path,
                "--label",
                "bug",
                "--label",
                "regression",
            ],
            call["argv"],
        )
        self.assertEqual("ghe.example.test", call["host"])
        self.assertEqual(approved_body, call["body"])
        self.assertNotEqual(str(self.body_file), snapshot_path)
        self.assertFalse(Path(snapshot_path).exists())

    def test_create_publishes_snapshot_when_source_changes_after_copy(self) -> None:
        approved_body = self.body_file.read_text(encoding="utf-8")
        replacement = "MUTATED AFTER CREATE SNAPSHOT\n"
        self.environment["RACE_BODY_FILE"] = str(self.body_file)
        self.environment["RACE_REPLACEMENT"] = replacement

        result = self.issue_create("create", self.body_file, "bug")

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(replacement, self.body_file.read_text(encoding="utf-8"))
        calls = self.transcript()
        self.assertEqual(1, len(calls))
        self.assertEqual(approved_body, calls[0]["body"])
        snapshot_path = calls[0]["body_file"]
        self.assertNotEqual(str(self.body_file), snapshot_path)
        self.assertFalse(Path(snapshot_path).exists())


class CreateIssueSkillContractTests(unittest.TestCase):
    def test_skill_resolves_repository_policy_before_issue_context(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing skill: {SKILL}")
        text = SKILL.read_text(encoding="utf-8")

        policy = text.find("../../lib/repository/policy.md")
        context = text.find("../../lib/github/issue-context.md")
        self.assertGreaterEqual(policy, 0)
        self.assertGreater(context, policy)
        policy_block = text[policy:context]
        self.assertIn("applicable repository instructions", policy_block)
        self.assertIn("missing or conflicting required policy", policy_block)

    def test_skill_orders_complete_text_confirmation_and_create(self) -> None:
        self.assertTrue(SKILL.is_file(), f"missing skill: {SKILL}")
        text = SKILL.read_text(encoding="utf-8")

        shown = text.find("## Show the complete issue in text")
        confirmation = text.find("## Wait for explicit confirmation")
        create = text.find("## Create exactly the approved issue")
        self.assertGreaterEqual(shown, 0)
        self.assertGreater(confirmation, shown)
        self.assertGreater(create, confirmation)
        shown_block = text[shown:confirmation]
        for field in ("host", "repository", "title", "label", "complete body"):
            with self.subTest(field=field):
                self.assertIn(field, shown_block.lower())
        self.assertIn("directly in the reply", shown_block)
        self.assertIn("one label per line", shown_block.lower())

    def test_skill_has_no_approval_token_step(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertNotIn("ISSUE_CREATE:", text)
        self.assertNotIn("TOKEN", text)
        self.assertNotIn("token", text.lower())
        self.assertIn(
            "scripts/issue-create.sh create HOST REPO TITLE\nBODY_FILE LABEL...", text
        )

    def test_skill_requires_discovered_fields_and_open_closed_deduplication(
        self,
    ) -> None:
        self.assertTrue(SKILL.is_file(), f"missing skill: {SKILL}")
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("required: true", text)
        self.assertIn("DUPLICATE", text)
        self.assertIn("RELATED", text)
        self.assertIn("NO_MATCH", text)
        self.assertIn("open and closed", text.lower())
        self.assertIn("Related: #N", text)
        self.assertIn("stop only for `duplicate`", text.lower())
        self.assertIn("never present a draft that still holds a placeholder", text)
        self.assertIn("../../lib/github/scripts/issue-context.sh", text)

    def test_skill_requires_read_only_verification_before_retrying_unknown_outcome(
        self,
    ) -> None:
        text = SKILL.read_text(encoding="utf-8").lower()

        self.assertIn("created_response_unvalidated", text)
        self.assertIn("outcome:unknown", text)
        self.assertIn("read-only verification", text)
        self.assertIn("do not retry", text)
        self.assertIn('gh_host="$github_host"', text)
        self.assertIn('--repo "$issue_repo"', text)
        self.assertIn("exact canonical issue url", text)
        self.assertIn("signal", text)
        self.assertIn("sigpipe", text)
        self.assertIn("stderr descriptor saved", text)


if __name__ == "__main__":
    unittest.main()
