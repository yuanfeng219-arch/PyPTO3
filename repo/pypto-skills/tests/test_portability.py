from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

from tests.skill_assertions import DEVELOPER_PLUGIN, ROOT, USER_PLUGIN

BANNED_TEXT = (
    "hw-native-sys/pypto",
    "hw-native-sys/simpler",
    "hw-native-sys/pypto-lib",
    "upstream/main",
    "origin/main",
    "AskUserQuestion",
    "EnterPlanMode",
    "Task tool",
)

DEPLOYABLE_ROOTS = (
    DEVELOPER_PLUGIN / "skills",
    DEVELOPER_PLUGIN / "lib",
    USER_PLUGIN / "skills",
    USER_PLUGIN / "lib",
)

REQUIRED_GITHUB_REFERENCES = (
    ROOT / "lib/github/setup.md",
    ROOT / "lib/github/lookup-pr.md",
    ROOT / "lib/github/branch-naming.md",
    ROOT / "lib/github/pr-description.md",
    ROOT / "lib/github/commit-and-push.md",
    ROOT / "lib/github/common-issues.md",
    ROOT / "lib/github/detect-permission.md",
    ROOT / "lib/github/fetch-comments.md",
    ROOT / "lib/github/reply-and-resolve.md",
    ROOT / "lib/github/checkout-fork-branch.md",
    ROOT / "lib/github/issue-context.md",
    ROOT / "lib/github/issue-templates.md",
)

REQUIRED_REPOSITORY_REFERENCES = (
    ROOT / "lib/repository/policy.md",
    ROOT / "lib/repository/scope.md",
)

SCOPE_GATED_SKILLS = (
    "skills/auto-pr/SKILL.md",
    "skills/clean-branches/SKILL.md",
    "skills/create-issue/SKILL.md",
    "skills/fix-issue/SKILL.md",
    "skills/fix-pr/SKILL.md",
    "skills/git-commit/SKILL.md",
    "skills/github-pr/SKILL.md",
)

GITHUB_CONTEXT_VARIABLES = (
    "REPO_ROOT",
    "CURRENT_BRANCH",
    "DEFAULT_BRANCH",
    "BASE_REMOTE",
    "BASE_REF",
    "PUSH_REMOTE",
    "PR_REPO",
    "PR_HEAD_PREFIX",
    "ROLE",
)

REFERENCE_INPUTS = {
    "setup.md": frozenset({"REPO_IDENTITY_HELPER"}),
    "lookup-pr.md": frozenset(
        {
            "CURRENT_BRANCH",
            "GITHUB_HOST",
            "PR_HEAD_BRANCH",
            "PR_HEAD_PREFIX",
            "PR_LOOKUP_ALLOW_NONE",
            "PR_LOOKUP_HELPER",
            "PR_NUMBER",
            "PR_REPO",
        }
    ),
    "branch-naming.md": frozenset(
        {"BRANCH_PREFIX", "BRANCH_SUMMARY", "CURRENT_BRANCH", "DEFAULT_BRANCH"}
    ),
    "pr-description.md": frozenset(
        {
            "BASE_REF",
            "GITHUB_HOST",
            "PR_BODY",
            "PR_NUMBER",
            "PR_REPO",
            "PR_ROUTE",
            "PR_TITLE",
        }
    ),
    "commit-and-push.md": frozenset(
        {
            "BASE_REMOTE",
            "CURRENT_BRANCH",
            "DEFAULT_BRANCH",
            "GITHUB_HOST",
            "HEAD_REPO",
            "LOCAL_REPO",
            "MAINTAINER_CHECKOUT_VERIFIED",
            "PREPARE_PUSH_HELPER",
            "PR_HEAD_BRANCH",
            "PR_REPO",
            "PUSH_REMOTE",
            "PUSH_TRANSACTION_HELPER",
            "ROLE",
            "WORKTREE_VALIDATION",
            "WORK_BRANCH",
        }
    ),
    "common-issues.md": frozenset({"PR_NUMBER", "PR_REPO", "REPOSITORY_NODE_ID"}),
    "detect-permission.md": frozenset({"GITHUB_HOST", "PR_NUMBER", "PR_REPO"}),
    "fetch-comments.md": frozenset(
        {
            "COMMENTS_CURSOR",
            "PR_NUMBER",
            "PR_REPO",
            "REVIEWS_CURSOR",
            "THREADS_CURSOR",
        }
    ),
    "reply-and-resolve.md": frozenset(
        {
            "COMMENT_DATABASE_ID",
            "HANDLED_LEDGER",
            "HANDLED_NODE_IDS",
            "PR_NUMBER",
            "PR_REPO",
            "REPLY_BODY",
            "THREAD_ID",
        }
    ),
    "checkout-fork-branch.md": frozenset(
        {"HEAD_REPO", "PR_HEAD_BRANCH", "PR_NUMBER", "PUSH_REMOTE", "ROLE"}
    ),
    "issue-context.md": frozenset(),
    "issue-templates.md": frozenset(),
}

BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)```", re.DOTALL)
SHELL_VARIABLE_USE_RE = re.compile(r"\$(?:{!?([A-Z][A-Z0-9_]*)|([A-Z][A-Z0-9_]*))")
SHELL_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=", re.MULTILINE)
SHELL_FOR_VARIABLE_RE = re.compile(r"^\s*for\s+([A-Z][A-Z0-9_]*)\s+in\b", re.MULTILINE)
SHELL_WHILE_READ_VARIABLE_RE = re.compile(
    r"^\s*while\b[^\n]*\bread(?:\s+-[A-Za-z]+)*\s+"
    r"([A-Z][A-Z0-9_]*)\s*;",
    re.MULTILINE,
)


def deployable_files() -> list[Path]:
    return sorted(
        path
        for root in DEPLOYABLE_ROOTS
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    )


def bash_source(path: Path) -> str:
    return "\n".join(BASH_BLOCK_RE.findall(path.read_text(encoding="utf-8")))


def shell_inputs(path: Path) -> set[str]:
    source = bash_source(path)
    definitions: dict[str, list[int]] = {}

    for pattern in (
        SHELL_ASSIGNMENT_RE,
        SHELL_FOR_VARIABLE_RE,
        SHELL_WHILE_READ_VARIABLE_RE,
    ):
        for match in pattern.finditer(source):
            line_end = source.find("\n", match.end())
            definition_position = len(source) if line_end < 0 else line_end
            definitions.setdefault(match.group(1), []).append(definition_position)

    inputs = set()
    for match in SHELL_VARIABLE_USE_RE.finditer(source):
        variable = match.group(1) or match.group(2)
        if not any(
            position < match.start() for position in definitions.get(variable, [])
        ):
            inputs.add(variable)
    return inputs


class PortabilityTests(unittest.TestCase):
    def test_deployable_content_has_no_banned_text(self) -> None:
        for path in deployable_files():
            text = path.read_text(encoding="utf-8")
            for banned in BANNED_TEXT:
                with self.subTest(path=path, banned=banned):
                    self.assertNotIn(banned, text)

    def test_required_github_references_exist(self) -> None:
        for path in REQUIRED_GITHUB_REFERENCES:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"missing required reference: {path}")

    def test_required_repository_references_exist(self) -> None:
        for path in REQUIRED_REPOSITORY_REFERENCES:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"missing required reference: {path}")

    def test_git_commit_delegates_consumer_repository_policy(self) -> None:
        skill = ROOT / "skills/git-commit/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("../../lib/repository/policy.md", text)
        self.assertNotRegex(text, r"\b(?:pytest|cargo test|npm test)\b")
        self.assertNotRegex(text, r"\b(?:feat|fix|refactor|chore|docs|test)\([^)]*\):")
        self.assertNotRegex(text, r"(?m)^\s*git add (?:-A|--all|\.|\*)")

    def test_create_issue_has_no_fixed_repository_or_project_policy(self) -> None:
        skill = ROOT / "skills/create-issue/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("../../lib/github/issue-context.md", text)
        self.assertIn("../../lib/github/issue-templates.md", text)
        self.assertNotRegex(text, r"(?i)project\s+#?\d+")
        self.assertNotRegex(text, r"(?m)^\s*gh issue create\b")
        self.assertLessEqual(len(text.splitlines()), 200)

    def test_fix_issue_has_no_fixed_repository_project_or_test_policy(self) -> None:
        skill = ROOT / "skills/fix-issue/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("../../lib/repository/policy.md", text)
        self.assertIn("../../lib/github/issue-context.md", text)
        self.assertIn("../../lib/github/branch-naming.md", text)
        self.assertNotRegex(text, r"(?i)project\s+#?\d+")
        self.assertNotRegex(text, r"\b(?:pytest|cargo test|npm test)\b")
        self.assertNotRegex(text, r"(?m)^\s*(?:fix|feat|refactor|docs|support)/")
        self.assertLessEqual(len(text.splitlines()), 200)

    def test_setup_defines_context_contract(self) -> None:
        setup = ROOT / "lib/github/setup.md"
        self.assertTrue(setup.is_file(), f"missing required reference: {setup}")
        setup_text = setup.read_text(encoding="utf-8")
        definitions = set(
            re.findall(r"(?m)^\s*(?:export )?([A-Z][A-Z0-9_]*)=", setup_text)
        )

        for variable in GITHUB_CONTEXT_VARIABLES:
            with self.subTest(variable=variable):
                self.assertIn(variable, definitions)

    def test_repository_identity_never_uses_ambient_cli_selection(self) -> None:
        helper = ROOT / "lib/github/scripts/repo-identity.sh"
        self.assertTrue(helper.is_file(), f"missing required helper: {helper}")
        self.assertTrue(
            os.access(helper, os.X_OK), f"helper is not executable: {helper}"
        )

        setup = ROOT / "lib/github/setup.md"
        text = setup.read_text(encoding="utf-8")
        self.assertIn("scripts/repo-identity.sh", text)
        self.assertIn(
            '"$REPO_IDENTITY_HELPER" resolve --require-push',
            bash_source(setup),
        )

        # `gh repo view` answers "what is gh's base repository here", which
        # prefers a parent over the checkout's own fork and prompts when
        # several remotes qualify. Identity comes from the remotes instead.
        for path in deployable_files():
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8")
                shell = content if path.suffix == ".sh" else bash_source(path)
                self.assertNotIn("gh repo view", shell)

    def test_push_target_prefers_a_writable_fork_and_fails_explicitly(self) -> None:
        collapsed = " ".join(
            (ROOT / "lib/github/setup.md").read_text(encoding="utf-8").split()
        )
        self.assertIn(
            "a fork the authenticated account both owns and can push to wins",
            collapsed,
        )
        self.assertIn(
            "Write access is not ownership",
            collapsed,
        )
        self.assertIn(
            "the base repository wins only when the account can push to it",
            collapsed,
        )
        self.assertIn(
            "Never fall back to a repository the account cannot push to",
            collapsed,
        )

    def test_issue_identity_delegates_to_the_shared_resolver(self) -> None:
        issue_helper = (ROOT / "lib/github/scripts/issue-context.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("repo-identity.sh", issue_helper)
        self.assertIn(
            "scripts/repo-identity.sh",
            (ROOT / "lib/github/issue-context.md").read_text(encoding="utf-8"),
        )

    def test_references_consume_only_explicit_inputs_before_definition(
        self,
    ) -> None:
        self.assertEqual(
            {path.name for path in REQUIRED_GITHUB_REFERENCES},
            set(REFERENCE_INPUTS),
        )
        for path in REQUIRED_GITHUB_REFERENCES:
            with self.subTest(path=path):
                self.assertEqual(REFERENCE_INPUTS[path.name], shell_inputs(path))

    def test_remote_validation_covers_fetch_and_push_destinations(self) -> None:
        setup = bash_source(ROOT / "lib/github/setup.md")
        self.assertIn("GITHUB_HOST=", setup)
        self.assertIn("git remote get-url --all", setup)
        self.assertIn("git remote get-url --push --all", setup)
        self.assertIn("PUSH_URL_COUNT", setup)
        self.assertRegex(
            setup,
            r'\[ "\$REMOTE_HOST" != "\$GITHUB_HOST" \]',
        )

    def test_push_branch_requires_verified_role_context(self) -> None:
        reference = bash_source(ROOT / "lib/github/commit-and-push.md")
        self.assertIn("owner|fork)", reference)
        self.assertIn(
            '[ "$PR_HEAD_BRANCH" != "$CURRENT_BRANCH" ]',
            reference,
        )
        self.assertIn("MAINTAINER_CHECKOUT_VERIFIED", reference)
        self.assertIn(
            'remote_targets_repo "$PUSH_REMOTE" "$EXPECTED_PUSH_REPO"',
            reference,
        )

    def test_author_workflow_requires_head_repository_push_permission(
        self,
    ) -> None:
        reference = bash_source(ROOT / "lib/github/detect-permission.md")
        self.assertIn(
            'HEAD_CAN_PUSH=$(gh api --hostname "$GITHUB_HOST" "repos/$HEAD_REPO"',
            reference,
        )
        self.assertIn('[ "$HEAD_CAN_PUSH" != "true" ]', reference)

    def test_validation_runs_in_the_worktree_without_an_isolation_boundary(
        self,
    ) -> None:
        reference = ROOT / "lib/github/commit-and-push.md"
        runner = ROOT / "lib/github/scripts/worktree-validation.sh"
        self.assertTrue(runner.is_file(), f"missing required runner: {runner}")
        self.assertFalse((ROOT / "lib/github/scripts/validation-sandbox.sh").exists())

        text = reference.read_text(encoding="utf-8")
        self.assertIn("scripts/worktree-validation.sh", text)
        self.assertIn("WORKTREE_VALIDATION", text)
        self.assertIn("harness permission controls", " ".join(text.split()))
        self.assertIn(
            "the main checkout or a linked `git worktree add` checkout",
            " ".join(text.split()),
        )

        for path in (
            reference,
            ROOT / "skills/github-pr/SKILL.md",
            ROOT / "skills/fix-pr/SKILL.md",
        ):
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8").lower()
                for stale in ("bubblewrap", "validation-sandbox", "validation sandbox"):
                    self.assertNotIn(stale, content)

    def test_rewritten_pushes_use_force_with_lease(self) -> None:
        reference = ROOT / "lib/github/commit-and-push.md"
        self.assertTrue(reference.is_file(), f"missing required reference: {reference}")
        text = reference.read_text(encoding="utf-8")
        self.assertIn("git push --force-with-lease", text)
        self.assertNotRegex(text, r"git push\s+--force(?!-with-lease)")

    def test_github_pr_uses_all_shared_workflow_references(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        expected_links = (
            "../../lib/github/setup.md",
            "../../lib/github/lookup-pr.md",
            "../../lib/github/branch-naming.md",
            "../../lib/github/commit-and-push.md",
            "../../lib/github/detect-permission.md",
            "../../lib/github/checkout-fork-branch.md",
        )
        for link in expected_links:
            with self.subTest(link=link):
                self.assertIn(link, text)

    def test_pull_request_skills_fit_the_portable_instruction_budget(self) -> None:
        for relative_path in (
            "skills/auto-pr/SKILL.md",
            "skills/fix-pr/SKILL.md",
            "skills/github-pr/SKILL.md",
        ):
            path = ROOT / relative_path
            with self.subTest(path=relative_path):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    210,
                )

    def test_mutating_skills_gate_on_repository_scope(self) -> None:
        for relative_path in SCOPE_GATED_SKILLS:
            path = ROOT / relative_path
            with self.subTest(path=relative_path):
                self.assertTrue(path.is_file(), f"missing required skill: {path}")
                text = path.read_text(encoding="utf-8")
                self.assertIn("../../lib/repository/scope.md", text)
                self.assertIn("scope gate", text)

    def test_shared_contracts_route_mutations_through_the_scope_gate(self) -> None:
        policy = (ROOT / "lib/repository/policy.md").read_text(encoding="utf-8")
        self.assertIn("(scope.md)", policy)

        setup = (ROOT / "lib/github/setup.md").read_text(encoding="utf-8")
        self.assertIn("../repository/scope.md", setup)
        # Setup's fetches write to the local repository, so the gate precedes
        # them; setup must not advertise itself as wholly read-only.
        self.assertNotIn("Every step below is read-only", setup)
        self.assertRegex(
            " ".join(setup.split()),
            r"Section 3 is not: its `git fetch` calls contact a remote",
        )

    def test_setup_running_skills_gate_before_the_setup_fetches(self) -> None:
        for relative_path in (
            "skills/clean-branches/SKILL.md",
            "skills/fix-pr/SKILL.md",
            "skills/github-pr/SKILL.md",
        ):
            path = ROOT / relative_path
            with self.subTest(path=relative_path):
                text = path.read_text(encoding="utf-8")
                gate = text.find("../../lib/repository/scope.md")
                setup = text.find("../../lib/github/setup.md")
                self.assertGreaterEqual(gate, 0)
                self.assertGreaterEqual(setup, 0)
                self.assertLess(gate, setup)
                self.assertRegex(" ".join(text.split()), r"setup'?s? fetches")

    def test_scope_gate_fails_closed_and_requires_explicit_confirmation(self) -> None:
        scope = ROOT / "lib/repository/scope.md"
        self.assertTrue(scope.is_file(), f"missing required reference: {scope}")
        text = scope.read_text(encoding="utf-8")
        collapsed = " ".join(text.split())

        self.assertIn("REPO_SCOPE=family", text)
        self.assertIn("REPO_SCOPE=foreign", text)
        self.assertIn("Fail closed", text)

        gate_block = BASH_BLOCK_RE.search(text)
        self.assertIsNotNone(gate_block)
        block = "" if gate_block is None else gate_block.group(1)
        # `foreign` must be the starting value, raised only by a complete sweep.
        self.assertLess(
            block.find("REPO_SCOPE=foreign"),
            block.find("REPO_SCOPE=family"),
        )
        # Every identity command is status-checked, so a partial sweep cannot
        # let an already-emitted marker classify the checkout as family.
        self.assertIn("SCOPE_REMOTES=$(git remote) || return 1", block)
        self.assertIn(
            'SCOPE_URLS=$(git remote get-url --all "$SCOPE_REMOTE") || return 1',
            block,
        )
        self.assertRegex(
            block,
            r"SCOPE_PUSH_URLS=\$\(git remote get-url --push --all "
            r'"\$SCOPE_REMOTE"\) \|\|\s*\n\s*return 1',
        )
        self.assertRegex(block, r"SCOPE_FILES=\$\(git ls-files[\s\S]*?\) \|\| return 1")
        self.assertNotRegex(block, r"git remote\s*\|")
        self.assertIn("SCOPE_DISCOVERY=failed", block)

        # Prose is not identity: a declaration file earns `family` only through
        # an affirmative opt-in token, never by mentioning the family at all.
        self.assertIn("pto-family-repository", block)
        self.assertIn('grep -Eih "$SCOPE_DECLARATION"', block)
        self.assertNotIn('grep -Eih "$SCOPE_MARKER"', block)
        self.assertRegex(collapsed, r"is a topic, not a claim of membership")

        # A remote's host and userinfo are infrastructure, not membership, so
        # only the repository path of each URL reaches the marker.
        self.assertIn("scope_url_path()", block)
        self.assertIn('scope_url_path "$SCOPE_URL" || return 1', block)
        self.assertRegex(collapsed, r"Only the path is identity")
        self.assertRegex(collapsed, r"Read-only inspection is never gated")
        self.assertRegex(
            collapsed,
            r"Wait for a confirmation that names this repository",
        )
        self.assertRegex(
            collapsed,
            r"does not transfer to another repository, another skill, a later "
            r"invocation, or a later session",
        )
        self.assertRegex(
            collapsed,
            r"Standing authorization from an autonomous caller such as `auto-pr` "
            r"does not cover this gate",
        )
        self.assertNotRegex(text, r"(?i)\bhw-native-sys\b")

    def test_auto_pr_contains_no_publication_or_repair_implementation(self) -> None:
        skill = ROOT / "skills/auto-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("../git-commit/SKILL.md", text)
        self.assertIn("../github-pr/SKILL.md", text)
        self.assertIn("../fix-pr/SKILL.md", text)
        self.assertIn("../../lib/repository/policy.md", text)
        self.assertNotRegex(text, r"(?m)^\s*(?:gh|git)\s+(?:pr|api|push|commit)\b")
        self.assertNotIn("resolveReviewThread", text)
        self.assertNotIn("reviewThreads", text)

    def test_github_pr_selects_create_branch_before_push_authority(self) -> None:
        text = (ROOT / "skills/github-pr/SKILL.md").read_text(encoding="utf-8")
        branch_selection = text.find("## Select and verify the working branch")
        authority_capture = text.find("## Capture push authority before commit work")
        self.assertGreaterEqual(branch_selection, 0)
        self.assertGreater(authority_capture, branch_selection)

    def test_github_pr_fails_closed_when_branch_state_commands_fail(self) -> None:
        source = bash_source(ROOT / "skills/github-pr/SKILL.md")
        self.assertIn(
            "WORKTREE_STATUS=$(git status --porcelain) || {",
            source,
        )
        self.assertIn(
            'COMMITS_AHEAD=$(git rev-list --count "$BASE_REF"..HEAD) || {',
            source,
        )
        self.assertNotRegex(source, r'\[ [^\n]*"\$\(git (?:status|rev-list)')

    def test_github_pr_supports_create_and_existing_pr_update_routes(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertNotIn("gh pr create", text)
        self.assertIn('"$PR_CONTEXT_HELPER" create', text)
        self.assertIn("gh pr edit", text)
        self.assertIn("existing pull request", text.lower())
        self.assertIn("DEFAULT_BRANCH", text)
        self.assertIn("PR_REPO", text)
        self.assertIn("--force-with-lease", text)
        self.assertIn("PR_ROUTE=$(printf '%s' \"$PR_LOOKUP_RESULT\"", text)
        self.assertIn('[ "$PR_ROUTE" = "create" ]', text)
        self.assertIn('[ "$PR_ROUTE" = "update" ]', text)

    def test_github_pr_delegates_repository_commit_policy(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("repository-local `git-commit` skill", text)
        self.assertNotRegex(text, r"\b(?:feat|fix|refactor|chore|docs|test)/")
        self.assertNotIn("## Testing\n- [ ]", text)

    def test_github_pr_is_host_aware_and_fork_safe(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn('GH_HOST="$GITHUB_HOST"', text)
        self.assertIn("../../lib/github/scripts/pr-context.sh", text)
        self.assertIn("HEAD_REPO=$LOCAL_REPO", text)
        self.assertIn("ROLE", text)
        self.assertIn("HEAD_REPO", text)
        self.assertIn("MAINTAINER_CHECKOUT_VERIFIED", text)

        guard = text.find('"$PR_CONTEXT_HELPER" guard-branch')
        commit = text.find("repository-local `git-commit` skill")
        self.assertGreaterEqual(guard, 0)
        self.assertGreater(commit, guard)

    def test_pull_request_lookup_uses_supported_host_pinned_rest_api(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        reference = ROOT / "lib/github/lookup-pr.md"
        helper = ROOT / "lib/github/scripts/pr-context.sh"
        for path in (skill, reference, helper):
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"missing required file: {path}")
        if not all(path.is_file() for path in (skill, reference, helper)):
            return

        skill_text = skill.read_text(encoding="utf-8")
        reference_text = reference.read_text(encoding="utf-8")
        helper_text = helper.read_text(encoding="utf-8")
        self.assertIn("../../lib/github/scripts/pr-context.sh", skill_text)
        self.assertIn("scripts/pr-context.sh", reference_text)
        for document_text in (skill_text, reference_text):
            self.assertIn(
                'gh api --hostname "$GITHUB_HOST" --method GET',
                document_text,
            )
            self.assertIn('"repos/$PR_REPO/pulls"', document_text)
            self.assertIn('-f "head=$HEAD_SELECTOR"', document_text)
            self.assertIn("--paginate --slurp", document_text)
            self.assertIn("separately with `jq -e 'add'`", document_text)
            self.assertNotIn("--slurp --jq", document_text)
        self.assertNotRegex(bash_source(skill), r"gh pr list[\s\S]{0,200}--head")
        self.assertNotRegex(reference_text, r"gh pr list[\s\S]{0,200}--head")
        self.assertIn('gh api --hostname "$GITHUB_HOST" --method GET', helper_text)
        self.assertIn('"repos/$PR_REPO/pulls"', helper_text)
        self.assertIn('-f "head=$HEAD_SELECTOR"', helper_text)
        self.assertIn("--paginate --slurp", helper_text)
        self.assertIn("| jq -ce '", helper_text)
        self.assertNotIn("--slurp --jq", helper_text)

    def test_pull_request_creation_uses_host_pinned_rest_post(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        helper = ROOT / "lib/github/scripts/pr-context.sh"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        self.assertTrue(helper.is_file(), f"missing required helper: {helper}")
        if not skill.is_file() or not helper.is_file():
            return

        skill_text = skill.read_text(encoding="utf-8")
        helper_text = helper.read_text(encoding="utf-8")
        self.assertNotIn("gh pr create", skill_text)
        self.assertIn('"$PR_CONTEXT_HELPER" create', skill_text)
        for text in (skill_text, helper_text):
            self.assertIn('gh api --hostname "$GITHUB_HOST" --method POST', text)
            self.assertIn('"repos/$PR_REPO/pulls"', text)
            self.assertIn('"head_repo=$HEAD_REPO_NAME"', text)
            self.assertIn(".html_url", text)

    def test_author_guard_checks_branch_and_repository_before_commit(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        guard = text.find('"$PR_CONTEXT_HELPER" guard-branch')
        commit = text.find("repository-local `git-commit` skill")
        self.assertGreaterEqual(guard, 0)
        self.assertGreater(commit, guard)
        guard_block = text[guard:commit]
        self.assertIn('"$CURRENT_BRANCH" "$PR_HEAD_BRANCH"', guard_block)
        self.assertIn('"$LOCAL_REPO" "$HEAD_REPO"', guard_block)

    def test_known_pr_number_is_validated_before_positional_gh_use(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        reference = ROOT / "lib/github/lookup-pr.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        self.assertTrue(reference.is_file(), f"missing required reference: {reference}")
        if not skill.is_file() or not reference.is_file():
            return

        skill_text = skill.read_text(encoding="utf-8")
        reference_text = reference.read_text(encoding="utf-8")
        self.assertIn('"$PR_CONTEXT_HELPER" validate-number "$PR_NUMBER"', skill_text)
        validation = reference_text.find(
            '"$PR_LOOKUP_HELPER" validate-number "$PR_NUMBER"'
        )
        positional_use = reference_text.find('gh pr view "$PR_NUMBER"')
        self.assertGreaterEqual(validation, 0)
        self.assertGreater(positional_use, validation)

    def test_github_pr_derives_title_and_body_from_pr_commit_range(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count('"$BASE_REF"..HEAD'), 2)
        self.assertNotIn("Brief description of changes", text)
        self.assertNotIn("Key change 1", text)

    def test_fix_pr_uses_all_shared_workflow_references(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        expected_links = (
            "../../lib/github/setup.md",
            "../../lib/github/lookup-pr.md",
            "../../lib/github/fetch-comments.md",
            "../../lib/github/detect-permission.md",
            "../../lib/github/checkout-fork-branch.md",
            "../../lib/github/commit-and-push.md",
            "../../lib/github/reply-and-resolve.md",
            "../../lib/github/common-issues.md",
        )
        for link in expected_links:
            with self.subTest(link=link):
                self.assertIn(link, text)

    def test_fix_pr_fetches_every_feedback_surface_and_page(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        for surface in ("reviewThreads", "reviews", "comments"):
            with self.subTest(surface=surface):
                self.assertIn(surface, text)
        self.assertIn("hasNextPage", text)
        self.assertIn("endCursor", text)
        self.assertIn("nested `comments`", text)
        self.assertIn("handled ledger", text)

    def test_fix_pr_waits_for_pending_ci_and_supports_external_checks(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("Pending checks are not clean", text)
        self.assertIn("whole-run logs", text)
        self.assertIn("completed", text)
        self.assertIn("external check", text.lower())
        self.assertIn("details URL", text)

    def test_fix_pr_is_host_aware_and_selects_a_verified_write_path(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn('GH_HOST="$GITHUB_HOST"', text)
        permission = text.find("../../lib/github/detect-permission.md")
        checkout = text.find("../../lib/github/checkout-fork-branch.md")
        commit = text.find("../../lib/github/commit-and-push.md")
        self.assertGreaterEqual(permission, 0)
        self.assertGreater(checkout, permission)
        self.assertGreater(commit, checkout)
        for role in ("owner", "fork", "maintainer"):
            with self.subTest(role=role):
                self.assertIn(f"`{role}`", text)

    def test_fix_pr_requires_confirmation_before_scoped_fixes(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        findings = text.find("## Classify and present findings")
        confirmation = text.find("## Explicit confirmation gate")
        fixes = text.find("## Apply selected fixes")
        self.assertGreaterEqual(findings, 0)
        self.assertGreater(confirmation, findings)
        self.assertGreater(fixes, confirmation)
        self.assertIn("fix immediately", text)

    def test_github_pr_composes_a_reviewer_ready_description(self) -> None:
        skill = ROOT / "skills/github-pr/SKILL.md"
        reference = ROOT / "lib/github/pr-description.md"
        for path in (skill, reference):
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"missing required file: {path}")
        if not skill.is_file() or not reference.is_file():
            return

        skill_text = skill.read_text(encoding="utf-8")
        self.assertIn("../../lib/github/pr-description.md", skill_text)
        self.assertNotIn("PR_BODY=$(git log", skill_text)

        reference_text = reference.read_text(encoding="utf-8")
        for required in ("## Summary", "## Changes", "## Verification"):
            with self.subTest(section=required):
                self.assertIn(required, reference_text)
        self.assertIn('[ "$PR_BODY" = "$PR_TITLE" ]', reference_text)
        self.assertRegex(reference_text, r"only a list of commit subjects")
        self.assertRegex(reference_text, r"Never invent a\s+motivation")
        self.assertIn("no generated-by branding", reference_text)
        self.assertIn("Preserve an existing description", reference_text)

    def test_publication_skips_interactive_branch_and_commit_approval(self) -> None:
        auto_pr = (ROOT / "skills/auto-pr/SKILL.md").read_text(encoding="utf-8")
        self.assertRegex(
            auto_pr,
            r"standing authorization from this explicit `auto-pr`\s+invocation "
            r"covering branch naming, the commit set and its message, and the\s+"
            r"pull-request description",
        )
        self.assertRegex(
            auto_pr,
            r"never pause to have the\s+user approve a branch name, a commit "
            r"count, or a description",
        )
        self.assertIn("Standing authorization covers nothing", auto_pr)

        branch_naming = (ROOT / "lib/github/branch-naming.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("## Approve the summary", branch_naming)
        self.assertRegex(branch_naming, r"direct invocation needs explicit user")
        self.assertRegex(branch_naming, r"use it without asking")

        git_commit = (ROOT / "skills/git-commit/SKILL.md").read_text(encoding="utf-8")
        self.assertRegex(git_commit, r"the preview is a record rather than a prompt")
        self.assertRegex(
            git_commit,
            r"without waiting for approval of the authorized paths,\s+"
            r"the commit count, or the message",
        )
        self.assertIn("stop rule still applies unchanged", git_commit)

    def test_waiting_on_checks_never_substitutes_for_a_feedback_recheck(self) -> None:
        fix_pr = (ROOT / "skills/fix-pr/SKILL.md").read_text(encoding="utf-8")
        recheck = fix_pr.find("## Recheck and bound the loop")
        self.assertGreaterEqual(recheck, 0)
        recheck_block = fix_pr[recheck:]
        self.assertIn("Waiting is not\nrechecking", recheck_block)
        self.assertIn("every poll re-runs that same", recheck_block)
        self.assertIn("checks-only poll is never a recheck", recheck_block)
        self.assertRegex(
            recheck_block,
            r"Green checks\s+alone are not terminal.*re-fetch every feedback surface "
            r"when the last check\s+completes",
        )

        fetch = (ROOT / "lib/github/fetch-comments.md").read_text(encoding="utf-8")
        self.assertIn("## An early fetch is provisional", fetch)
        self.assertIn("Reviewers post asynchronously", fetch)
        self.assertRegex(fetch, r'means "nothing yet", never "no\s+feedback"')
        self.assertRegex(fetch, r"Re-fetch all three surfaces")
        self.assertIn("settling rule", fix_pr)

        auto_pr = (ROOT / "skills/auto-pr/SKILL.md").read_text(encoding="utf-8")
        inventory = auto_pr[auto_pr.find("1. Delegate one read-only inventory") :]
        self.assertRegex(
            inventory,
            r"inline threads, review bodies, conversation\s+comments, and check "
            r"states together",
        )
        self.assertRegex(
            inventory,
            r"Any wait between\s+iterations re-runs that whole inventory on each "
            r"poll, never check states\s+alone",
        )
        self.assertRegex(
            auto_pr,
            r"Green\s+checks alone do not establish this; require a full feedback "
            r"inventory taken\s+after the last check completed",
        )

    def test_fix_pr_auto_pr_authorization_is_narrow_and_fail_closed(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        findings = text.find("## Classify and present findings")
        composed = text.find("## Validate auto-pr composed authorization")
        confirmation = text.find("## Explicit confirmation gate")
        self.assertGreater(composed, findings)
        self.assertGreater(confirmation, composed)
        gate = text[confirmation : text.find("## Apply selected fixes")]
        self.assertIn("Do not edit until the user confirms", gate)
        self.assertIn("every direct invocation", text)
        self.assertRegex(text, r"only when the active caller is `auto-pr`")
        for requirement in (
            "active caller is `auto-pr`",
            "exact host, repository, number, and head",
            "unchanged numbered inventory entry and stable finding ID",
            "`ci-objective`, `correctness`, or `style-policy`",
            "successful guard iteration and attempt evidence",
            "standing authorization from an explicit `auto-pr` invocation",
            "independently revalidate",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, text)
        self.assertIn("fall back to the explicit confirmation gate", text)
        self.assertIn("unknown or deferred kind", text)
        for mismatch in (
            "identity or head mismatch",
            "inventory entry or stable finding ID mismatch",
            "kind or classification mismatch",
            "guard or ledger mismatch",
        ):
            with self.subTest(mismatch=mismatch):
                self.assertIn(mismatch, text)
        self.assertIn("do not auto-repair", text)
        self.assertIn("without incrementing it", text)
        self.assertRegex(text, r"scope\s+growth")
        self.assertIn("same stable key", text)

    def test_fix_pr_delegates_repository_policy(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("repository-local instructions", text)
        self.assertIn("repository-local testing skill", text)
        self.assertIn("repository-local `git-commit` skill", text)
        self.assertNotRegex(text, r"\b(?:pytest|cargo test|npm test)\b")
        self.assertNotIn('git commit -m "fix(pr)', text)

    def test_fix_pr_folds_commits_and_pushes_with_shared_safety(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("fixup", text)
        self.assertIn("autosquash", text)
        self.assertIn("PR-owned commit", text)
        self.assertIn("--force-with-lease", text)

    def test_fix_pr_replies_then_resolves_only_verified_fixes(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        verified = text.find("Verify the selected fixes")
        reply = text.find("Reply first")
        resolve = text.find("Resolve second")
        self.assertGreaterEqual(verified, 0)
        self.assertGreater(reply, verified)
        self.assertGreater(resolve, reply)
        self.assertIn("isResolved", text)

    def test_fix_pr_requires_verified_resolution_before_completion(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        reference = ROOT / "lib/github/reply-and-resolve.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        self.assertTrue(reference.is_file(), f"missing required reference: {reference}")
        if not skill.is_file() or not reference.is_file():
            return

        skill_text = skill.read_text(encoding="utf-8")
        reference_text = reference.read_text(encoding="utf-8")
        self.assertIn("fully paginate", skill_text)
        self.assertIn("isResolved=true", skill_text)
        self.assertIn("report the workflow incomplete", skill_text)
        self.assertIn("never the iteration, task, or PR complete", skill_text)
        self.assertIn("mutation response alone is not final", reference_text)
        self.assertIn("every recorded ID", reference_text)
        self.assertIn("isResolved=true", reference_text)
        self.assertIn("Never report the iteration or task as", reference_text)

    def test_fix_pr_rechecks_with_iteration_and_stuck_bounds(self) -> None:
        skill = ROOT / "skills/fix-pr/SKILL.md"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("maximum of 5 iterations", text)
        self.assertIn("same fingerprint", text)
        self.assertIn("final recheck", text)
        self.assertIn("blocker", text)

    def test_clean_branches_uses_portable_safe_deletion_contract(self) -> None:
        skill = ROOT / "skills/clean-branches/SKILL.md"
        helper = ROOT / "skills/clean-branches/scripts/clean-branches.sh"
        self.assertTrue(skill.is_file(), f"missing required skill: {skill}")
        self.assertTrue(helper.is_file(), f"missing required helper: {helper}")
        if not skill.is_file():
            return

        text = skill.read_text(encoding="utf-8")
        self.assertIn("../../lib/github/setup.md", text)
        self.assertIn("scripts/clean-branches.sh", text)
        self.assertIn("DEFAULT_BRANCH", text)
        self.assertIn("headRefOid", text)
        self.assertIn("Approved OID", text)
        self.assertIn("Never delete from `$BASE_REMOTE`", text)

        approval = re.search(r"(?im)^## Explicit approval gate$", text)
        self.assertIsNotNone(approval)
        if approval is None:
            return

        destructive_commands = (
            '"$CLEAN_BRANCHES_HELPER" delete-local',
            '"$CLEAN_BRANCHES_HELPER" delete-remote',
        )
        for command in destructive_commands:
            with self.subTest(command=command):
                command_position = text.find(command)
                self.assertGreaterEqual(command_position, 0)
                self.assertLess(approval.start(), command_position)


if __name__ == "__main__":
    unittest.main()
