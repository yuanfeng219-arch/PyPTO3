---
name: github-pr
description: Use when creating or updating a GitHub pull request from committed or uncommitted local work, including fork, maintainer-edit, and GitHub Enterprise repositories.
---

# GitHub Pull Request

Never run the full system-test suite locally. Run only system-test cases directly relevant to the changed or requested scope; use CI for the full suite.
If CI cannot run it, report the limitation instead of substituting a local full-suite run.

## Establish context and choose a route

Resolve the [repository scope gate](../../lib/repository/scope.md) before the
setup fetches below and before any push or pull-request write. In a foreign
repository, warn with the intended push target and wait for confirmation.

Read and run [GitHub workflow setup](../../lib/github/setup.md); preserve its
context and use the discovered host for every GitHub CLI command:

```bash
export GH_HOST="$GITHUB_HOST"
```

Resolve [PR context](../../lib/github/scripts/pr-context.sh) from this loaded
skill, never `REPO_ROOT`/CWD, then pass it to [lookup](../../lib/github/lookup-pr.md):

```bash
if [ -n "${PR_NUMBER:-}" ]; then
  "$PR_CONTEXT_HELPER" validate-number "$PR_NUMBER" >/dev/null || exit 1
  PR_LOOKUP_ALLOW_NONE=false
else
  PR_LOOKUP_ALLOW_NONE=true
fi
PR_LOOKUP_HELPER="$PR_CONTEXT_HELPER"
```

Run lookup's host-pinned, exact fork-owner/head query:

```text
gh api --hostname "$GITHUB_HOST" --method GET \
  "repos/$PR_REPO/pulls" -f state=open -f "head=$HEAD_SELECTOR" \
  -f per_page=100 --paginate --slurp
# Aggregate pages separately with `jq -e 'add'`; `gh api` rejects combining them.
```

```bash
PR_ROUTE=$(printf '%s' "$PR_LOOKUP_RESULT" | jq -r '.route')
case "$PR_ROUTE" in
  create|update) ;;
  *) echo "Error: unsupported pull-request route: $PR_ROUTE" >&2; exit 1 ;;
esac
```

An existing pull request always yields `update`, never an early successful exit.

## Verify an existing PR and its writable head

For the update route, read and run [permission
detection](../../lib/github/detect-permission.md). This distinguishes the
author (`ROLE=owner` or `ROLE=fork`) from a maintainer and verifies permission
on `HEAD_REPO`.

Immediately after permission detection, guard the verified head before any
repository-local commit workflow:

```bash
if [ "$PR_ROUTE" = "update" ]; then
  "$PR_CONTEXT_HELPER" guard-branch \
    "$ROLE" "$CURRENT_BRANCH" "$PR_HEAD_BRANCH" \
    "$LOCAL_REPO" "$HEAD_REPO" || exit 1
fi
```

For an owner or fork, either a branch mismatch or a head/local repository
identity mismatch stops here, so dirty changes cannot be committed to the
wrong head.

When `ROLE=maintainer` and the PR head is not the current local branch, require
a clean worktree, then read and run [cross-fork
checkout](../../lib/github/checkout-fork-branch.md). Do not construct a
same-named local branch or push directly to an unverified fork. That reference
sets `WORK_BRANCH`, `PR_HEAD_BRANCH`, and `MAINTAINER_CHECKOUT_VERIFIED` for the
shared push workflow.

## Select and verify the working branch

For the update route, do not enter this section until `guard-branch` has
succeeded. There is no shared dirty-worktree or commit step before that
route-specific branch-and-repository identity gate.

```bash
WORKTREE_STATUS=$(git status --porcelain) || {
  echo "Error: failed to inspect worktree status" >&2
  exit 1
}
COMMITS_AHEAD=$(git rev-list --count "$BASE_REF"..HEAD) || {
  echo "Error: failed to inspect commits ahead of base" >&2
  exit 1
}
```

Rerun this exact block after commit. Never put either substitution inside `test` or `[ ]`; that hides the Git command's failure status.

For the create route, if `CURRENT_BRANCH` equals `DEFAULT_BRANCH`, or has no
commits ahead but has uncommitted work, obtain `BRANCH_SUMMARY` and any
repository-required `BRANCH_PREFIX`, then read and run [branch
naming](../../lib/github/branch-naming.md), which defines when that summary
needs explicit approval. Never invent a prefix. Only after its checkout
succeeds, set `PR_HEAD_BRANCH="$CURRENT_BRANCH"` and `HEAD_REPO="$LOCAL_REPO"`.

## Capture push authority before commit work

Read [prepare, validate, and push](../../lib/github/commit-and-push.md) and
resolve all three trusted helpers from that reference, never `REPO_ROOT`/CWD. For
`create`, this follows branch checkout; for `update`, it follows the verified
head checkout. Enter its single-use transaction immediately before commit
work. Pass `GITHUB_HOST`/`PR_REPO` as the expected base identity and the
verified head host/repository, remote, and branch as the head authority.

## Commit intentionally

Use the repository-local `git-commit` skill for review, staging, message
syntax, and commit shape. If unavailable, ask; do not invent a workflow.
Define the transaction mutation with trusted edits and Git built-ins only.
The transaction disables repository-configured hooks; run repository code only
through the validation runner, never inside the mutation callback.

Give the bundled validation runner the focused and broader command. It runs
repository-selected checks in the working checkout at exactly
`PREPARED_HEAD_OID`, so Git metadata, submodule contents, toolchain, and
network stay available; harness permission controls govern that execution. It
refuses a mismatched `HEAD` or a dirty worktree, and fails when validation
leaves artifacts behind. Shared `prepare` derives rewrite state from the fresh
remote OID, and `push` uses explicit `--force-with-lease` when needed.

One function/subshell invocation is one transaction, with readonly authority
scoped inside it. A failed runner, changed `HEAD`, conflict, or later fix
iteration must start a fresh transaction and recapture all identities, OIDs,
and rewrite state. Never reuse a lease or prepared value. Resolve conflicts
under repository policy or `git rebase --abort`; never destructively reset work.

## Create or update the pull request

Derive title and body only from the post-rebase PR commit range. `PR_RANGE`
below is the message evidence for [PR
description](../../lib/github/pr-description.md), which composes and validates
both and owns `PR_BODY`; a bare commit-subject list is not a description.

```bash
PR_RANGE=$(git log --reverse --format='%H%n%s%n%b' "$BASE_REF"..HEAD)
PR_TITLE=$(git log --reverse --format='%s' "$BASE_REF"..HEAD | sed -n '1p')
if [ -z "$PR_RANGE" ] || [ -z "$PR_TITLE" ]; then
  echo "Error: the pull-request commit range is empty" >&2
  exit 1
fi
```

Create through host-pinned REST. GitHub's [official create contract](https://docs.github.com/en/rest/pulls/pulls#create-a-pull-request)
requires `head_repo` to be the head repository name for same-organization
cross-repository PRs. The helper derives it only from verified `HEAD_REPO`.

```text
gh api --hostname "$GITHUB_HOST" --method POST \
  "repos/$PR_REPO/pulls" -f "title=$PR_TITLE" -f "body=$PR_BODY" \
  -f "head=$HEAD_SELECTOR" -f "base=$BASE_BRANCH"
# Same-owner cross-repository only:
-f "head_repo=$HEAD_REPO_NAME"
```

```bash
if [ "$PR_ROUTE" = "create" ]; then
  HEAD_REPO=$LOCAL_REPO
  PR_URL=$("$PR_CONTEXT_HELPER" create \
    "$GITHUB_HOST" "$PR_REPO" "$HEAD_REPO" "$CURRENT_BRANCH" \
    "$DEFAULT_BRANCH" "$PR_TITLE" "$PR_BODY") || exit 1
fi
```

The helper returns only a validated non-empty `.html_url`; failures stop.

Update the existing PR rather than abandoning it:

```bash
if [ "$PR_ROUTE" = "update" ]; then
  GH_HOST="$GITHUB_HOST" gh pr edit "$PR_NUMBER" \
    --repo "$PR_REPO" \
    --base "$DEFAULT_BRANCH" \
    --title "$PR_TITLE" \
    --body "$PR_BODY" || exit 1
  PR_URL=$(GH_HOST="$GITHUB_HOST" gh pr view "$PR_NUMBER" \
    --repo "$PR_REPO" --json url --jq '.url') || exit 1
fi
```

Do not add generated-by branding or issue-closing text that the commit range or
user did not supply.

## Report the result

Return the exact push target, whether the description was composed or
preserved, and the PR URL, number, state, base, and head:

```bash
GH_HOST="$GITHUB_HOST" gh pr view "${PR_NUMBER:-$PR_URL}" \
  --repo "$PR_REPO" \
  --json number,url,state,isDraft,baseRefName,headRefName
```
