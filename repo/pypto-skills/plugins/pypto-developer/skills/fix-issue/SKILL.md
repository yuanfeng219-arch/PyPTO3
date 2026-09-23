---
name: fix-issue
description: Use when inspecting, designing, or implementing work for a GitHub issue, especially from a fork checkout or when existing ownership, project status, or related pull requests may conflict.
---

# Fix Issue

Inspect first, design from repository evidence, and perform no ownership,
project, branch, or implementation mutation before explicit approval.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## Resolve context and run preflight

Resolve the [repository scope gate](../../lib/repository/scope.md) before any
ownership, project, branch, or implementation mutation. In a foreign
repository, warn with the intended mutation and wait for explicit confirmation.

Read the authoritative [repository policy](../../lib/repository/policy.md), then
read [issue context](../../lib/github/issue-context.md). Invoke the shared
[issue-context helper](../../lib/github/scripts/issue-context.sh) `repository`
route and retain `GITHUB_HOST`, `LOCAL_REPO`, `ISSUE_REPO`, and
`DEFAULT_BRANCH`. The issue belongs to `ISSUE_REPO`; preserve the validated
`LOCAL_REPO` fork as the contributor's writable repository.

Run `fix-preflight` with the discovered host, issue repository, and issue
number. Pass `--in-progress-status` only when repository policy supplies the
actual in-progress value. Capture and show the complete JSON and exit status:

| Status | Meaning | Action |
| --- | --- | --- |
| `0` | Open with no configured conflict | Continue read-only work |
| `20` | Closed | Stop; never override |
| `21` | Assigned | Report assignees; require explicit conflict approval |
| `22` | Configured in-progress state | Report status; require explicit conflict approval |
| `23` | Active linked pull request | Report links; require explicit conflict approval |

The primary exit is not the complete conflict set. Read the emitted `conflicts`
array and enumerate every entry before asking to proceed: `closed`, `assigned`,
`in_progress`, and `active_pull_request`. For each entry, show the matching
assignee logins, project item/status identity, or active pull-request number,
head repository, head branch, and state from the same JSON. Never infer from
exit `21` that assignment is the only conflict.

Do not run `--allow-conflict` merely because the user asked to fix the issue.
Use it only after showing the complete set and receiving explicit approval for
each conflict. Closed is non-overridable even when other conflicts are present.
The override only rechecks read-only state; it does not authorize mutations.

Shape every identity or conflict stop report with these four fields: issue
target (`GITHUB_HOST/ISSUE_REPO`), base (`DEFAULT_BRANCH`), later write target
(`LOCAL_REPO`), and gate (the exact approvals still missing). Mark unvalidated
user-supplied values as expected rather than established. State that the base
and write target remain unused until approval.

## Reproduce and inspect

Read the issue record, applicable instructions, relevant implementation,
history, tests, and documentation. Reproduce or validate the reported behavior
using only commands selected by repository policy. Do not invent a generic test
command or edit code while investigating. If a preflight conflict exists,
continue only with read-only inspection and keep every mutation blocked.

## Design the implementation

Present the root cause or evidence gap, affected paths, implementation shape,
regression coverage, repository-selected verification, documentation impact,
and meaningful alternatives. Separate established facts from assumptions.
Make the design concrete enough to approve before changing a file.

## Wait for approval

Wait for explicit approval of the implementation design. If preflight reported
any conflict, also require explicit approval for every identity in the emitted
`approval_envelope` and explicit approval for each conflict category. Design
approval alone does not resolve them. Preserve the
exact envelope as compact JSON; do not reconstruct it from the primary exit or
from the conflict category names.

Pass that canonical approval envelope to the authoritative override read:

```bash
"$ISSUE_CONTEXT_HELPER" fix-preflight "$GITHUB_HOST" "$ISSUE_REPO" \
  "$ISSUE_NUMBER" --in-progress-status "$REPOSITORY_IN_PROGRESS_VALUE" \
  --allow-conflict --approved-envelope-json "$APPROVED_ENVELOPE_JSON"
```

Omit `--in-progress-status` only when policy supplied no value. The helper
compares the newly read canonical payload with the approved JSON and returns
`0` only for an exact match whose primary conflict is `21`, `22`, or `23`. The
payload binds the host/repository/issue identity, issue state, configured status,
ordered conflict categories, assignee logins, matching project item/status
records, and every active pull-request number/head/state. Replacing an assignee,
project item, or active PR fails closed even when the category is unchanged.
Exit `20`, `21`, `22`, or `23` from this read is non-success: show the
authoritative new record and obtain fresh approval for every current identity.
A changed set that became clean also fails safely rather than returning `0`.
Stop unconditionally when `closed` is present or exit `20` is returned.

## Optionally claim and update status

Perform claim or status actions only when included in the approved design.
Self-assignment uses the discovered host and issue repository:

```bash
GH_HOST="$GITHUB_HOST" gh issue edit "$ISSUE_NUMBER" \
  --repo "$ISSUE_REPO" --add-assignee @me
```

Offer a project update only when repository policy explicitly identifies the
project and status field plus the desired value. Keep assignment and project
updates separate. If an optional project update fails, report that partial
result without treating the code fix as failed.

## Create the branch

Read [branch naming](../../lib/github/branch-naming.md). Refresh from the
validated `ISSUE_REPO` default branch using the remote proven to target it,
while preserving the writable `LOCAL_REPO` fork for later push. Set
`DEFAULT_BRANCH`, `CURRENT_BRANCH`, and the approved `BRANCH_SUMMARY`; set
`BRANCH_PREFIX` only when user or repository policy requires one. Apply the
shared contract after approval. Never invent a prefix, remote, or base branch.

## Implement the approved design

Edit only approved paths and preserve unrelated user changes. Follow applicable
repository instructions and repository-local development workflows. Add the
regression test before the fix when the repository requires test-driven work.
Update documentation when behavior changes.

## Run repository-selected verification

Run exactly the verification, review, formatting, and type-check commands
selected by repository policy for the changed scope. Do not substitute familiar
commands. Resolve failures before handoff or report a concrete blocker.

## Hand off the result

Summarize issue context, approved changes, optional mutation outcomes, and exact
verification evidence. Use the repository-local review, commit, and pull-request
workflows only when the user requested those follow-up actions. Preserve the
issue repository as the base and the contributor fork as the write target.
