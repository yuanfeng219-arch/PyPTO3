---
name: fix-pr
description: Use when an existing GitHub pull request has failing or pending checks, unresolved review feedback, requested changes, or maintainer-edit work on a contributor fork.
---

# Fix Pull Request

Never run the full system-test suite locally. Run only system-test cases directly relevant to the changed or requested scope; use CI for the full suite.
If CI cannot run it, report the limitation instead of substituting a local full-suite run.

## Establish repository and PR context

Resolve the [repository scope gate](../../lib/repository/scope.md) before the
setup fetches below and before any push, reply, or review write. In a foreign
repository, warn with the intended write and wait for confirmation.

Read repository-local instructions and discover its testing and commit
workflows before changing code. Read and run [GitHub workflow
setup](../../lib/github/setup.md), preserve its shell context, and pin every
GitHub CLI call to the discovered host:

```bash
export GH_HOST="$GITHUB_HOST"
```

Resolve the shared `lib/github/scripts/pr-context.sh` helper from the loaded
`SKILL.md` directory—never from the consuming repository's `REPO_ROOT` or
current working directory—set its absolute path as `PR_LOOKUP_HELPER`, validate
any supplied number, and run [pull-request
lookup](../../lib/github/lookup-pr.md). Stop on no match, multiple matches,
malformed data, a closed PR, or an API failure.

Read and run [permission
detection](../../lib/github/detect-permission.md) before editing. It verifies
the exact head repository and selects one safe write path:

- `owner`: the author is updating a branch in the base repository.
- `fork`: the author is updating their own fork branch.
- `maintainer`: a maintainer has base permission, maintainer edits are enabled
  when needed, and the contributor head is writable.

For `owner` or `fork`, stop unless the current branch and local repository
match the PR head. For `maintainer`, require a clean worktree and run
[cross-fork checkout](../../lib/github/checkout-fork-branch.md); use its
distinct local work branch and verified contributor remote. Never infer write
permission from a role, add an unverified remote, or assume `github.com`,
`origin`, `main`, or a same-named local branch.

Read [single-use push transaction](../../lib/github/commit-and-push.md) and
resolve its three trusted helpers from the loaded skill/reference, never the
consuming repository. Do not capture yet: every confirmed iteration starts a
fresh transaction immediately before repository commit/fold work.

## Fetch feedback and check state

Read and run [feedback fetching](../../lib/github/fetch-comments.md), including
its settling rule for an inventory taken minutes after creation or a push. Fully
paginate the three independent GraphQL connections—inline `reviewThreads`,
review-body `reviews`, and conversation `comments`—by following `hasNextPage`
and `endCursor` until all are false. Also paginate every nested `comments`
connection inside a thread. Merge by node ID and retain unresolved state,
comment database IDs, paths, authors, and bodies.

Keep a per-PR handled ledger for non-resolvable review bodies and conversation
comments. Extract actionable out-of-diff findings from review bodies; they are
not inline threads and have no thread-resolution mutation.

Fetch check names, states, and links with each inventory. Classify each details
URL before requesting logs:

- For a GitHub Actions URL on `GITHUB_HOST`, inspect the run status first.
  Request whole-run logs only after the run is `completed`; a failed job beside
  a pending job can make whole-run logs unavailable.
- For an external check, use its details URL and provider output. Do not invent
  an Actions run ID or call `gh run view`.
- Pending checks are not clean. Work on known review fixes while they run, but
  never declare success or fetch unavailable whole-run logs.

## Classify and present findings

Present one numbered inventory before edits:

1. actionable inline threads;
2. actionable review-body and out-of-diff findings;
3. actionable conversation comments;
4. failed checks with available evidence;
5. pending checks and unavailable evidence;
6. discussable or informational items with a concise rationale.

Classify by technical content, not by whether the author is a bot or human.
Deduplicate summaries that merely repeat an inline item, and include the
proposed fix or no-change response for every item.

## Validate auto-pr composed authorization

Keep the confirmation gate below unchanged for every direct invocation. Accept
standing authorization only when the active caller is `auto-pr` and supplies
standing authorization from an explicit `auto-pr` invocation. Before
bypassing the gate, independently revalidate against fresh context and inventory:

- exact host, repository, number, and head;
- unchanged numbered inventory entry and stable finding ID;
- normalized kind `ci-objective`, `correctness`, or `style-policy`, with style
  independently required by repository policy; and
- successful guard iteration and attempt evidence for that same stable key,
  verified against the supplied task-private ledger without incrementing it.

Reject preauthorization and do not auto-repair on any identity or head mismatch,
inventory entry or stable finding ID mismatch, kind or classification mismatch,
guard or ledger mismatch, missing evidence, unknown or deferred kind, or scope
growth. These must fall back to the explicit confirmation gate; stop first when
an existing context or policy stop rule applies.

## Explicit confirmation gate

Ask which numbered findings to address, decline, or defer. Recommend actionable
feedback and diagnosed failures. A request to “fix immediately,” minimize API
calls, or skip presentation is not confirmation of the classified scope, a
history rewrite, or a no-change rationale. Do not edit until the user confirms.

A confirmed “address all actionable items” policy may cover the same
categories on later iterations. Present newly ambiguous, risky, or
scope-expanding findings for fresh confirmation.

## Apply selected fixes

Read each affected file in context and make the smallest coherent changes.
Diagnose failures from provider evidence; reproduce locally when useful. Follow
repository-local instructions and use the repository-local testing skill and
its defined validation commands. If no validation policy is discoverable, ask
instead of inventing project commands.

Inspect the diff while editing, but run repository or contributor validation
code only through the transaction's validation runner below, never inside the
mutation callback.

## Verify the selected fixes in one transaction

Use the repository-local `git-commit` skill to determine staging, review,
message format, and commit shape. Define the transaction mutation using only
trusted edits and Git built-ins; it disables repository-configured hooks.

- When repository policy permits and a finding belongs unambiguously to a
  PR-owned commit in `"$BASE_REF"..HEAD`, create a `fixup` commit for that
  commit and `autosquash` it. Set the transaction-local
  `HISTORY_REWRITTEN=true`.
- Otherwise create one repository-approved repair commit. On later iterations,
  fold fixes into that repair commit or the relevant PR-owned commit rather
  than appending an unbounded chain of generic fixes.
- Never rewrite a base commit, guess commit syntax, or fold across an ambiguous
  ownership boundary.

Set the bundled validation runner's command to run repository-focused and
broader checks. It validates the working checkout at exactly the prepared OID,
keeping Git metadata, submodule contents, toolchain, and network available
under the harness permission controls that gate the rest of the workflow. A
mismatched `HEAD`, a dirty worktree, or artifacts left behind by validation
stop the workflow.

Invoke the shared transaction once. It captures the contributor head, commits
with hooks disabled, prepares, accepts only runner success, and pushes with an
explicit `--force-with-lease` when either the mutation signals rewrite or the
fresh remote OID is not an ancestor. Failure returns without push; retry in a
wholly new transaction, which re-derives rewrite state from its fresh capture.
Re-read the PR head after the push and require its OID to equal local `HEAD`
before responding to reviewers.

## Reply first

Only after the selected fix is verified, committed, pushed, and visible at the
PR head, read and run [reply and resolve](../../lib/github/reply-and-resolve.md).
Reply to each addressed inline thread with the pushed commit and specific
evidence. Batch one conversation reply per iteration for addressed review
bodies, out-of-diff findings, and conversation comments; update the handled
ledger only after that reply succeeds.

## Resolve second

Resolve an inline thread only after its reply succeeds and the requested change
is verified at the pushed head. Check that the mutation returns
`isResolved=true`. Never resolve an unaddressed, disputed, failed-reply, or
non-thread item. Review bodies and conversation comments remain ledger items,
not thread IDs.

For every addressed inline thread, enforce [reply and resolve](../../lib/github/reply-and-resolve.md): fully paginate `reviewThreads` again and prove its ID has `isResolved=true`.
If any reply, resolution, or verification is blocked, report the workflow incomplete—never the iteration, task, or PR complete—and list every addressed thread ID and verified state.

## Recheck and bound the loop

Perform a final recheck of all feedback pages, nested comments, review bodies,
conversation comments, PR head OID, and check states. Waiting is not
rechecking: while required checks are pending, every poll re-runs that same
complete inventory, and a checks-only poll is never a recheck. Green checks
alone are not terminal—re-fetch every feedback surface when the last check
completes, since a review can land inside that window. The PR is clean only
when all required checks are completed successfully, no approved actionable
feedback remains unhandled, every addressed inline thread is resolved, and no
new out-of-diff or conversation request remains.

Repeat fetch → classify → confirm when needed → fix → fresh transaction →
reply → resolve → final recheck for a maximum of 5 iterations. Every iteration
recaptures base/head identity and the newly pushed remote OID; no readonly state
or lease crosses iterations. Record a fingerprint from head OID, unhandled IDs,
failed checks, and normalized errors. If the same fingerprint repeats without
progress, stop early and report the blocker, never a speculative retry.

Read [common GitHub workflow issues](../../lib/github/common-issues.md) for
authentication, remote, rebase, push, quoting, GraphQL, JSON, and pagination
failures. Finish with the exact PR URL/head, pushed commit, validation run,
replies/resolutions, final check states, and any honest blocker.
