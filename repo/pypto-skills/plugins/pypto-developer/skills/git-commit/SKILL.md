---
name: git-commit
description: Use when creating a Git commit or preparing authorized repository changes for commit.
---

# Git Commit

Create one verified commit from exactly the task-owned change. Treat existing
worktree and index state as user-owned until inspection establishes otherwise.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## Load the shared contracts

Read [repository policy](../../lib/repository/policy.md) before choosing tests,
review steps, hooks, documentation work, or a message. Use the
[exact-path staging helper](../../lib/repository/scripts/stage-owned.sh) for the
only staging mutation.

Resolve the [repository scope gate](../../lib/repository/scope.md) before
staging or committing. In a foreign repository, warn with the intended commit
and wait for explicit confirmation; standing authorization does not cover it.

## Inspect and establish ownership

1. Resolve the repository root and inspect the branch, status, unstaged diff,
   staged diff, and untracked names.
2. Resolve applicable nested instruction files for every changed path,
   including unrelated paths that will remain untouched.
3. Build an explicit list of authorized repository-root-relative paths. Inspect
   unrelated changes without editing or staging them.
4. Review each authorized diff for correctness, scope, secrets, generated
   artifacts, and repository-required documentation.

If ownership cannot be separated at file or hunk granularity, stop for user
direction. A whole file is not authorized merely because one of its hunks is.

## Run repository-selected verification

Run the exact verification, review, hook, or documentation commands selected by
the repository policy. Do not invent a command from a source repository or from
familiar tooling. Record each command, exit status, and concise result.

If verification modifies files, inspect them again, establish ownership, and
rerun the applicable checks. Stop on a failure or on ambiguous policy; do not
bypass hooks.

## Derive and preview the commit

Derive the subject, body, and trailers from repository policy and unambiguous history.
Do not assume Conventional Commits, a fixed type list, a trailer, or a subject
shape. When the subject alone cannot carry why the change is needed or what
changes for a caller, state both in the message body: publication derives the
pull-request description from these messages.

Immediately before staging, show this complete preview:

```text
Authorized paths (exact repository-root-relative paths):
- <path>

Verification results:
- <exact command>: <exit status and result>

Complete commit message:
<subject, body, and trailers exactly as they will be committed>
```

Keep the complete commit message in a file whose bytes match the preview. Stop
if any path, result, or message detail is unresolved.

For a caller that supplies standing authorization from an explicit autonomous
invocation, such as `auto-pr`, the preview is a record rather than a prompt:
publish it and continue without waiting for approval of the authorized paths,
the commit count, or the message. Every ownership, policy, and verification
stop rule still applies unchanged.

## Stage the exact paths

Invoke `stage-owned.sh PATH...` once with every previewed path as a separate,
quoted argument. Confirm its printed staged names equal the authorized list,
then inspect `git diff --cached --check`, the complete cached diff, and
`git diff --cached --name-only`.

Never use broad staging (`git add -A`, `git add --all`, `git add .`, wildcards,
directories, or pathspec magic). If the helper reports an unrelated pre-staged
path or a set mismatch, stop without committing.

## Commit and verify

Commit from the unchanged previewed message file. Do not amend an existing
commit unless the user explicitly authorized that distinct operation.

Verify with `git show --format=fuller --name-status HEAD`, inspect the exact
message, and compare the committed paths with the authorized list. Finally,
inspect status to confirm unrelated tracked and untracked changes remain
untouched. Report the commit identifier, subject, verification evidence,
committed paths, and remaining unrelated state.

Apply the shared policy's ambiguity and ownership stop rules throughout this
workflow.
