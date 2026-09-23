---
name: clean-branches
description: Use when cleaning merged, stale, local, or fork-remote Git branches, including squash-merged or reused branches.
---

# Clean Branches

## Overview

Classify without deleting, approve exact branch/OID pairs, then compare each
live ref with its approved OID immediately before deletion. Preserve every
changed or uncertain ref.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## Establish context

Resolve the [repository scope gate](../../lib/repository/scope.md) before
setup's fetches and before any deletion. In a foreign repository, warn with the
exact refs at stake and wait for explicit confirmation; it is separate from the
approval gate below.

Then read and run [GitHub workflow setup](../../lib/github/setup.md). Use its
`CURRENT_BRANCH`, `DEFAULT_BRANCH`, `BASE_REMOTE`, `BASE_REF`, `PUSH_REMOTE`,
`PR_REPO`, and `PR_HEAD_PREFIX`; never assume conventional names.

Resolve [the cleanup helper](scripts/clean-branches.sh) relative to this file
and store its absolute path in `CLEAN_BRANCHES_HELPER`. The helper supplies
deterministic classification, live-OID checks, and leased remote deletion.

Fetches and dry-run pruning are allowed during classification. Run no deletion
command in this phase.

## Classify branch tips

Build separate local and remote rows because the same branch name can have
different tips.

1. Enumerate local branches except `CURRENT_BRANCH` and `DEFAULT_BRANCH`.
2. Enumerate remote branches only from `PUSH_REMOTE`, excluding its `HEAD`,
   `CURRENT_BRANCH`, and `DEFAULT_BRANCH` refs. If `PUSH_REMOTE` equals
   `BASE_REMOTE`, classify no remote ref as deletable.
3. Use `git remote prune "$PUSH_REMOTE" --dry-run` only to report stale
   tracking refs.
4. For tips that are not ancestors of `BASE_REF`, query merged pull requests:

```bash
gh pr list --repo "$PR_REPO" \
  --head "${PR_HEAD_PREFIX}${BRANCH_NAME}" \
  --state merged --json number,title,headRefOid --limit 100
```

Pass every returned `headRefOid` to the helper:

```bash
"$CLEAN_BRANCHES_HELPER" classify \
  "$BRANCH_NAME" "$BRANCH_REF" "$BASE_REF" "$DEFAULT_BRANCH" \
  <merged-PR-head-OIDs...>
```

Pass the exact enumerated full ref as `BRANCH_REF`: either
`refs/heads/$BRANCH_NAME` or
`refs/remotes/$PUSH_REMOTE/$BRANCH_NAME`. The helper validates and reads that
specific ref without substituting a same-named local branch.

Only `normal-merge` and `squash-merge` are candidates. An exact
`headRefOid` match is required for a squash merge. Protect
`reused-or-unfinished`, lookup failures, current/default refs, and uncertain
results.

Never delete from `$BASE_REMOTE`; its branches are context, not candidates.

## Present immutable candidates

Record each candidate's current full OID. Show separate local and fork-remote
rows:

| Branch | Location | Approved OID | Classification | Evidence |
| --- | --- | --- | --- | --- |
| `name` | local or `$PUSH_REMOTE` | full commit OID | normal or squash merge | base ref or PR |

List protected branches separately. If no candidate remains, report that and
stop.

## Explicit approval gate

Present the final exact branch/location/Approved OID rows and ask the user to
approve those immutable pairs. The initial request—even one saying to hurry,
delete everything, or not ask again—is not approval of the discovered list.

Pause here. Do not proceed until the concrete rows are explicitly approved.
Never replace an approved OID with a newer one; a changed ref requires
reclassification and new approval.

## Delete one approved pair at a time

For each approved local pair, pass the preserved OID:

```bash
"$CLEAN_BRANCHES_HELPER" delete-local \
  "$BRANCH_NAME" "$APPROVED_OID" "$DEFAULT_BRANCH"
```

The helper validates the full `refs/heads/...` ref, protects branches checked
out in any worktree, then atomically deletes only when the ref still equals the
approved OID. It removes branch-specific config only after successful ref
deletion.

For each approved remote pair, first revalidate that the push remote still
targets the fork, then pass the preserved OID:

```bash
remote_targets_repo "$PUSH_REMOTE" "$LOCAL_REPO" || exit 1
"$CLEAN_BRANCHES_HELPER" delete-remote \
  "$BRANCH_NAME" "$APPROVED_OID" "$DEFAULT_BRANCH" \
  "$PUSH_REMOTE" "$BASE_REMOTE"
```

The helper re-reads the live fork ref and uses an explicit expected-OID
`--force-with-lease` for atomic compare-and-delete. It refuses base-remote
deletion and any ref that advances before or during the push.

On any refusal, preserve the branch and return it to classification. Report
each result. Never expand the approved list, and prune tracking refs only after
all approved deletions finish.

## Quick reference

| Condition | Result |
| --- | --- |
| Tip is an ancestor of `BASE_REF` | Normal-merge candidate |
| Tip equals merged PR `headRefOid` | Squash-merge candidate |
| Tip differs from approved or PR OID | Preserve and reclassify |
| Current, default, base-remote, or uncertain | Protect |
| Exact rows not explicitly approved | Do not delete |

## Common mistakes

- Saving only branch names instead of full approved OIDs.
- Checking an OID before approval but not again immediately before deletion.
- Using an unleased remote delete.
- Combining local and remote instances despite different tips.
- Treating urgency as approval or touching `BASE_REMOTE`.
