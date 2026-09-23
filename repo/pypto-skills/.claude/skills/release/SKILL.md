---
name: release
description: Use when cutting a pypto-skills release — bumping the plugin version so installed consumers can update, and tagging or publishing that version.
---

# Release

Publish a new plugin version. The version string in the plugin manifests is the
only signal a consumer has that an update exists: skill edits that ship without
a bump reach nobody who already installed the plugin.

This skill is local to this repository and is deliberately not bundled into
either published plugin.

## Know the manifests

Four files carry a version, two per plugin:

- `plugins/pypto-developer/.claude-plugin/plugin.json`
- `plugins/pypto-developer/.codex-plugin/plugin.json`
- `plugins/pypto-user/.claude-plugin/plugin.json`
- `plugins/pypto-user/.codex-plugin/plugin.json`

A plugin's Claude and Codex manifests must carry identical versions;
`tests/test_plugin_structure.py` enforces it. Neither `marketplace.json`
records a version, so there is no third place to edit.

Both plugins move on one shared version line: bump both to the same value even
when only one of them changed. Depart from that only on explicit user
direction.

## Establish the starting state

1. Confirm the checkout is on the default branch, clean, and current with the
   remote. Release from committed state, never from a dirty tree.
2. Read the current version from any manifest and confirm all four agree.
3. For each plugin, find the commit that actually set the current version, then
   list what changed since. Walk the manifest's history and take the oldest
   commit still carrying the current value — the last commit that *touched* the
   manifest is the wrong baseline, because a later edit to a description or a
   keyword would hide every unreleased change behind it:

```bash
MANIFEST=plugins/PLUGIN/.claude-plugin/plugin.json
CURRENT=$(jq -er '.version' "$MANIFEST") || exit 1
HISTORY=$(git log --format='%H' -- "$MANIFEST") || exit 1

# Walk in the current shell, never through a pipeline: a `while` on the right
# of a `|` runs in a subshell, so a failed read there would be masked by the
# exit status of the last pipeline stage and yield a plausible wrong baseline.
BASELINE=""
while IFS= read -r COMMIT; do
  [ -n "$COMMIT" ] || continue
  RAW=$(git show "$COMMIT:$MANIFEST") || exit 1
  AT_COMMIT=$(printf '%s' "$RAW" | jq -er '.version') || exit 1
  [ "$AT_COMMIT" = "$CURRENT" ] || break
  BASELINE=$COMMIT
done <<EOF
$HISTORY
EOF

[ -n "$BASELINE" ] || exit 1
git log --oneline "$BASELINE..HEAD" -- plugins/PLUGIN/
```

If that range is empty for both plugins there is nothing to release; say so and
stop rather than bumping a version over no change. An empty range from a
baseline resolved this way means the version really is current — never accept
an empty range that came from an unresolved or failed baseline lookup.

## Propose a level, then let the user choose

The version is always the user's decision. Never apply a number the user has
not named, however clear the evidence looks.

Judge the change set against what a consumer can ask a skill to do, and form a
recommendation:

- Patch — wording, corrections, and tightened contracts that leave every
  existing invocation working the same way.
- Minor — a new skill, a new bundled reference or helper, or a capability a
  caller could not reach before.
- Major — a removed or renamed skill, or a contract change that breaks an
  invocation that worked before.

Present all three candidates computed from the current version, mark exactly
one as recommended, and give the evidence in one line each so the user can
overrule it without rereading the log:

```text
Current: 0.1.1  (set by <commit> <subject>)
Changes since: <count> commits — <one line per plugin>

  Patch  0.1.2  <what in the change set argues for patch>   [recommended]
  Minor  0.2.0  <what in the change set argues for minor>
  Major  1.0.0  <what in the change set argues for major>
```

State which evidence drove the recommendation and what would change it. Wait
for an explicit choice; accept an arbitrary version the user names instead,
provided it sorts above the current one. Only then continue.

## Apply and verify

Edit the version in all four manifests, then run the full local gate:

```bash
python -m unittest discover -s tests
ruff check tests && ruff format --check tests && pyright
claude plugin validate ./plugins/pypto-developer --strict
claude plugin validate ./plugins/pypto-user --strict
claude plugin validate . --strict
git ls-files -z -- '*.sh' | xargs -0 -r -n 1 bash -n
```

Record each command and its exit status. Confirm by inspection that the four
manifests now read the intended version. Stop on any failure.

## Publish

Commit through the repository's `git-commit` contract and open the pull request
through `github-pr`. Derive the message from repository history rather than
assuming a convention; state the new version and the change set it covers.

The marketplace serves the default branch, so the release becomes visible to
consumers when the pull request merges — not when it is opened.

## Tag after merge

Once the bump is on the default branch, tag each plugin from an updated,
clean checkout. Preview first:

```bash
claude plugin tag ./plugins/pypto-developer --dry-run
claude plugin tag ./plugins/pypto-user --dry-run
```

Each call creates `NAME--vVERSION` and refuses to run against uncommitted
changes or an existing tag. Push tags only with explicit user authorization,
adding `--push` once the dry run reads correctly.

## Report

Close with the released version, the commits it covers, the verification
evidence, the merge state, any tags created, and the consumer update commands:

```bash
claude plugin marketplace update pypto-skills
claude plugin update pypto-developer@pypto-skills   # restart to apply

codex plugin marketplace upgrade                    # refresh the snapshot
codex plugin add pypto-developer@pypto-skills       # reinstall at the new version
```
