# Pull-Request Description

Compose a title and body a reviewer can act on without opening the diff. Every
statement must come from the pull-request commit range, the changes it
contains, or verification this workflow actually recorded. Never invent a
motivation, an issue link, a measurement, or a result.

## Inputs

- `BASE_REF`: base of the pull-request commit range, from [setup](setup.md).
- `PR_ROUTE`, `GITHUB_HOST`, `PR_REPO`, and `PR_NUMBER` for an existing pull
  request.
- Recorded verification: the exact commands and outcomes this workflow already
  produced. Absent evidence stays absent.

This reference sets `PR_TITLE` and `PR_BODY` for the caller.

## Read the range before writing

```bash
git log --reverse --format='%H%n%s%n%b' "$BASE_REF"..HEAD
git diff --stat "$BASE_REF"..HEAD
```

Read the diff of every change whose purpose the commit messages do not already
explain. A description written from subjects alone is not acceptable.

## Title

Use the single commit subject when the range holds one logical change. For
several commits, write one subject covering the whole range in the shape the
repository's own history uses; take that shape from [repository
policy](../repository/policy.md) and never invent a type, scope, or prefix the
repository does not use. Keep it on one line with no trailing period.

## Body

Set `PR_BODY` from this structure. Drop a section only when the range supplies
no evidence for it; never keep a heading filled with placeholder text.

```text
## Summary
<2-5 sentences: what the change does, why it is needed, and what a caller or
user observes afterwards>

## Changes
- `<path or area>`: <what changed there and why>

## Verification
- <exact command>: <exit status and concise result>
```

Requirements:

- The body must carry information the title does not. A body that restates the
  title, or that is only a list of commit subjects, is a failed composition.
- State behavior and consequence, not file mechanics. "Parses only the decorated
  class instead of the whole file" beats "edited the parser".
- Name any behavior change, interface change, migration step, or follow-up the
  range implies, and say plainly when there is none.
- Quantify only what the range or recorded evidence establishes. Never estimate
  a speedup, a size, or a risk.
- List under `## Verification` only commands this workflow ran, with their real
  outcomes. When none ran, say so instead of implying coverage.
- Keep issue references, co-authors, and other links only when a commit message
  or the user supplied them. Add no generated-by branding.

## Preserve an existing description

On the update route, read the current body before replacing it:

```bash
if [ "$PR_ROUTE" = "update" ]; then
  EXISTING_BODY=$(GH_HOST="$GITHUB_HOST" gh pr view "$PR_NUMBER" \
    --repo "$PR_REPO" --json body --jq '.body') || exit 1
fi
```

Keep every human-authored section, review agreement, and checklist state;
refresh only the parts derived from the commit range and fold new commits into
them. When the existing body cannot be updated without discarding human
content, keep it unchanged and report that it was preserved.

## Validate before publishing

```bash
if [ -z "$PR_TITLE" ] || [ -z "$PR_BODY" ] ||
   [ "$PR_BODY" = "$PR_TITLE" ]; then
  echo "Error: compose a description that explains the change" >&2
  exit 1
fi
```

An empty title, an empty body, or a body equal to the title stops publication.
