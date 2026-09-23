# Branch Naming

Create a valid branch name from an approved change summary without imposing a
repository-wide commit convention.

## Inputs

- `BRANCH_SUMMARY`: short description of the change, derived from the change
  under work rather than from an unrelated topic.
- `BRANCH_PREFIX`: optional prefix required by repository instructions or the
  user. Do not invent one when no policy supplies it.
- Context from [setup](setup.md).

## Approve the summary

A direct invocation needs explicit user approval of `BRANCH_SUMMARY`. A caller
that supplies standing authorization from an explicit autonomous invocation,
such as `auto-pr`, replaces that approval: derive the summary from the change,
use it without asking, and report the created branch name in the result.
Standing authorization covers only the name, never any other stop rule.

## Generate and validate

```bash
if [ -z "${BRANCH_SUMMARY:-}" ]; then
  echo "Error: BRANCH_SUMMARY is required" >&2
  exit 1
fi

BRANCH_SLUG=$(printf '%s' "$BRANCH_SUMMARY" |
  tr '[:upper:]' '[:lower:]' |
  sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' |
  cut -c1-50)
if [ -z "$BRANCH_SLUG" ]; then
  echo "Error: the change summary does not produce a usable branch name" >&2
  exit 1
fi

if [ -n "${BRANCH_PREFIX:-}" ]; then
  NEW_BRANCH="${BRANCH_PREFIX%/}/$BRANCH_SLUG"
else
  NEW_BRANCH="$BRANCH_SLUG"
fi

if ! git check-ref-format --branch "$NEW_BRANCH" >/dev/null 2>&1; then
  echo "Error: invalid branch name: $NEW_BRANCH" >&2
  exit 1
fi
if [ "$NEW_BRANCH" = "$DEFAULT_BRANCH" ] ||
   [ "$NEW_BRANCH" = "$CURRENT_BRANCH" ]; then
  echo "Error: choose a new branch distinct from current and default branches" >&2
  exit 1
fi
if git show-ref --verify --quiet "refs/heads/$NEW_BRANCH"; then
  echo "Error: local branch already exists: $NEW_BRANCH" >&2
  exit 1
fi

git switch --create "$NEW_BRANCH"
CURRENT_BRANCH="$NEW_BRANCH"
```

`CURRENT_BRANCH` is updated so later shared references continue to use the
same context variable.
