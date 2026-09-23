# Look Up a Pull Request

Requires the context from [setup](setup.md). Resolve
[`scripts/pr-context.sh`](scripts/pr-context.sh) relative to this file and set
its absolute path in `PR_LOOKUP_HELPER` before running these blocks.

This reference resolves `PR_LOOKUP_RESULT`, `PR_MATCH_COUNT`, `PR_NUMBER`,
`PR_STATE`, and `PR_HEAD_BRANCH`. Set `PR_LOOKUP_ALLOW_NONE=true` only when a
zero-match result should select a create route.

## Look up a known number

Validate a user-supplied number as a canonical positive decimal (no zero or
leading-zero forms) before passing it positionally to `gh`:

```bash
if [ -n "${PR_NUMBER:-}" ]; then
  "$PR_LOOKUP_HELPER" validate-number "$PR_NUMBER" >/dev/null || exit 1
  PR_DATA=$(GH_HOST="$GITHUB_HOST" gh pr view "$PR_NUMBER" \
    --repo "$PR_REPO" --json number,state,headRefName,title) || {
    echo "Error: pull request #$PR_NUMBER was not found in $PR_REPO" >&2
    exit 1
  }
  if ! printf '%s' "$PR_DATA" | jq -e '
    type == "object" and
    (.number | type == "number") and
    (.state | type == "string" and length > 0) and
    (.headRefName | type == "string" and length > 0) and
    (.title | type == "string")
  ' >/dev/null 2>&1; then
    echo "Error: pull-request metadata is malformed" >&2
    exit 1
  fi
  PR_LOOKUP_RESULT=$(printf '%s' "$PR_DATA" | jq -c '{
    route: "update",
    match_count: 1,
    pr: .
  }')
fi
```

## Look up the current or named head branch

When no number is supplied, use the fork prefix discovered by setup. Set
`PR_HEAD_BRANCH` before this block to search a branch other than
`CURRENT_BRANCH`.

The helper uses the supported, host-pinned REST request:

```text
gh api --hostname "$GITHUB_HOST" --method GET \
  "repos/$PR_REPO/pulls" -f state=open -f "head=$HEAD_SELECTOR" \
  -f per_page=100 --paginate --slurp
# Aggregate the returned page array separately with `jq -e 'add'`; `gh api`
# rejects combining `--slurp` with `--jq`.
```

It validates the combined response and fails closed on malformed, failed, or
multiple-match results.

```bash
if [ -z "${PR_NUMBER:-}" ]; then
  PR_HEAD_BRANCH=${PR_HEAD_BRANCH:-$CURRENT_BRANCH}
  HEAD_SELECTOR="${PR_HEAD_PREFIX}${PR_HEAD_BRANCH}"
  PR_LOOKUP_ALLOW_NONE_VALUE=${PR_LOOKUP_ALLOW_NONE:-false}
  case "$PR_LOOKUP_ALLOW_NONE_VALUE" in
    true)
      PR_LOOKUP_RESULT=$("$PR_LOOKUP_HELPER" lookup \
        "$GITHUB_HOST" "$PR_REPO" "$HEAD_SELECTOR" --allow-none) || exit 1
      ;;
    false)
      PR_LOOKUP_RESULT=$("$PR_LOOKUP_HELPER" lookup \
        "$GITHUB_HOST" "$PR_REPO" "$HEAD_SELECTOR") || exit 1
      ;;
    *)
      echo "Error: PR_LOOKUP_ALLOW_NONE must be true or false" >&2
      exit 1
      ;;
  esac
  PR_DATA=$(printf '%s' "$PR_LOOKUP_RESULT" | jq -c '.pr')
fi
```

## Normalize and validate outputs

```bash
PR_MATCH_COUNT=$(printf '%s' "$PR_LOOKUP_RESULT" | jq -r '.match_count')
case "$PR_MATCH_COUNT" in
  0)
    PR_NUMBER=""
    PR_STATE=""
    PR_HEAD_BRANCH=""
    ;;
  1)
    PR_NUMBER=$(printf '%s' "$PR_DATA" | jq -r '.number')
    PR_STATE=$(printf '%s' "$PR_DATA" | jq -r '.state | ascii_upcase')
    PR_HEAD_BRANCH=$(printf '%s' "$PR_DATA" | jq -r '.headRefName')
    if ! "$PR_LOOKUP_HELPER" validate-number "$PR_NUMBER" >/dev/null ||
       [ -z "$PR_STATE" ] || [ "$PR_STATE" = "null" ] ||
       [ -z "$PR_HEAD_BRANCH" ] || [ "$PR_HEAD_BRANCH" = "null" ]; then
      echo "Error: pull-request metadata is incomplete" >&2
      exit 1
    fi
    ;;
  *)
    echo "Error: invalid normalized pull-request match count" >&2
    exit 1
    ;;
esac
```

To offer a user a choice rather than guessing:

```bash
GH_HOST="$GITHUB_HOST" gh pr list --repo "$PR_REPO" --state open \
  --json number,title,headRefName,author
```

Never select the first result when more than one pull request matches.
