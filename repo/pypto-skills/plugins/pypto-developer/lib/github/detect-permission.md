# Detect Pull-Request Write Permission

Run [setup](setup.md) and [pull-request lookup](lookup-pr.md) first. This
reference distinguishes an author updating an owner or fork branch from a
maintainer updating somebody else's branch. It also redirects `PUSH_REMOTE` to
the exact pull-request head repository.

## 1. Fetch and validate metadata

```bash
PR_DATA=$(gh pr view "$PR_NUMBER" --repo "$PR_REPO" \
  --json "number,state,headRefName,headRepository,headRepositoryOwner,maintainerCanModify,author") || {
  echo "Error: pull-request metadata could not be read" >&2
  exit 1
}

PR_STATE=$(printf '%s' "$PR_DATA" | jq -r '.state')
PR_HEAD_BRANCH=$(printf '%s' "$PR_DATA" | jq -r '.headRefName')
HEAD_REPO_OWNER=$(printf '%s' "$PR_DATA" | jq -r '.headRepositoryOwner.login')
HEAD_REPO_NAME=$(printf '%s' "$PR_DATA" | jq -r '.headRepository.name')
PR_AUTHOR=$(printf '%s' "$PR_DATA" | jq -r '.author.login')
MAINTAINER_CAN_MODIFY=$(printf '%s' "$PR_DATA" | jq -r '.maintainerCanModify')
CURRENT_USER=$(gh api user --jq '.login')

if [ "$PR_STATE" != "OPEN" ]; then
  echo "Error: pull request #$PR_NUMBER is not open" >&2
  exit 1
fi
if [ -z "$HEAD_REPO_OWNER" ] || [ "$HEAD_REPO_OWNER" = "null" ] ||
   [ -z "$HEAD_REPO_NAME" ] || [ "$HEAD_REPO_NAME" = "null" ]; then
  echo "Error: pull-request head repository is unavailable" >&2
  exit 1
fi
HEAD_REPO="$HEAD_REPO_OWNER/$HEAD_REPO_NAME"
```

## 2. Determine role

```bash
BASE_CAN_PUSH=$(gh api "repos/$PR_REPO" --jq '.permissions.push // false') || {
  echo "Error: base-repository permission could not be checked" >&2
  exit 1
}
HEAD_CAN_PUSH=$(gh api --hostname "$GITHUB_HOST" "repos/$HEAD_REPO" \
  --jq '.permissions.push // false') || {
  echo "Error: head-repository permission could not be checked" >&2
  exit 1
}

if [ "$PR_AUTHOR" = "$CURRENT_USER" ]; then
  if [ "$HEAD_CAN_PUSH" != "true" ]; then
    echo "Error: current user cannot push to PR head repository $HEAD_REPO" >&2
    exit 1
  fi
  if [ "$HEAD_REPO" = "$PR_REPO" ]; then
    ROLE="owner"
  else
    ROLE="fork"
  fi
elif [ "$HEAD_REPO" = "$PR_REPO" ] && [ "$BASE_CAN_PUSH" = "true" ]; then
  ROLE="maintainer"
elif [ "$HEAD_REPO" != "$PR_REPO" ] &&
     [ "$BASE_CAN_PUSH" = "true" ] &&
     [ "$MAINTAINER_CAN_MODIFY" = "true" ]; then
  ROLE="maintainer"
else
  echo "Error: no verified write path exists for pull request #$PR_NUMBER" >&2
  echo "The author may need to enable maintainer edits" >&2
  exit 1
fi
```

## 3. Select the head-repository remote

The `remote_targets_repo` helper comes from setup. Do not silently add a remote
because its URL and credentials are a repository-level choice.

```bash
PR_HEAD_REMOTE=""
while IFS= read -r REMOTE_NAME; do
  if remote_targets_repo "$REMOTE_NAME" "$HEAD_REPO"; then
    PR_HEAD_REMOTE="$REMOTE_NAME"
    break
  fi
done < <(git remote)

if [ -z "$PR_HEAD_REMOTE" ]; then
  echo "Error: no Git remote points to PR head repository $HEAD_REPO" >&2
  echo "Add a remote for that repository and rerun permission detection" >&2
  exit 1
fi

git fetch "$PR_HEAD_REMOTE" "$PR_HEAD_BRANCH" || {
  echo "Error: failed to fetch $PR_HEAD_REMOTE/$PR_HEAD_BRANCH" >&2
  exit 1
}
PUSH_REMOTE="$PR_HEAD_REMOTE"
```

## Context produced

- `ROLE`: `owner`, `fork`, or `maintainer`
- `PR_STATE`, `PR_HEAD_BRANCH`, `PR_AUTHOR`
- `HEAD_REPO` and the matching `PUSH_REMOTE`
- `MAINTAINER_CAN_MODIFY`, `HEAD_CAN_PUSH`

Do not remove the selected remote after pushing; its tracking relationship is
needed for later pull-request lookup.
