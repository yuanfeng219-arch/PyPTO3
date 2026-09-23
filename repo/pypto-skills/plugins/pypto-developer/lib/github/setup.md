# GitHub Workflow Setup

Run this reference first and keep the resulting variables in the same shell.
It discovers repository, branch, and remote context without assuming remote
names or a default branch.

Resolve the [repository scope gate](../repository/scope.md) before the first
repository, remote, or GitHub mutation. These workflows act on their own
authority only in a PTO-family repository; anywhere else they warn and wait for
explicit user confirmation.

Sections 1 and 2 are read-only discovery and may run before the gate resolves.
Section 3 is not: its `git fetch` calls contact a remote and write objects,
remote-tracking refs, and `FETCH_HEAD` into the local repository. Resolve the
gate before that step, not merely before a push.

## 1. Authenticate and enter the worktree

```bash
if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is required" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "Error: GitHub CLI authentication is required; run 'gh auth login'" >&2
  exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "Error: the current directory is not in a Git worktree" >&2
  exit 1
}
cd "$REPO_ROOT" || exit 1

CURRENT_BRANCH=$(git branch --show-current)
if [ -z "$CURRENT_BRANCH" ]; then
  echo "Error: a checked-out branch is required; detached HEAD is unsupported" >&2
  exit 1
fi
```

## 2. Resolve GitHub repository identity

`LOCAL_REPO` is the writable repository this checkout pushes to. `PR_REPO` is
the repository that receives pull requests: the parent for a fork, otherwise
the same repository.

Resolve [`scripts/repo-identity.sh`](scripts/repo-identity.sh) relative to this
reference as the absolute executable `REPO_IDENTITY_HELPER`, never from the
consumer repository. Never substitute `gh repo view` or any other ambient
GitHub CLI repository selection. `gh` resolves a *base* repository from the
remote set, prefers a parent over the checkout's own fork, and asks the user to
pick one when several remotes qualify. That answers a different question than
"which repository does this checkout push to", so a fork checkout silently
resolves to its upstream and the fork disappears from the whole workflow.

`resolve --require-push` reads every remote fetch and push URL, pins one host,
validates each candidate through host-pinned REST metadata, and selects the
write target by evidence:

- a fork the authenticated account both owns and can push to wins. Write
  access is not ownership: an account can be a collaborator on somebody else's
  fork, and such a fork never becomes the push target, whether it is the only
  writable one or one of several;
- more than one owned writable fork is an error rather than a guess;
- with no owned writable fork, the base repository wins only when the account
  can push to it;
- with neither, the helper fails explicitly and names the base repository and
  every fork remote it rejected. Never fall back to a repository the account
  cannot push to, and never defer that failure until Git refuses the push.

```bash
REPO_IDENTITY=$("$REPO_IDENTITY_HELPER" resolve --require-push) || exit 1
GITHUB_HOST=$(printf '%s' "$REPO_IDENTITY" | jq -r '.github_host')
LOCAL_REPO=$(printf '%s' "$REPO_IDENTITY" | jq -r '.local_repo')
PR_REPO=$(printf '%s' "$REPO_IDENTITY" | jq -r '.base_repo')
DEFAULT_BRANCH=$(printf '%s' "$REPO_IDENTITY" | jq -r '.default_branch')
for IDENTITY_NAME in GITHUB_HOST LOCAL_REPO PR_REPO DEFAULT_BRANCH; do
  case "${!IDENTITY_NAME}" in
    "" | null)
      echo "Error: repository identity field $IDENTITY_NAME is empty" >&2
      exit 1
      ;;
  esac
done
```

## 3. Match remotes by repository identity

Remote names are local choices. Every fetch URL and the single effective push
URL must match the discovered GitHub host and expected `owner/name`. Reject
split push destinations rather than choosing one. `PUSH_REMOTE` is the remote
that matches the verified writable `LOCAL_REPO`, so a fork checkout pushes to
its fork remote even when an upstream remote is also configured.

```bash
remote_url_identity() {
  REMOTE_URL=$1
  case "$REMOTE_URL" in
    git@*:*/*)
      REMOTE_HOST=${REMOTE_URL#git@}
      REMOTE_HOST=${REMOTE_HOST%%:*}
      REMOTE_PATH=${REMOTE_URL#*:}
      ;;
    ssh://*/*/*)
      REMOTE_PATH=${REMOTE_URL#ssh://}
      REMOTE_AUTHORITY=${REMOTE_PATH%%/*}
      REMOTE_HOST=${REMOTE_AUTHORITY##*@}
      REMOTE_HOST=${REMOTE_HOST%%:*}
      REMOTE_PATH=${REMOTE_PATH#*/}
      ;;
    http://*/*/*|https://*/*/*)
      REMOTE_PATH=${REMOTE_URL#*://}
      REMOTE_AUTHORITY=${REMOTE_PATH%%/*}
      REMOTE_HOST=${REMOTE_AUTHORITY##*@}
      REMOTE_HOST=${REMOTE_HOST%%:*}
      REMOTE_PATH=${REMOTE_PATH#*/}
      ;;
    *)
      return 1
      ;;
  esac
  REMOTE_PATH=${REMOTE_PATH#/}
  REMOTE_PATH=${REMOTE_PATH%.git}
  case "$REMOTE_PATH" in
    */*/*|""|/*) return 1 ;;
    */*) ;;
    *) return 1 ;;
  esac
  REMOTE_HOST=$(printf '%s' "$REMOTE_HOST" | tr '[:upper:]' '[:lower:]')
  REMOTE_REPO=$(printf '%s' "$REMOTE_PATH" | tr '[:upper:]' '[:lower:]')
}

remote_targets_repo() {
  REMOTE_NAME=$1
  EXPECTED_REPO=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
  FETCH_URLS=$(git remote get-url --all "$REMOTE_NAME" 2>/dev/null) ||
    return 1
  [ -n "$FETCH_URLS" ] || return 1

  while IFS= read -r FETCH_URL; do
    remote_url_identity "$FETCH_URL" || return 1
    if [ "$REMOTE_HOST" != "$GITHUB_HOST" ] ||
       [ "$REMOTE_REPO" != "$EXPECTED_REPO" ]; then
      return 1
    fi
  done <<EOF
$FETCH_URLS
EOF

  PUSH_URLS=$(git remote get-url --push --all "$REMOTE_NAME" 2>/dev/null) ||
    return 1
  PUSH_URL_COUNT=$(printf '%s\n' "$PUSH_URLS" | sed '/^$/d' | wc -l)
  while IFS= read -r PUSH_URL; do
    remote_url_identity "$PUSH_URL" || return 1
    if [ "$REMOTE_HOST" != "$GITHUB_HOST" ] ||
       [ "$REMOTE_REPO" != "$EXPECTED_REPO" ]; then
      echo "Error: remote $REMOTE_NAME has a mismatched push destination" >&2
      return 1
    fi
  done <<EOF
$PUSH_URLS
EOF
  if [ "$PUSH_URL_COUNT" -ne 1 ]; then
    echo "Error: remote $REMOTE_NAME must have exactly one push destination" >&2
    return 1
  fi
}

BASE_REMOTE=""
PUSH_REMOTE=""
while IFS= read -r REMOTE_NAME; do
  if remote_targets_repo "$REMOTE_NAME" "$PR_REPO" &&
     [ -z "$BASE_REMOTE" ]; then
    BASE_REMOTE="$REMOTE_NAME"
  fi
  if remote_targets_repo "$REMOTE_NAME" "$LOCAL_REPO" &&
     [ -z "$PUSH_REMOTE" ]; then
    PUSH_REMOTE="$REMOTE_NAME"
  fi
done < <(git remote)

if [ -z "$BASE_REMOTE" ]; then
  echo "Error: no remote safely targets $GITHUB_HOST/$PR_REPO" >&2
  exit 1
fi
if [ -z "$PUSH_REMOTE" ]; then
  echo "Error: no remote safely targets $GITHUB_HOST/$LOCAL_REPO" >&2
  exit 1
fi

git fetch "$BASE_REMOTE" "$DEFAULT_BRANCH" || {
  echo "Error: failed to fetch $BASE_REMOTE/$DEFAULT_BRANCH" >&2
  exit 1
}
if [ "$PUSH_REMOTE" != "$BASE_REMOTE" ]; then
  git fetch "$PUSH_REMOTE" || {
    echo "Error: failed to fetch push remote $PUSH_REMOTE" >&2
    exit 1
  }
fi

BASE_REF="$BASE_REMOTE/$DEFAULT_BRANCH"
if ! git rev-parse --verify "$BASE_REF^{commit}" >/dev/null 2>&1; then
  echo "Error: base ref $BASE_REF is unavailable after fetch" >&2
  exit 1
fi
```

## 4. Set role and pull-request head prefix

```bash
if [ "$LOCAL_REPO" = "$PR_REPO" ]; then
  ROLE="owner"
  PR_HEAD_PREFIX=""
else
  ROLE="fork"
  PR_HEAD_PREFIX="${LOCAL_REPO%%/*}:"
fi
```

`detect-permission.md` may later change `ROLE` to `maintainer` and redirect
`PUSH_REMOTE` to a pull-request author's remote.

## Context produced

| Variable | Meaning |
| --- | --- |
| `REPO_ROOT` | Current Git worktree root |
| `CURRENT_BRANCH` | Checked-out local branch |
| `DEFAULT_BRANCH` | GitHub default branch of `PR_REPO` |
| `BASE_REMOTE` | Remote matching `PR_REPO` |
| `BASE_REF` | `$BASE_REMOTE/$DEFAULT_BRANCH` |
| `PUSH_REMOTE` | Remote matching the contributor's writable repository |
| `LOCAL_REPO` | Verified writable `owner/name`: the fork when one exists |
| `PR_REPO` | `owner/name` repository receiving the pull request |
| `PR_HEAD_PREFIX` | Fork-owner prefix for pull-request head lookup, or empty |
| `ROLE` | `owner` or `fork`; permission detection can set `maintainer` |

The same shell also retains `GITHUB_HOST`, `LOCAL_REPO`, and
`remote_targets_repo` so later references can revalidate a write target
immediately before pushing.
