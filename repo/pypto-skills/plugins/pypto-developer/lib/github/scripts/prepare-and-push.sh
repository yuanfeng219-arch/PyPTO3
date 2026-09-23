#!/usr/bin/env bash

set -u

usage() {
  cat >&2 <<'EOF'
Usage:
  prepare-and-push.sh prepare EXPECTED_BASE_HOST EXPECTED_BASE_REPO \
    EXPECTED_HEAD_HOST EXPECTED_HEAD_REPO BASE_REMOTE DEFAULT_BRANCH \
    PUSH_REMOTE CURRENT_BRANCH PUSH_BRANCH HISTORY_REWRITTEN EXPECTED_REMOTE_OID
  prepare-and-push.sh push EXPECTED_BASE_HOST EXPECTED_BASE_REPO \
    EXPECTED_HEAD_HOST EXPECTED_HEAD_REPO BASE_REMOTE DEFAULT_BRANCH \
    PUSH_REMOTE CURRENT_BRANCH PUSH_BRANCH PREPARED_HEAD_OID PREPARED_BASE_OID \
    PREPARED_REMOTE_OID HISTORY_REWRITTEN

EXPECTED_REMOTE_OID is the verified pre-rewrite remote head, or - for a
verified unpublished branch. PREPARED_REMOTE_OID uses UNPUBLISHED.
EOF
  exit 2
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

require_boolean() {
  case "$2" in
    true|false) ;;
    *) fail "$1 must be true or false" ;;
  esac
}

require_oid() {
  if ! printf '%s\n' "$2" |
    grep -Eq '^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$'; then
    fail "$1 is not a full Git object ID"
  fi
}

require_branch() {
  git check-ref-format --branch "$2" >/dev/null 2>&1 ||
    fail "$1 is not a valid branch name"
}

require_host() {
  if ! printf '%s\n' "$1" |
    grep -Eq '^([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)(\.([A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?))*$'; then
    fail "EXPECTED_HOST is not a valid GitHub host"
  fi
}

require_repo() {
  if ! printf '%s\n' "$1" |
    grep -Eq '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'; then
    fail "EXPECTED_REPO must be an owner/name repository"
  fi
  case "$1" in
    ./*|../*|*/.|*/..) fail "EXPECTED_REPO contains an invalid component" ;;
  esac
}

require_remote() {
  REMOTE_VALUE=$1
  if ! printf '%s\n' "$REMOTE_VALUE" |
    grep -Eq '^[A-Za-z0-9][A-Za-z0-9._/-]*$'; then
    fail "invalid Git remote name: $REMOTE_VALUE"
  fi
  git check-ref-format "refs/remotes/$REMOTE_VALUE/check" >/dev/null 2>&1 ||
    fail "invalid Git remote name: $REMOTE_VALUE"
  REMOTE_FOUND=false
  while IFS= read -r REMOTE_CANDIDATE; do
    if [ "$REMOTE_CANDIDATE" = "$REMOTE_VALUE" ]; then
      REMOTE_FOUND=true
      break
    fi
  done < <(git remote)
  [ "$REMOTE_FOUND" = "true" ] ||
    fail "Git remote $REMOTE_VALUE is unavailable"
}

remote_url_identity() {
  REMOTE_URL=$1
  case "$REMOTE_URL" in
    git@*:*/*)
      REMOTE_HOST=${REMOTE_URL#git@}
      REMOTE_HOST=${REMOTE_HOST%%:*}
      REMOTE_PATH=${REMOTE_URL#*:}
      ;;
    ssh://*/*/*|http://*/*/*|https://*/*/*)
      REMOTE_PATH=${REMOTE_URL#*://}
      REMOTE_AUTHORITY=${REMOTE_PATH%%/*}
      REMOTE_HOST=${REMOTE_AUTHORITY##*@}
      REMOTE_HOST=${REMOTE_HOST%%:*}
      REMOTE_PATH=${REMOTE_PATH#*/}
      ;;
    *) return 1 ;;
  esac
  REMOTE_PATH=${REMOTE_PATH#/}
  REMOTE_PATH=${REMOTE_PATH%.git}
  case "$REMOTE_PATH" in
    */*/*|""|/*) return 1 ;;
    */*) ;;
    *) return 1 ;;
  esac
  REMOTE_HOST=$(printf '%s' "$REMOTE_HOST" |
    tr '[:upper:]' '[:lower:]') || return 1
  REMOTE_REPO=$(printf '%s' "$REMOTE_PATH" |
    tr '[:upper:]' '[:lower:]') || return 1
}

remote_fetches_repo() {
  REMOTE_NAME=$1
  EXPECTED_HOST_LOWER=$2
  EXPECTED_REPO_LOWER=$3
  FETCH_URLS=$(git remote get-url --all "$REMOTE_NAME" 2>/dev/null) ||
    return 1
  FETCH_URL_COUNT=0
  while IFS= read -r FETCH_URL; do
    [ -n "$FETCH_URL" ] || continue
    FETCH_URL_COUNT=$((FETCH_URL_COUNT + 1))
    remote_url_identity "$FETCH_URL" || return 1
    [ "$REMOTE_HOST" = "$EXPECTED_HOST_LOWER" ] &&
      [ "$REMOTE_REPO" = "$EXPECTED_REPO_LOWER" ] || return 1
  done <<EOF
$FETCH_URLS
EOF
  [ "$FETCH_URL_COUNT" -gt 0 ]
}

remote_targets_repo() {
  REMOTE_NAME=$1
  EXPECTED_HOST_LOWER=$2
  EXPECTED_REPO_LOWER=$3
  remote_fetches_repo \
    "$REMOTE_NAME" "$EXPECTED_HOST_LOWER" "$EXPECTED_REPO_LOWER" || return 1
  PUSH_URLS=$(git remote get-url --push --all "$REMOTE_NAME" 2>/dev/null) ||
    return 1
  PUSH_URL_COUNT=0
  while IFS= read -r PUSH_URL; do
    [ -n "$PUSH_URL" ] || continue
    PUSH_URL_COUNT=$((PUSH_URL_COUNT + 1))
    remote_url_identity "$PUSH_URL" || return 1
    [ "$REMOTE_HOST" = "$EXPECTED_HOST_LOWER" ] &&
      [ "$REMOTE_REPO" = "$EXPECTED_REPO_LOWER" ] || return 1
  done <<EOF
$PUSH_URLS
EOF
  [ "$PUSH_URL_COUNT" -eq 1 ]
}

validate_base_remote() {
  REMOTE_NAME=$1
  EXPECTED_HOST_LOWER=$2
  EXPECTED_REPO_LOWER=$3
  EXPECTED_HOST_DISPLAY=$4
  EXPECTED_REPO_DISPLAY=$5
  FETCH_URLS=$(git remote get-url --all "$REMOTE_NAME" 2>/dev/null) ||
    fail "failed to inspect base remote $REMOTE_NAME"
  FETCH_URL_COUNT=0
  while IFS= read -r FETCH_URL; do
    [ -n "$FETCH_URL" ] || continue
    FETCH_URL_COUNT=$((FETCH_URL_COUNT + 1))
    remote_url_identity "$FETCH_URL" ||
      fail "base remote $REMOTE_NAME does not fetch from $EXPECTED_HOST_DISPLAY/$EXPECTED_REPO_DISPLAY"
    [ "$REMOTE_HOST" = "$EXPECTED_HOST_LOWER" ] &&
      [ "$REMOTE_REPO" = "$EXPECTED_REPO_LOWER" ] ||
      fail "base remote $REMOTE_NAME does not fetch from $EXPECTED_HOST_DISPLAY/$EXPECTED_REPO_DISPLAY"
  done <<EOF
$FETCH_URLS
EOF
  [ "$FETCH_URL_COUNT" -eq 1 ] ||
    fail "base remote $REMOTE_NAME must have exactly one fetch URL"
}

remote_head_oid() {
  REMOTE_OUTPUT=$(git ls-remote --heads "$1" "refs/heads/$2") ||
    fail "failed to inspect $1/$2"
  REMOTE_LINE_COUNT=0
  REMOTE_OID=""
  while IFS= read -r REMOTE_LINE; do
    [ -n "$REMOTE_LINE" ] || continue
    REMOTE_LINE_COUNT=$((REMOTE_LINE_COUNT + 1))
    REMOTE_OID=${REMOTE_LINE%%[[:space:]]*}
  done <<EOF
$REMOTE_OUTPUT
EOF
  [ "$REMOTE_LINE_COUNT" -le 1 ] ||
    fail "multiple remote refs matched $1/$2"
  printf '%s\n' "$REMOTE_OID"
}

read_branch() {
  BRANCH_RESULT=$(git branch --show-current) ||
    fail "failed to inspect current branch"
  printf '%s\n' "$BRANCH_RESULT"
}

read_status() {
  STATUS_RESULT=$(git status --porcelain) ||
    fail "failed to inspect worktree status"
  printf '%s\n' "$STATUS_RESULT"
}

prepare() {
  [ "$#" -eq 11 ] || usage
  EXPECTED_BASE_HOST=$1
  EXPECTED_BASE_REPO=$2
  EXPECTED_HEAD_HOST=$3
  EXPECTED_HEAD_REPO=$4
  BASE_REMOTE=$5
  DEFAULT_BRANCH=$6
  PUSH_REMOTE=$7
  CURRENT_BRANCH=$8
  PUSH_BRANCH=$9
  HISTORY_REWRITTEN_INPUT=${10}
  EXPECTED_REMOTE_OID=${11}

  require_host "$EXPECTED_BASE_HOST"
  require_repo "$EXPECTED_BASE_REPO"
  require_host "$EXPECTED_HEAD_HOST"
  require_repo "$EXPECTED_HEAD_REPO"
  require_boolean "HISTORY_REWRITTEN" "$HISTORY_REWRITTEN_INPUT"
  case "$EXPECTED_REMOTE_OID" in
    -) ;;
    *) require_oid "EXPECTED_REMOTE_OID" "$EXPECTED_REMOTE_OID" ;;
  esac
  require_branch "DEFAULT_BRANCH" "$DEFAULT_BRANCH"
  require_branch "CURRENT_BRANCH" "$CURRENT_BRANCH"
  require_branch "PUSH_BRANCH" "$PUSH_BRANCH"
  require_remote "$BASE_REMOTE"
  require_remote "$PUSH_REMOTE"
  EXPECTED_BASE_HOST_LOWER=$(printf '%s' "$EXPECTED_BASE_HOST" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_BASE_HOST"
  EXPECTED_BASE_REPO_LOWER=$(printf '%s' "$EXPECTED_BASE_REPO" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_BASE_REPO"
  EXPECTED_HEAD_HOST_LOWER=$(printf '%s' "$EXPECTED_HEAD_HOST" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_HEAD_HOST"
  EXPECTED_HEAD_REPO_LOWER=$(printf '%s' "$EXPECTED_HEAD_REPO" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_HEAD_REPO"
  validate_base_remote \
    "$BASE_REMOTE" "$EXPECTED_BASE_HOST_LOWER" "$EXPECTED_BASE_REPO_LOWER" \
    "$EXPECTED_BASE_HOST" "$EXPECTED_BASE_REPO"
  remote_targets_repo \
    "$PUSH_REMOTE" "$EXPECTED_HEAD_HOST_LOWER" "$EXPECTED_HEAD_REPO_LOWER" ||
    fail "remote $PUSH_REMOTE does not target $EXPECTED_HEAD_HOST/$EXPECTED_HEAD_REPO"

  REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) ||
    fail "the current directory is not in a Git worktree"
  cd "$REPO_ROOT" || fail "failed to enter the Git worktree"
  ACTUAL_BRANCH=$(read_branch) || exit 1
  [ "$ACTUAL_BRANCH" = "$CURRENT_BRANCH" ] ||
    fail "expected branch $CURRENT_BRANCH, found $ACTUAL_BRANCH"
  WORKTREE_STATUS=$(read_status) || exit 1
  [ -z "$WORKTREE_STATUS" ] || fail "worktree must be clean before prepare"

  git fetch "$BASE_REMOTE" "$DEFAULT_BRANCH" ||
    fail "failed to refresh $BASE_REMOTE/$DEFAULT_BRANCH"
  BASE_REF="refs/remotes/$BASE_REMOTE/$DEFAULT_BRANCH"
  PREPARE_BEFORE_HEAD=$(git rev-parse HEAD) ||
    fail "failed to read HEAD before prepare"
  git rebase "$BASE_REF" >&2 ||
    fail "rebase stopped; resolve and restart prepare, or abort"
  PREPARED_HEAD_OID=$(git rev-parse HEAD) ||
    fail "failed to read prepared HEAD"
  PREPARED_BASE_OID=$(git rev-parse "$BASE_REF^{commit}") ||
    fail "failed to read prepared base tip"
  REMOTE_BASE_OID=$(remote_head_oid "$BASE_REMOTE" "$DEFAULT_BRANCH") ||
    exit 1
  require_oid "remote base OID" "$REMOTE_BASE_OID"
  [ "$REMOTE_BASE_OID" = "$PREPARED_BASE_OID" ] ||
    fail "base changed while prepare was running"
  COMMITS_AHEAD=$(git rev-list --count "$PREPARED_BASE_OID"..HEAD) ||
    fail "failed to count commits ahead of the prepared base"
  case "$COMMITS_AHEAD" in
    ""|*[!0-9]*) fail "commit count is not a non-negative integer" ;;
  esac
  [ "$COMMITS_AHEAD" -gt 0 ] ||
    fail "$CURRENT_BRANCH has no commits ahead of the prepared base"
  WORKTREE_STATUS=$(read_status) || exit 1
  [ -z "$WORKTREE_STATUS" ] || fail "worktree changed during prepare"

  PREPARED_REMOTE_OID=$(remote_head_oid "$PUSH_REMOTE" "$PUSH_BRANCH") ||
    exit 1
  if [ "$EXPECTED_REMOTE_OID" = "-" ]; then
    [ -z "$PREPARED_REMOTE_OID" ] ||
      fail "remote branch exists but was expected to be unpublished"
    PREPARED_REMOTE_VALUE="UNPUBLISHED"
  else
    [ "$PREPARED_REMOTE_OID" = "$EXPECTED_REMOTE_OID" ] ||
      fail "remote head changed before prepare result"
    PREPARED_REMOTE_VALUE=$PREPARED_REMOTE_OID
  fi

  HISTORY_REWRITTEN=$HISTORY_REWRITTEN_INPUT
  if [ -z "$PREPARED_REMOTE_OID" ]; then
    HISTORY_REWRITTEN=false
  elif [ "$PREPARE_BEFORE_HEAD" != "$PREPARED_HEAD_OID" ]; then
    HISTORY_REWRITTEN=true
    git merge-base --is-ancestor \
      "$PREPARED_REMOTE_OID" "$PREPARE_BEFORE_HEAD"
    ANCESTOR_STATUS=$?
    case "$ANCESTOR_STATUS" in
      0) ;;
      1) fail "remote branch is not contained in pre-prepare history" ;;
      *) fail "failed to inspect pre-prepare ancestry" ;;
    esac
  fi
  if [ "$HISTORY_REWRITTEN" = "false" ] &&
    [ -n "$PREPARED_REMOTE_OID" ]; then
    git merge-base --is-ancestor \
      "$PREPARED_REMOTE_OID" "$PREPARED_HEAD_OID"
    ANCESTOR_STATUS=$?
    case "$ANCESTOR_STATUS" in
      0) ;;
      1) HISTORY_REWRITTEN=true ;;
      *) fail "failed to inspect prepared ancestry" ;;
    esac
  fi

  printf '{"version":1,"prepared_head_oid":"%s",' "$PREPARED_HEAD_OID"
  printf '"prepared_base_oid":"%s",' "$PREPARED_BASE_OID"
  printf '"prepared_remote_oid":"%s",' "$PREPARED_REMOTE_VALUE"
  printf '"history_rewritten":%s}\n' "$HISTORY_REWRITTEN"
}

push_prepared() {
  [ "$#" -eq 13 ] || usage
  EXPECTED_BASE_HOST=$1
  EXPECTED_BASE_REPO=$2
  EXPECTED_HEAD_HOST=$3
  EXPECTED_HEAD_REPO=$4
  BASE_REMOTE=$5
  DEFAULT_BRANCH=$6
  PUSH_REMOTE=$7
  CURRENT_BRANCH=$8
  PUSH_BRANCH=$9
  PREPARED_HEAD_OID=${10}
  PREPARED_BASE_OID=${11}
  PREPARED_REMOTE_VALUE=${12}
  HISTORY_REWRITTEN=${13}

  require_host "$EXPECTED_BASE_HOST"
  require_repo "$EXPECTED_BASE_REPO"
  require_host "$EXPECTED_HEAD_HOST"
  require_repo "$EXPECTED_HEAD_REPO"
  require_remote "$BASE_REMOTE"
  require_remote "$PUSH_REMOTE"
  require_branch "DEFAULT_BRANCH" "$DEFAULT_BRANCH"
  require_branch "CURRENT_BRANCH" "$CURRENT_BRANCH"
  require_branch "PUSH_BRANCH" "$PUSH_BRANCH"
  require_oid "PREPARED_HEAD_OID" "$PREPARED_HEAD_OID"
  require_oid "PREPARED_BASE_OID" "$PREPARED_BASE_OID"
  case "$PREPARED_REMOTE_VALUE" in
    UNPUBLISHED) ;;
    *) require_oid "PREPARED_REMOTE_OID" "$PREPARED_REMOTE_VALUE" ;;
  esac
  require_boolean "HISTORY_REWRITTEN" "$HISTORY_REWRITTEN"

  EXPECTED_BASE_HOST_LOWER=$(printf '%s' "$EXPECTED_BASE_HOST" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_BASE_HOST"
  EXPECTED_BASE_REPO_LOWER=$(printf '%s' "$EXPECTED_BASE_REPO" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_BASE_REPO"
  EXPECTED_HEAD_HOST_LOWER=$(printf '%s' "$EXPECTED_HEAD_HOST" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_HEAD_HOST"
  EXPECTED_HEAD_REPO_LOWER=$(printf '%s' "$EXPECTED_HEAD_REPO" |
    tr '[:upper:]' '[:lower:]') ||
    fail "failed to normalize EXPECTED_HEAD_REPO"
  validate_base_remote \
    "$BASE_REMOTE" "$EXPECTED_BASE_HOST_LOWER" "$EXPECTED_BASE_REPO_LOWER" \
    "$EXPECTED_BASE_HOST" "$EXPECTED_BASE_REPO"
  remote_targets_repo \
    "$PUSH_REMOTE" "$EXPECTED_HEAD_HOST_LOWER" "$EXPECTED_HEAD_REPO_LOWER" ||
    fail "remote $PUSH_REMOTE does not target $EXPECTED_HEAD_HOST/$EXPECTED_HEAD_REPO"
  if [ "$PUSH_BRANCH" = "$DEFAULT_BRANCH" ] &&
    [ "$EXPECTED_BASE_HOST_LOWER" = "$EXPECTED_HEAD_HOST_LOWER" ] &&
    [ "$EXPECTED_BASE_REPO_LOWER" = "$EXPECTED_HEAD_REPO_LOWER" ]; then
    fail "refusing to push the protected base branch"
  fi

  git rev-parse --show-toplevel >/dev/null 2>&1 ||
    fail "the current directory is not in a Git worktree"
  ACTUAL_BRANCH=$(read_branch) || exit 1
  [ "$ACTUAL_BRANCH" = "$CURRENT_BRANCH" ] ||
    fail "checked-out branch changed after prepare"
  ACTUAL_HEAD_OID=$(git rev-parse HEAD) ||
    fail "failed to read local HEAD"
  [ "$ACTUAL_HEAD_OID" = "$PREPARED_HEAD_OID" ] ||
    fail "local HEAD changed after prepare"
  WORKTREE_STATUS=$(read_status) || exit 1
  [ -z "$WORKTREE_STATUS" ] || fail "worktree changed after prepare"

  LOCAL_BASE_OID=$(git rev-parse \
    "refs/remotes/$BASE_REMOTE/$DEFAULT_BRANCH^{commit}") ||
    fail "prepared base ref is unavailable"
  REMOTE_BASE_OID=$(remote_head_oid "$BASE_REMOTE" "$DEFAULT_BRANCH") ||
    exit 1
  if [ "$LOCAL_BASE_OID" != "$PREPARED_BASE_OID" ] ||
    [ "$REMOTE_BASE_OID" != "$PREPARED_BASE_OID" ]; then
    fail "base tip changed after prepare"
  fi
  ACTUAL_REMOTE_OID=$(remote_head_oid "$PUSH_REMOTE" "$PUSH_BRANCH") ||
    exit 1
  if [ "$PREPARED_REMOTE_VALUE" = "UNPUBLISHED" ]; then
    [ -z "$ACTUAL_REMOTE_OID" ] ||
      fail "remote head changed after prepare"
  else
    [ "$ACTUAL_REMOTE_OID" = "$PREPARED_REMOTE_VALUE" ] ||
      fail "remote head changed after prepare"
  fi

  validate_base_remote \
    "$BASE_REMOTE" "$EXPECTED_BASE_HOST_LOWER" "$EXPECTED_BASE_REPO_LOWER" \
    "$EXPECTED_BASE_HOST" "$EXPECTED_BASE_REPO"
  remote_targets_repo \
    "$PUSH_REMOTE" "$EXPECTED_HEAD_HOST_LOWER" "$EXPECTED_HEAD_REPO_LOWER" ||
    fail "remote $PUSH_REMOTE does not target $EXPECTED_HEAD_HOST/$EXPECTED_HEAD_REPO"

  if [ "$PREPARED_REMOTE_VALUE" = "UNPUBLISHED" ]; then
    git push "$PUSH_REMOTE" "$CURRENT_BRANCH:$PUSH_BRANCH" ||
      fail "normal first push failed"
    PUSH_MODE=normal
  elif [ "$HISTORY_REWRITTEN" = "true" ]; then
    git push \
      --force-with-lease="refs/heads/$PUSH_BRANCH:$PREPARED_REMOTE_VALUE" \
      "$PUSH_REMOTE" "$CURRENT_BRANCH:$PUSH_BRANCH" ||
      fail "leased rewritten push failed"
    PUSH_MODE=leased
  else
    git merge-base --is-ancestor \
      "$PREPARED_REMOTE_VALUE" "$PREPARED_HEAD_OID"
    ANCESTOR_STATUS=$?
    case "$ANCESTOR_STATUS" in
      0) ;;
      1) fail "prepared normal push is not a fast-forward" ;;
      *) fail "failed to inspect push ancestry" ;;
    esac
    git push "$PUSH_REMOTE" "$CURRENT_BRANCH:$PUSH_BRANCH" ||
      fail "normal push failed"
    PUSH_MODE=normal
  fi

  PUSHED_REMOTE_OID=$(remote_head_oid "$PUSH_REMOTE" "$PUSH_BRANCH") ||
    exit 1
  [ "$PUSHED_REMOTE_OID" = "$PREPARED_HEAD_OID" ] ||
    fail "remote head does not equal prepared HEAD after push"
  printf 'Push mode: %s\n' "$PUSH_MODE"
  printf 'Pushed HEAD: %s\n' "$PREPARED_HEAD_OID"
}

[ "$#" -ge 1 ] || usage
COMMAND=$1
shift
case "$COMMAND" in
  prepare) prepare "$@" ;;
  push) push_prepared "$@" ;;
  *) usage ;;
esac
