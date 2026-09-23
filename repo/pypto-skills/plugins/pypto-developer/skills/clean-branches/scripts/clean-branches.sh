#!/usr/bin/env bash

set -eu

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
Usage:
  clean-branches.sh classify BRANCH REF BASE_REF DEFAULT_BRANCH [PR_HEAD_OID...]
  clean-branches.sh delete-local BRANCH APPROVED_OID DEFAULT_BRANCH
  clean-branches.sh delete-remote BRANCH APPROVED_OID DEFAULT_BRANCH PUSH_REMOTE BASE_REMOTE
EOF
  exit 2
}

branch_ref_for() {
  short_branch=$1
  full_branch_ref="refs/heads/$short_branch"
  git check-ref-format --branch "$short_branch" >/dev/null 2>&1 &&
    git check-ref-format "$full_branch_ref" >/dev/null 2>&1 ||
    fail "invalid branch name: $short_branch"
  printf '%s\n' "$full_branch_ref"
}

classification_ref_for() {
  short_branch=$1
  supplied_ref=$2
  local_branch_ref=$(branch_ref_for "$short_branch")
  git check-ref-format "$supplied_ref" >/dev/null 2>&1 ||
    fail "invalid full branch ref: $supplied_ref"
  case "$supplied_ref" in
    "$local_branch_ref" | refs/remotes/*/"$short_branch")
      ;;
    *)
      fail "branch ref does not match branch name: $supplied_ref"
      ;;
  esac
  printf '%s\n' "$supplied_ref"
}

require_expected_oid() {
  case "$1" in
    "" | *[!0-9a-f]*)
      fail "expected OID must be a full lowercase hexadecimal object ID"
      ;;
  esac
  canonical_oid=$(git rev-parse --verify "$1^{commit}" 2>/dev/null) ||
    fail "expected OID is not an available commit: $1"
  [ "$canonical_oid" = "$1" ] ||
    fail "expected OID must be the full commit ID: $1"
}

protect_branch() {
  branch=$1
  branch_ref=$2
  default_branch=$3
  current_branch=$(git branch --show-current)
  [ -n "$current_branch" ] ||
    fail "detached HEAD is unsupported"
  [ "$branch" != "$current_branch" ] ||
    fail "refusing to delete current branch: $branch"
  [ "$branch" != "$default_branch" ] ||
    fail "refusing to delete default branch: $branch"
  worktree_output=$(git worktree list --porcelain) ||
    fail "unable to enumerate worktrees"
  if grep -Fqx "branch $branch_ref" <<<"$worktree_output"; then
    fail "refusing to delete branch checked out in a linked worktree: $branch"
  else
    worktree_search_status=$?
    [ "$worktree_search_status" -eq 1 ] ||
      fail "unable to search enumerated worktrees"
  fi
}

classify() {
  [ "$#" -ge 4 ] || usage
  branch=$1
  branch_ref=$2
  base_ref=$3
  default_branch=$4
  shift 4

  branch_ref=$(classification_ref_for "$branch" "$branch_ref")
  current_branch=$(git branch --show-current)
  [ -n "$current_branch" ] ||
    fail "detached HEAD is unsupported"
  if [ "$branch" = "$current_branch" ]; then
    printf '%s\n' protected-current
    return
  fi
  if [ "$branch" = "$default_branch" ]; then
    printf '%s\n' protected-default
    return
  fi

  branch_oid=$(git rev-parse --verify "$branch_ref^{commit}") ||
    fail "branch ref is unavailable: $branch_ref"
  base_oid=$(git rev-parse --verify "$base_ref^{commit}") ||
    fail "base ref is unavailable: $base_ref"
  if git merge-base --is-ancestor "$branch_oid" "$base_oid"; then
    printf '%s\n' normal-merge
    return
  else
    merge_base_status=$?
    [ "$merge_base_status" -eq 1 ] ||
      fail "unable to compare branch ref with base ref"
  fi

  for pr_head_oid in "$@"; do
    if [ "$branch_oid" = "$pr_head_oid" ]; then
      printf '%s\n' squash-merge
      return
    fi
  done
  printf '%s\n' reused-or-unfinished
}

delete_local() {
  [ "$#" -eq 3 ] || usage
  branch=$1
  approved_oid=$2
  default_branch=$3

  branch_ref=$(branch_ref_for "$branch")
  require_expected_oid "$approved_oid"
  protect_branch "$branch" "$branch_ref" "$default_branch"

  actual_oid=$(git rev-parse --verify "$branch_ref^{commit}" 2>/dev/null) ||
    fail "approved local branch no longer exists: $branch"
  [ "$actual_oid" = "$approved_oid" ] ||
    fail "local branch changed since approval: $branch"

  git update-ref -d "$branch_ref" "$approved_oid" ||
    fail "local branch changed during atomic deletion: $branch"
  git config --remove-section "branch.$branch" 2>/dev/null || :
}

delete_remote() {
  [ "$#" -eq 5 ] || usage
  branch=$1
  approved_oid=$2
  default_branch=$3
  push_remote=$4
  base_remote=$5

  branch_ref=$(branch_ref_for "$branch")
  require_expected_oid "$approved_oid"
  protect_branch "$branch" "$branch_ref" "$default_branch"
  [ "$push_remote" != "$base_remote" ] ||
    fail "refusing to delete from base remote: $base_remote"
  git remote get-url "$push_remote" >/dev/null 2>&1 ||
    fail "push remote is unavailable: $push_remote"

  remote_line=$(
    git ls-remote --exit-code --refs "$push_remote" "$branch_ref"
  ) || fail "approved remote branch no longer exists: $push_remote/$branch"
  IFS=$'\t' read -r actual_oid actual_ref <<<"$remote_line"
  [ "$actual_ref" = "$branch_ref" ] ||
    fail "unexpected remote ref for $push_remote/$branch"
  [ "$actual_oid" = "$approved_oid" ] ||
    fail "remote branch changed since approval: $push_remote/$branch"

  git push \
    --force-with-lease="$branch_ref:$approved_oid" \
    "$push_remote" --delete "$branch_ref"
}

[ "$#" -gt 0 ] || usage
command_name=$1
shift
case "$command_name" in
  classify)
    classify "$@"
    ;;
  delete-local)
    delete_local "$@"
    ;;
  delete-remote)
    delete_remote "$@"
    ;;
  *)
    usage
    ;;
esac
