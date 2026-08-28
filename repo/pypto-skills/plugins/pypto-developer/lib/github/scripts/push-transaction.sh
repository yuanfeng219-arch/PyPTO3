#!/usr/bin/env bash

# Source this trusted library, then call pr_push_transaction with one mutation
# callback and one absolute trusted validation runner. Each call runs in a
# subshell so authority is single-use and readonly state cannot leak.

pr_transaction_fail() {
  echo "Error: $*" >&2
  return 1
}

pr_transaction_remote_oid() {
  local remote_name=$1
  local branch_name=$2
  local output line oid=""
  local count=0
  output=$(git ls-remote --heads \
    "$remote_name" "refs/heads/$branch_name") ||
    pr_transaction_fail "failed to inspect $remote_name/$branch_name" ||
    return 1
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    count=$((count + 1))
    oid=${line%%[[:space:]]*}
  done <<EOF
$output
EOF
  [ "$count" -le 1 ] ||
    pr_transaction_fail "multiple remote heads matched $remote_name/$branch_name" ||
    return 1
  if [ -n "$oid" ]; then
    printf '%s\n' "$oid"
  else
    printf '%s\n' "-"
  fi
}

pr_push_transaction() (
  [ "$#" -eq 12 ] ||
    pr_transaction_fail \
      "transaction requires helper, mutation, runner, identities, remotes, and branches" ||
    return 1
  local prepare_push_helper=$1
  local mutation_callback=$2
  local validation_runner=$3
  local expected_base_host=$4
  local expected_base_repo=$5
  local expected_head_host=$6
  local expected_head_repo=$7
  local base_remote=$8
  local default_branch=$9
  local push_remote=${10}
  local current_branch=${11}
  local push_branch=${12}
  local expected_remote_oid prepare_result prepared_fields
  local prepared_head_oid prepared_base_oid prepared_remote_oid
  local validated_head_oid validated_status
  local validation_command=${VALIDATION_COMMAND:-}
  local GIT_CONFIG_COUNT=1
  local GIT_CONFIG_KEY_0=core.hooksPath
  local GIT_CONFIG_VALUE_0=/dev/null
  local HISTORY_REWRITTEN=false

  case "$prepare_push_helper" in
    /*) ;;
    *) pr_transaction_fail "prepare/push helper path must be absolute"; return 1 ;;
  esac
  [ -x "$prepare_push_helper" ] ||
    pr_transaction_fail "prepare/push helper is not executable" ||
    return 1
  case "$validation_runner" in
    /*) ;;
    *) pr_transaction_fail "validation runner path must be absolute"; return 1 ;;
  esac
  [ -x "$validation_runner" ] ||
    pr_transaction_fail "validation runner is not executable" ||
    return 1
  [ -n "$validation_command" ] ||
    pr_transaction_fail "VALIDATION_COMMAND must not be empty" ||
    return 1
  case "$mutation_callback" in
    *[!A-Za-z0-9_:]*|"") pr_transaction_fail "invalid callback name"; return 1 ;;
  esac
  declare -F "$mutation_callback" >/dev/null ||
    pr_transaction_fail "mutation callback is unavailable" ||
    return 1

  expected_remote_oid=$(pr_transaction_remote_oid \
    "$push_remote" "$push_branch") || return 1
  export GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
  readonly GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
  readonly prepare_push_helper mutation_callback validation_runner
  readonly validation_command
  readonly expected_base_host expected_base_repo
  readonly expected_head_host expected_head_repo
  readonly base_remote default_branch push_remote current_branch push_branch
  readonly expected_remote_oid

  "$mutation_callback" ||
    pr_transaction_fail "repository mutation/commit callback failed" ||
    return 1
  case "$HISTORY_REWRITTEN" in
    true|false) ;;
    *) pr_transaction_fail "mutation callback set invalid rewrite state"; return 1 ;;
  esac

  prepare_result=$("$prepare_push_helper" prepare \
    "$expected_base_host" "$expected_base_repo" \
    "$expected_head_host" "$expected_head_repo" \
    "$base_remote" "$default_branch" "$push_remote" \
    "$current_branch" "$push_branch" \
    "$HISTORY_REWRITTEN" "$expected_remote_oid") || return 1
  prepared_fields=$(printf '%s' "$prepare_result" | jq -er '
    if .version == 1
       and (.prepared_head_oid | type == "string")
       and (.prepared_base_oid | type == "string")
       and (.prepared_remote_oid | type == "string")
       and (.history_rewritten | type == "boolean")
    then [.prepared_head_oid, .prepared_base_oid, .prepared_remote_oid,
          (.history_rewritten | tostring)] | @tsv
    else error("malformed prepare result") end') ||
    pr_transaction_fail "failed to parse prepare result" ||
    return 1
  IFS=$'\t' read -r prepared_head_oid prepared_base_oid \
    prepared_remote_oid HISTORY_REWRITTEN <<EOF
$prepared_fields
EOF
  readonly prepared_head_oid prepared_base_oid prepared_remote_oid
  readonly HISTORY_REWRITTEN

  "$validation_runner" "$prepared_head_oid" "$validation_command" ||
    pr_transaction_fail "trusted validation runner failed" ||
    return 1
  validated_head_oid=$(git rev-parse HEAD) ||
    pr_transaction_fail "failed to inspect validated HEAD" ||
    return 1
  [ "$validated_head_oid" = "$prepared_head_oid" ] ||
    pr_transaction_fail "validation changed prepared HEAD" ||
    return 1
  validated_status=$(git status --porcelain) ||
    pr_transaction_fail "failed to inspect validated worktree" ||
    return 1
  [ -z "$validated_status" ] ||
    pr_transaction_fail "validation changed prepared worktree" ||
    return 1

  "$prepare_push_helper" push \
    "$expected_base_host" "$expected_base_repo" \
    "$expected_head_host" "$expected_head_repo" \
    "$base_remote" "$default_branch" "$push_remote" \
    "$current_branch" "$push_branch" \
    "$prepared_head_oid" "$prepared_base_oid" \
    "$prepared_remote_oid" "$HISTORY_REWRITTEN"
)
