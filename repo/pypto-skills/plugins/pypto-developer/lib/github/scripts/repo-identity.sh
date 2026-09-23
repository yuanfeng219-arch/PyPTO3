#!/usr/bin/env bash

set -eu

LC_ALL=C
export LC_ALL

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

detail() {
  printf '  %s\n' "$*" >&2
}

usage() {
  cat >&2 <<'EOF'
Usage:
  repo-identity.sh resolve [--require-push]
EOF
  exit 2
}

validate_host() {
  case "$1" in
    "" | .* | -* | *[!A-Za-z0-9.-]*) return 1 ;;
    *) return 0 ;;
  esac
}

validate_repo() {
  [[ "$1" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]
}

url_identity() {
  local url=$1 authority path host repo
  case "$url" in
    git@*:*/*)
      host=${url#git@}
      host=${host%%:*}
      path=${url#*:}
      ;;
    ssh://*/*/* | http://*/*/* | https://*/*/*)
      path=${url#*://}
      authority=${path%%/*}
      host=${authority##*@}
      host=${host%%:*}
      path=${path#*/}
      ;;
    *) return 1 ;;
  esac
  path=${path#/}
  repo=${path%.git}
  host=$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')
  repo=$(printf '%s' "$repo" | tr '[:upper:]' '[:lower:]')
  validate_host "$host" && validate_repo "$repo" || return 1
  printf '%s\t%s\n' "$host" "$repo"
}

# Emit `full_name<TAB>base_repo<TAB>is_fork<TAB>can_push` for one repository
# response. `base_repo` is the parent of a fork and the repository itself
# otherwise; `can_push` is the authenticated account's push permission.
response_identity() {
  local expected_host=$1 expected_repo=$2 response=$3
  local full_name html_url response_host response_repo is_fork base_repo can_push
  local parent_url parent_host parent_repo
  full_name=$(printf '%s' "$response" | jq -er '.full_name | strings | select(length > 0)') ||
    fail "repository response has no valid full_name"
  validate_repo "$full_name" || fail "repository response has invalid full_name"
  if [[ "${full_name,,}" != "${expected_repo,,}" ]]; then
    fail "repository response does not match $expected_repo"
  fi
  html_url=$(printf '%s' "$response" | jq -er '.html_url | strings | select(length > 0)') ||
    fail "repository response has no valid html_url"
  IFS=$'\t' read -r response_host response_repo < <(url_identity "$html_url") ||
    fail "repository response has an invalid html_url"
  [ "$response_host" = "$expected_host" ] ||
    fail "repository response host does not match $expected_host"
  [[ "${response_repo,,}" = "${full_name,,}" ]] ||
    fail "repository response html_url does not match $full_name"

  is_fork=$(printf '%s' "$response" |
    jq -er '.fork | if type == "boolean" then tostring else error("invalid fork") end') ||
    fail "repository response has no valid fork flag"
  can_push=$(printf '%s' "$response" |
    jq -r 'if (.permissions.push? // false) == true then "true" else "false" end') ||
    fail "repository response has no readable push permission"
  base_repo=$full_name
  if [ "$is_fork" = "true" ]; then
    base_repo=$(printf '%s' "$response" |
      jq -er '.parent.full_name | strings | select(length > 0)') ||
      fail "fork response has no valid parent full_name"
    validate_repo "$base_repo" || fail "fork response has invalid parent full_name"
    parent_url=$(printf '%s' "$response" |
      jq -er '.parent.html_url | strings | select(length > 0)') ||
      fail "fork response has no valid parent html_url"
    IFS=$'\t' read -r parent_host parent_repo < <(url_identity "$parent_url") ||
      fail "fork response has an invalid parent html_url"
    [ "$parent_host" = "$expected_host" ] ||
      fail "fork parent host does not match $expected_host"
    [[ "${parent_repo,,}" = "${base_repo,,}" ]] ||
      fail "fork parent html_url does not match $base_repo"
  fi
  printf '%s\t%s\t%s\t%s\n' "$full_name" "$base_repo" "$is_fork" "$can_push"
}

# Report every fork remote and why no repository is writable, then stop. A
# missing write path is an explicit failure, never a silent fallback to a
# repository the account cannot push to or does not own.
report_unwritable() {
  local host=$1 base_repo=$2 fork_names=$3 unowned_names=$4 login=$5
  if [ -n "$fork_names" ]; then
    detail "fork remotes: $fork_names"
  else
    detail "fork remotes: none"
  fi
  if [ -n "$unowned_names" ]; then
    detail "writable but not owned by $login: $unowned_names"
  fi
  detail "base repository without push permission: $base_repo"
  detail "fork $base_repo on $host and add the fork as a Git remote,"
  detail "or request push access to $base_repo, then rerun this workflow"
  fail "no writable repository on $host for this checkout"
}

resolve() {
  local require_push=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --require-push)
        [ "$require_push" -eq 0 ] || usage
        require_push=1
        shift
        ;;
      *) usage ;;
    esac
  done

  local remote_output remote_name remote_url remote_kind identity host repo
  local github_host="" candidate metadata base_repo="" local_repo=""
  local candidate_full candidate_base candidate_is_fork candidate_can_push
  local base_metadata base_full base_parent base_is_fork base_can_push
  local default_branch base_url base_host base_repo_path owner login=""
  local fork_names="" local_can_push="" is_fork=false
  local writable_count=0 owned_count=0 owned_first="" owned_names=""
  local unowned_names=""
  declare -A candidates=()
  declare -A fork_push=()

  git rev-parse --show-toplevel >/dev/null 2>&1 ||
    fail "the current directory is not in a Git worktree"
  remote_output=$(git remote -v) || fail "unable to enumerate Git remotes"
  [ -n "$remote_output" ] || fail "the Git worktree has no remotes"
  while read -r remote_name remote_url remote_kind; do
    case "$remote_kind" in
      "(fetch)" | "(push)") ;;
      *) continue ;;
    esac
    identity=$(url_identity "$remote_url") ||
      fail "remote $remote_name has an unsupported or invalid URL"
    IFS=$'\t' read -r host repo <<<"$identity"
    if [ -z "$github_host" ]; then
      github_host=$host
    elif [ "$github_host" != "$host" ]; then
      fail "Git remotes span unrelated hosts"
    fi
    candidates["$repo"]=1
  done <<<"$remote_output"

  [ "${#candidates[@]}" -gt 0 ] || fail "no GitHub remote identity was found"
  for candidate in "${!candidates[@]}"; do
    metadata=$(gh api --hostname "$github_host" "repos/$candidate") ||
      fail "repository metadata could not be read for $candidate"
    IFS=$'\t' read -r candidate_full candidate_base candidate_is_fork candidate_can_push \
      < <(response_identity "$github_host" "$candidate" "$metadata")
    if [ -z "$base_repo" ]; then
      base_repo=$candidate_base
    elif [[ "${base_repo,,}" != "${candidate_base,,}" ]]; then
      fail "Git remotes span unrelated repositories"
    fi
    if [ "$candidate_is_fork" = "true" ]; then
      fork_push["$candidate_full"]=$candidate_can_push
    fi
  done

  base_metadata=$(gh api --hostname "$github_host" "repos/$base_repo") ||
    fail "base repository metadata could not be read for $base_repo"
  # The base repository is re-read on its own so its default branch, push
  # permission, and identity come from the target itself, not from a fork's
  # parent stanza. `base_parent` and `base_is_fork` describe the base and are
  # deliberately unused: the pull-request destination is this repository.
  IFS=$'\t' read -r base_full base_parent base_is_fork base_can_push \
    < <(response_identity "$github_host" "$base_repo" "$base_metadata")
  base_url=$(printf '%s' "$base_metadata" |
    jq -er '.html_url | strings | select(length > 0)') ||
    fail "base repository response has no valid html_url"
  IFS=$'\t' read -r base_host base_repo_path < <(url_identity "$base_url") ||
    fail "base repository response has an invalid html_url"
  [ "$base_host" = "$github_host" ] ||
    fail "base repository response host does not match $github_host"
  [[ "${base_repo_path,,}" = "${base_full,,}" ]] ||
    fail "base repository html_url does not match $base_full"
  default_branch=$(printf '%s' "$base_metadata" |
    jq -er '.default_branch | strings | select(length > 0)') ||
    fail "base repository has no valid default branch"
  base_repo=$base_full

  for candidate in "${!fork_push[@]}"; do
    fork_names="${fork_names:+$fork_names, }$candidate"
  done

  if [ "$require_push" -eq 1 ]; then
    # A fork is the write target only when the authenticated account both owns
    # it and can push to it. Write access is not ownership: an account can be
    # a collaborator on somebody else's fork, and that fork must never become
    # the push target, whether it is the only writable one or one of several.
    for candidate in "${!fork_push[@]}"; do
      [ "${fork_push[$candidate]}" = "true" ] || continue
      writable_count=$((writable_count + 1))
    done
    if [ "$writable_count" -gt 0 ]; then
      login=$(gh api --hostname "$github_host" user --jq '.login') ||
        fail "authenticated GitHub account could not be read"
      [ -n "$login" ] || fail "authenticated GitHub account is empty"
      for candidate in "${!fork_push[@]}"; do
        [ "${fork_push[$candidate]}" = "true" ] || continue
        owner=${candidate%%/*}
        if [[ "${owner,,}" = "${login,,}" ]]; then
          owned_count=$((owned_count + 1))
          owned_first=$candidate
          owned_names="${owned_names:+$owned_names, }$candidate"
        else
          unowned_names="${unowned_names:+$unowned_names, }$candidate"
        fi
      done
      if [ "$owned_count" -gt 1 ]; then
        detail "writable fork remotes owned by $login: $owned_names"
        fail "multiple writable fork repositories were found"
      fi
      if [ "$owned_count" -eq 1 ]; then
        local_repo=$owned_first
      fi
    fi
    if [ -n "$local_repo" ]; then
      local_can_push=true
    elif [ "$base_can_push" = "true" ]; then
      local_repo=$base_repo
      local_can_push=true
    else
      report_unwritable "$github_host" "$base_repo" "$fork_names" \
        "$unowned_names" "$login"
    fi
  else
    if [ "${#fork_push[@]}" -gt 1 ]; then
      detail "fork remotes: $fork_names"
      fail "multiple unrelated fork repositories were found"
    fi
    for candidate in "${!fork_push[@]}"; do
      local_repo=$candidate
      local_can_push=${fork_push[$candidate]}
    done
    if [ -z "$local_repo" ]; then
      local_repo=$base_repo
      local_can_push=$base_can_push
    fi
  fi

  if [[ "${local_repo,,}" != "${base_repo,,}" ]]; then
    is_fork=true
  fi

  jq -cn \
    --arg github_host "$github_host" \
    --arg local_repo "$local_repo" \
    --arg base_repo "$base_repo" \
    --arg default_branch "$default_branch" \
    --argjson is_fork "$is_fork" \
    --argjson local_can_push "$local_can_push" \
    '{github_host: $github_host, local_repo: $local_repo, base_repo: $base_repo,
      is_fork: $is_fork, local_can_push: $local_can_push,
      default_branch: $default_branch}'
}

[ "$#" -ge 1 ] || usage
COMMAND=$1
shift
case "$COMMAND" in
  resolve) resolve "$@" ;;
  *) usage ;;
esac
