#!/usr/bin/env bash

set -eu

LC_ALL=C
export LC_ALL

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
Usage:
  issue-context.sh repository
  issue-context.sh search HOST REPO QUERY
  issue-context.sh templates HOST REPO
  issue-context.sh fix-preflight HOST REPO NUMBER [--in-progress-status STATUS] [--allow-conflict --approved-envelope-json JSON]
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

# Repository identity is shared with the pull-request workflows: read it from
# the sibling helper rather than reimplementing the remote sweep here. Identity
# comes from the Git remotes, so no ambient GitHub CLI base-repository
# selection can pick the repository for this checkout.
repository_context() {
  [ "$#" -eq 0 ] || usage
  local script_directory helper identity
  script_directory=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) ||
    fail "the helper directory could not be resolved"
  helper="$script_directory/repo-identity.sh"
  [ -x "$helper" ] || fail "repository identity helper is missing: $helper"
  identity=$("$helper" resolve) || exit 1
  printf '%s' "$identity" | jq -ce '
    if type == "object" and
      (.github_host | type == "string" and length > 0) and
      (.local_repo | type == "string" and length > 0) and
      (.base_repo | type == "string" and length > 0) and
      (.default_branch | type == "string" and length > 0)
    then {github_host, local_repo, issue_repo: .base_repo, default_branch}
    else error("malformed repository identity") end' ||
    fail "repository identity could not be read"
}

search_issues() {
  [ "$#" -eq 3 ] || usage
  local host=$1 repo=$2 query=$3 response
  validate_host "$host" || fail "invalid GitHub host: $host"
  validate_repo "$repo" || fail "repository must be an owner/name identity"
  [ -n "$query" ] || fail "issue search query is empty"
  response=$(GH_HOST="$host" gh issue list --repo "$repo" --state all --limit 1000 \
    --search "$query" --json number,title,body,state,labels,url) ||
    fail "issue search failed for $host/$repo"
  printf '%s' "$response" | jq -ce '
    if type == "array" and all(.[ ];
      type == "object" and
      (.number | type == "number" and . > 0) and
      (.title | type == "string") and
      (.body | type == "string") and
      (.state == "OPEN" or .state == "CLOSED") and
      (.labels | type == "array") and
      (.url | type == "string" and length > 0))
    then . else error("malformed issue search response") end'
}

list_templates() {
  [ "$#" -eq 2 ] || usage
  local host=$1 repo=$2 response error_file status
  validate_host "$host" || fail "invalid GitHub host: $host"
  validate_repo "$repo" || fail "repository must be an owner/name identity"
  error_file=$(mktemp) || fail "unable to create an error capture file"
  trap "rm -f '$error_file'" EXIT
  status=0
  response=$(gh api --hostname "$host" --paginate \
    "repos/$repo/contents/.github/ISSUE_TEMPLATE" 2>"$error_file") || status=$?
  if [ "$status" -ne 0 ]; then
    if grep -Eq 'Not Found|HTTP 404' "$error_file"; then
      if gh api --hostname "$host" "repos/$repo" >/dev/null 2>>"$error_file"; then
        printf '%s\n' '[]'
        return
      fi
    fi
    cat "$error_file" >&2
    fail "issue template discovery failed for $host/$repo"
  fi
  printf '%s' "$response" | jq -ce '
    if type != "array" then error("malformed issue template response") else
      [.[] |
        select(.type == "file") |
        .name as $name |
        if ($name | ascii_downcase) == "config.yml" or
           ($name | ascii_downcase) == "config.yaml" then empty
        elif ($name | ascii_downcase | endswith(".yml")) or
             ($name | ascii_downcase | endswith(".yaml")) then
          {name, path, kind: "issue_form"}
        elif ($name | ascii_downcase | endswith(".md")) then
          {name, path, kind: "legacy_template"}
        else empty end]
    end'
}

fix_preflight() {
  [ "$#" -ge 3 ] || usage
  local host=$1 repo=$2 number=$3
  local in_progress_status="" allow_conflict=0 response record conflict=0
  local approved_envelope_json="" approved_envelope_canonical=""
  local approved_envelope_set=0 current_envelope
  shift 3

  validate_host "$host" || fail "invalid GitHub host: $host"
  validate_repo "$repo" || fail "repository must be an owner/name identity"
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || fail "issue number must be a positive integer"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --in-progress-status)
        [ "$#" -ge 2 ] || usage
        [ -z "$in_progress_status" ] || usage
        in_progress_status=$2
        [ -n "$in_progress_status" ] || fail "in-progress status is empty"
        shift 2
        ;;
      --allow-conflict)
        [ "$allow_conflict" -eq 0 ] || usage
        allow_conflict=1
        shift
        ;;
      --approved-envelope-json)
        [ "$#" -ge 2 ] || usage
        [ "$approved_envelope_set" -eq 0 ] || usage
        approved_envelope_json=$2
        approved_envelope_set=1
        shift 2
        ;;
      *) usage ;;
    esac
  done

  if [ "$allow_conflict" -eq 1 ]; then
    [ "$approved_envelope_set" -eq 1 ] || usage
  else
    [ "$approved_envelope_set" -eq 0 ] || usage
  fi
  if [ "$approved_envelope_set" -eq 1 ]; then
    approved_envelope_canonical=$(printf '%s' "$approved_envelope_json" |
      jq -cSe '
        def conflict_rank:
          if . == "closed" then 0
          elif . == "assigned" then 1
          elif . == "in_progress" then 2
          elif . == "active_pull_request" then 3
          else 99 end;
        if type == "object" and .version == 1 and
           (.issue | type == "object" and
             (.host | type == "string" and length > 0) and
             (.repository | type == "string" and length > 0) and
             (.number | type == "number" and . > 0) and
             (.state == "OPEN" or .state == "CLOSED")) and
           (.configured_in_progress_status | type == "string") and
           (.conflicts | type == "array" and length > 0 and
             all(.[]; type == "string") and
             all(.[]; . as $conflict |
             (["closed", "assigned", "in_progress", "active_pull_request"] |
              index($conflict)) != null) and
             length == (unique | length) and
             . == sort_by(conflict_rank)) and
           (.evidence | type == "object" and
             (.assignee_logins | type == "array" and
               all(.[]; type == "string" and length > 0)) and
             (.in_progress_project_items | type == "array" and
               all(.[]; type == "object" and
                 (.title | type == "string" and length > 0) and
                 (.status | type == "string" and length > 0))) and
             (.active_pull_requests | type == "array" and
               all(.[]; type == "object" and
                 (.number | type == "number" and . > 0) and
                 (.state == "OPEN") and
                 (.head_repository | type == "string" and length > 0) and
                 (.head_branch | type == "string" and length > 0))))
        then . else error("invalid approval envelope") end') ||
      fail "approved envelope must be a canonical fix-issue approval object"
  fi

  response=$(GH_HOST="$host" gh issue view "$number" --repo "$repo" \
    --json number,title,body,state,labels,assignees,url,projectItems,closedByPullRequestsReferences) ||
    fail "issue preflight failed for $host/$repo#$number"
  record=$(printf '%s' "$response" | jq -ce \
    --arg host "$host" \
    --arg repo "$repo" \
    --argjson issue_number "$number" \
    --arg configured_status "$in_progress_status" '
    if type == "object" and
       (.number == $issue_number) and
       (.title | type == "string") and
       (.body | type == "string") and
       (.state == "OPEN" or .state == "CLOSED") and
       (.labels | type == "array") and
       (.assignees | type == "array" and all(.[];
          type == "object" and (.login | type == "string" and length > 0))) and
       (.url == ("https://" + $host + "/" + $repo + "/issues/" +
         ($issue_number | tostring))) and
       (.projectItems | type == "array" and all(.[];
         type == "object" and
         (.title | type == "string" and length > 0) and
         ((.status? // null) == null or
           (.status | type == "object" and
             (.name | type == "string" and length > 0))))) and
       (.closedByPullRequestsReferences | type == "array")
    then {
      number,
      title,
      body,
      state,
      labels,
      assignees,
      url,
      project_items: [
        .projectItems[] | {
          title,
          status: ((.status? // null) | if . == null then null else .name end)
        }
      ],
      project_status: (
        ([
          .projectItems[]? |
          .status? |
          if type == "object" then .name else . end |
          select(type == "string" and length > 0)
        ]) as $statuses |
        ((if $configured_status == "" then
            ($statuses | first)
          else
            (($statuses | map(select(
              (. | ascii_downcase) == ($configured_status | ascii_downcase)
            ))) | first) // ($statuses | first)
          end) // null)
      ),
      linked_pull_requests: [
        .closedByPullRequestsReferences[] |
        if type == "object" and
           (.number | type == "number" and . > 0) and
           (.state == "OPEN" or .state == "CLOSED" or .state == "MERGED") and
           (.url == ("https://" + $host + "/" + $repo + "/pull/" +
             (.number | tostring))) and
           (.headRefName | type == "string" and length > 0) and
           (.headRepository.nameWithOwner | type == "string" and length > 0)
        then {
          number,
          state,
          url,
          head_repository: .headRepository.nameWithOwner,
          head_branch: .headRefName
        }
        else error("malformed linked pull request") end
      ]
    }
    else error("malformed issue preflight response") end') ||
    fail "issue preflight returned malformed data for $host/$repo#$number"
  record=$(printf '%s' "$record" | jq -ce \
    --arg host "$host" \
    --arg repo "$repo" \
    --arg configured_status "$in_progress_status" '
    (.assignees | map(.login) | unique | sort) as $assignee_logins |
    (.project_items | map(select(
      $configured_status != "" and .status != null and
      (.status | ascii_downcase) == ($configured_status | ascii_downcase)
    )) | unique | sort_by(.title, .status)) as $in_progress_items |
    (.linked_pull_requests | map(select(.state == "OPEN")) |
      unique | sort_by(.number, .head_repository, .head_branch, .state))
      as $active_pull_requests |
    [
      (if .state == "CLOSED" then "closed" else empty end),
      (if ($assignee_logins | length) > 0 then "assigned" else empty end),
      (if ($in_progress_items | length) > 0 then "in_progress" else empty end),
      (if ($active_pull_requests | length) > 0
       then "active_pull_request" else empty end)
    ] as $conflicts |
    . + {
      conflicts: $conflicts,
      approval_envelope: {
        version: 1,
        issue: {
          host: $host,
          repository: $repo,
          number: .number,
          state: .state
        },
        configured_in_progress_status: ($configured_status | ascii_downcase),
        conflicts: $conflicts,
        evidence: {
          assignee_logins: $assignee_logins,
          in_progress_project_items: $in_progress_items,
          active_pull_requests: $active_pull_requests
        }
      }
    }') ||
    fail "issue preflight conflict classification failed for $host/$repo#$number"

  printf '%s\n' "$record"
  conflict=$(printf '%s' "$record" | jq -er '
    if (.conflicts | index("closed")) != null then 20
    elif (.conflicts | index("assigned")) != null then 21
    elif (.conflicts | index("in_progress")) != null then 22
    elif (.conflicts | index("active_pull_request")) != null then 23
    else 0 end') ||
    fail "issue preflight primary classification failed for $host/$repo#$number"

  if [ "$allow_conflict" -eq 1 ]; then
    current_envelope=$(printf '%s' "$record" | jq -cS '.approval_envelope') ||
      fail "issue preflight approval envelope could not be compared"
    if [ "$current_envelope" != "$approved_envelope_canonical" ]; then
      if [ "$conflict" -eq 0 ]; then
        fail "approved envelope no longer matches the current clean issue"
      fi
      return "$conflict"
    fi
    if [ "$conflict" -ge 21 ] && [ "$conflict" -le 23 ]; then
      return 0
    fi
  fi
  return "$conflict"
}

[ "$#" -ge 1 ] || usage
command=$1
shift
case "$command" in
  repository) repository_context "$@" ;;
  search) search_issues "$@" ;;
  templates) list_templates "$@" ;;
  fix-preflight) fix_preflight "$@" ;;
  *) usage ;;
esac
