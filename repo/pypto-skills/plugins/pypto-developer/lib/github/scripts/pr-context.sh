#!/usr/bin/env bash

set -u

usage() {
  cat >&2 <<'EOF'
Usage:
  pr-context.sh lookup HOST PR_REPO HEAD_SELECTOR [--allow-none]
  pr-context.sh guard-branch ROLE CURRENT_BRANCH PR_HEAD_BRANCH LOCAL_REPO HEAD_REPO
  pr-context.sh create HOST PR_REPO HEAD_REPO BRANCH BASE TITLE BODY
  pr-context.sh validate-number PR_NUMBER
EOF
  exit 2
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

validate_repo_identity() {
  case "$1" in
    ""|/*|*/|*/*/*) return 1 ;;
    */*) return 0 ;;
    *) return 1 ;;
  esac
}

lookup() {
  [ "$#" -eq 3 ] || [ "$#" -eq 4 ] || usage
  GITHUB_HOST=$1
  PR_REPO=$2
  HEAD_SELECTOR=$3
  ALLOW_NONE=false
  if [ "$#" -eq 4 ]; then
    [ "$4" = "--allow-none" ] || usage
    ALLOW_NONE=true
  fi

  case "$GITHUB_HOST" in
    ""|*/*|*" "*) fail "invalid GitHub host: $GITHUB_HOST" ;;
  esac
  validate_repo_identity "$PR_REPO" ||
    fail "PR_REPO must be an owner/name identity"
  [ -n "$HEAD_SELECTOR" ] || fail "pull-request head selector is empty"

  PR_PAGES=$(gh api --hostname "$GITHUB_HOST" --method GET \
    "repos/$PR_REPO/pulls" \
    -f state=open \
    -f "head=$HEAD_SELECTOR" \
    -f per_page=100 \
    --paginate --slurp) || {
    fail "pull-request lookup failed for $PR_REPO head $HEAD_SELECTOR"
  }
  PR_MATCHES=$(printf '%s' "$PR_PAGES" | jq -ce '
    if type == "array" and all(.[]; type == "array")
    then add
    else error("malformed paginated response")
    end
  ') || fail "malformed pull-request response for $PR_REPO head $HEAD_SELECTOR"

  if ! printf '%s' "$PR_MATCHES" | jq -e '
    type == "array" and
    all(.[];
      type == "object" and
      (.number | type == "number") and
      (.number > 0) and
      (.number == (.number | floor)) and
      (.state | type == "string" and length > 0) and
      (.head | type == "object") and
      (.head.ref | type == "string" and length > 0) and
      (.title | type == "string")
    )
  ' >/dev/null 2>&1; then
    fail "malformed pull-request response for $PR_REPO head $HEAD_SELECTOR"
  fi

  MATCH_COUNT=$(printf '%s' "$PR_MATCHES" | jq -r 'length') ||
    fail "could not count pull-request matches"
  case "$MATCH_COUNT" in
    0)
      if [ "$ALLOW_NONE" != "true" ]; then
        fail "no open pull request has head $HEAD_SELECTOR in $PR_REPO"
      fi
      printf '%s\n' '{"route":"create","match_count":0,"pr":null}'
      ;;
    1)
      printf '%s' "$PR_MATCHES" | jq -c '{
        route: "update",
        match_count: 1,
        pr: (.[0] | {
          number,
          state: (.state | ascii_upcase),
          headRefName: .head.ref,
          title
        })
      }'
      ;;
    *)
      echo "Error: multiple open pull requests have head $HEAD_SELECTOR in $PR_REPO" >&2
      printf '%s' "$PR_MATCHES" |
        jq -r '.[] | "#\(.number) [\(.state)] \(.title)"' >&2
      exit 1
      ;;
  esac
}

guard_branch() {
  [ "$#" -eq 5 ] || usage
  ROLE=$1
  CURRENT_BRANCH=$2
  PR_HEAD_BRANCH=$3
  LOCAL_REPO=$4
  HEAD_REPO=$5
  [ -n "$CURRENT_BRANCH" ] || fail "current branch is empty"
  [ -n "$PR_HEAD_BRANCH" ] || fail "pull-request head branch is empty"
  validate_repo_identity "$LOCAL_REPO" ||
    fail "LOCAL_REPO must be an owner/name identity"
  validate_repo_identity "$HEAD_REPO" ||
    fail "HEAD_REPO must be an owner/name identity"

  case "$ROLE" in
    owner|fork)
      if [ "$PR_HEAD_BRANCH" != "$CURRENT_BRANCH" ]; then
        fail "$ROLE workflow is on $CURRENT_BRANCH but PR head is $PR_HEAD_BRANCH"
      fi
      NORMALIZED_LOCAL_REPO=$(printf '%s' "$LOCAL_REPO" |
        tr '[:upper:]' '[:lower:]')
      NORMALIZED_HEAD_REPO=$(printf '%s' "$HEAD_REPO" |
        tr '[:upper:]' '[:lower:]')
      if [ "$NORMALIZED_HEAD_REPO" != "$NORMALIZED_LOCAL_REPO" ]; then
        fail "$ROLE workflow uses $LOCAL_REPO but PR head repository is $HEAD_REPO"
      fi
      ;;
    maintainer)
      ;;
    *)
      fail "unsupported repository role: $ROLE"
      ;;
  esac
}

create_pull_request() {
  [ "$#" -eq 7 ] || usage
  GITHUB_HOST=$1
  PR_REPO=$2
  HEAD_REPO=$3
  CURRENT_BRANCH=$4
  BASE_BRANCH=$5
  PR_TITLE=$6
  PR_BODY=$7

  case "$GITHUB_HOST" in
    ""|*/*|*" "*) fail "invalid GitHub host: $GITHUB_HOST" ;;
  esac
  validate_repo_identity "$PR_REPO" ||
    fail "PR_REPO must be an owner/name identity"
  validate_repo_identity "$HEAD_REPO" ||
    fail "HEAD_REPO must be an owner/name identity"
  [ -n "$CURRENT_BRANCH" ] || fail "current branch is empty"
  [ -n "$BASE_BRANCH" ] || fail "base branch is empty"
  [ -n "$PR_TITLE" ] || fail "pull-request title is empty"
  [ -n "$PR_BODY" ] || fail "pull-request body is empty"

  NORMALIZED_PR_REPO=$(printf '%s' "$PR_REPO" |
    tr '[:upper:]' '[:lower:]')
  NORMALIZED_HEAD_REPO=$(printf '%s' "$HEAD_REPO" |
    tr '[:upper:]' '[:lower:]')
  HEAD_REPO_ARGUMENTS=()
  if [ "$NORMALIZED_HEAD_REPO" = "$NORMALIZED_PR_REPO" ]; then
    HEAD_SELECTOR=$CURRENT_BRANCH
  else
    HEAD_REPO_OWNER=${HEAD_REPO%%/*}
    HEAD_SELECTOR="$HEAD_REPO_OWNER:$CURRENT_BRANCH"
    PR_REPO_OWNER=$(printf '%s' "${PR_REPO%%/*}" |
      tr '[:upper:]' '[:lower:]')
    NORMALIZED_HEAD_REPO_OWNER=$(printf '%s' "$HEAD_REPO_OWNER" |
      tr '[:upper:]' '[:lower:]')
    if [ "$NORMALIZED_HEAD_REPO_OWNER" = "$PR_REPO_OWNER" ]; then
      HEAD_REPO_NAME=${HEAD_REPO#*/}
      HEAD_REPO_ARGUMENTS=(-f "head_repo=$HEAD_REPO_NAME")
    fi
  fi

  PR_RESPONSE=$(gh api --hostname "$GITHUB_HOST" --method POST \
    "repos/$PR_REPO/pulls" \
    -f "title=$PR_TITLE" \
    -f "body=$PR_BODY" \
    -f "head=$HEAD_SELECTOR" \
    -f "base=$BASE_BRANCH" \
    "${HEAD_REPO_ARGUMENTS[@]}") || {
    fail "pull-request creation failed for $PR_REPO head $HEAD_SELECTOR"
  }

  PR_URL=$(printf '%s' "$PR_RESPONSE" | jq -er '
    if type == "object" and
       (.html_url | type == "string" and length > 0)
    then .html_url
    else error("invalid html_url")
    end
  ' 2>/dev/null) ||
    fail "malformed pull-request creation response for $PR_REPO"
  printf '%s\n' "$PR_URL"
}

validate_number() {
  [ "$#" -eq 1 ] || usage
  case "$1" in
    ""|0*|*[!0-9]*) fail "PR_NUMBER must be a canonical positive integer" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

[ "$#" -ge 1 ] || usage
COMMAND=$1
shift
case "$COMMAND" in
  lookup) lookup "$@" ;;
  guard-branch) guard_branch "$@" ;;
  create) create_pull_request "$@" ;;
  validate-number) validate_number "$@" ;;
  *) usage ;;
esac
