#!/usr/bin/env bash

set -eu

LC_ALL=C
export LC_ALL

# Keep diagnostics independent from later command-local stderr redirections.
exec 3>&2

BODY_SNAPSHOT=""
CREATE_STATE="idle"
RESPONSE_STDOUT=""
RESPONSE_STDERR=""
TARGET_HOST=""
TARGET_REPO=""

cleanup_snapshot() {
  if [ -n "${BODY_SNAPSHOT:-}" ]; then
    rm -f -- "$BODY_SNAPSHOT"
    BODY_SNAPSHOT=""
  fi
}

cleanup_response_captures() {
  local status=0
  if [ -n "${RESPONSE_STDOUT:-}" ]; then
    if rm -f -- "$RESPONSE_STDOUT"; then
      RESPONSE_STDOUT=""
    else
      status=1
    fi
  fi
  if [ -n "${RESPONSE_STDERR:-}" ]; then
    if rm -f -- "$RESPONSE_STDERR"; then
      RESPONSE_STDERR=""
    else
      status=1
    fi
  fi
  return "$status"
}

trap cleanup_snapshot EXIT

fail() {
  if [ "$CREATE_STATE" = "prewrite" ]; then
    trap '' HUP INT TERM PIPE
    printf '%s\n' 'ISSUE_CREATE_OUTCOME:confirmed_not_created' >&3 2>/dev/null || :
  fi
  printf 'Error: %s\n' "$*" >&3 2>/dev/null || :
  exit 1
}

persist_diagnostic_fallback() {
  local diagnostic=$1
  if [ -n "${RESPONSE_STDERR:-}" ] && [ -f "$RESPONSE_STDERR" ]; then
    printf '\n%s\n' "$diagnostic" >>"$RESPONSE_STDERR" 2>/dev/null || :
  fi
}

emit_diagnostic() {
  local diagnostic=$1
  if ! printf '%s\n' "$diagnostic" >&3 2>/dev/null; then
    persist_diagnostic_fallback "$diagnostic"
  fi
}

post_write_stop() {
  local outcome=$1 status=$2 message=$3 diagnostic
  trap '' HUP INT TERM PIPE
  diagnostic=$(printf '%s\n%s\n%s\n%s\n%s\nError: %s %s' \
    "ISSUE_CREATE_OUTCOME:$outcome" \
    "ISSUE_CREATE_RESPONSE_STDOUT:$RESPONSE_STDOUT" \
    "ISSUE_CREATE_RESPONSE_STDERR:$RESPONSE_STDERR" \
    "ISSUE_CREATE_VERIFY_TARGET:$TARGET_HOST/$TARGET_REPO" \
    'ISSUE_CREATE_RETRY:blocked_pending_read_only_verification' \
    "$message" \
    'Raw responses remain in private mode-0600 files; redact before sharing.')
  emit_diagnostic "$diagnostic"
  exit "$status"
}

handle_signal() {
  local status=$1 signal_name=$2
  case "$CREATE_STATE" in
    prewrite)
      trap '' HUP INT TERM PIPE
      cleanup_response_captures || :
      emit_diagnostic "$(printf '%s\nError: interrupted by %s before the create mutation boundary.' \
        'ISSUE_CREATE_OUTCOME:confirmed_not_created' "$signal_name")"
      exit "$status"
      ;;
    mutation_started | validating_response | reporting_success)
      post_write_stop "unknown" 31 \
        "The create request was interrupted by $signal_name; creation may have occurred."
      ;;
    *) exit "$status" ;;
  esac
}

trap 'handle_signal 129 HUP' HUP
trap 'handle_signal 130 INT' INT
trap 'handle_signal 143 TERM' TERM
trap 'handle_signal 141 PIPE' PIPE

usage() {
  cat >&2 <<'EOF'
Usage:
  issue-create.sh create HOST REPO TITLE BODY_FILE [LABEL...]
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

has_control_character() {
  case "$1" in
    *[[:cntrl:]]*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_payload() {
  local host=$1 repo=$2 title=$3 body_file=$4
  shift 4
  validate_host "$host" || fail "invalid GitHub host: $host"
  validate_repo "$repo" || fail "repository must be an owner/name identity"
  [ -n "$title" ] || fail "issue title is empty"
  if has_control_character "$title"; then
    fail "issue title contains a control character"
  fi
  [ -f "$body_file" ] && [ -r "$body_file" ] ||
    fail "issue body file is not a readable regular file"
  [ -s "$body_file" ] || fail "issue body is empty"
  for label in "$@"; do
    [ -n "$label" ] || fail "issue labels cannot be empty"
    if has_control_character "$label"; then
      fail "issue label contains a control character"
    fi
    # `gh issue create --help` documents `--label "bug,help wanted"` as the form
    # for two labels, so the CLI would split this name instead of applying it.
    # The recorded payload would then claim an identity GitHub never received.
    case "$label" in
      *,*) fail "issue label contains a comma, which the CLI splits into separate labels: $label" ;;
    esac
  done
}

snapshot_body() {
  local source=$1 snapshot_directory
  snapshot_directory=${TMPDIR:-/tmp}
  if has_control_character "$snapshot_directory"; then
    fail "temporary directory contains a control character"
  fi
  [ -d "$snapshot_directory" ] && [ -w "$snapshot_directory" ] ||
    fail "temporary directory is not writable: $snapshot_directory"
  BODY_SNAPSHOT=$(mktemp "$snapshot_directory/issue-create.XXXXXX") ||
    fail "unable to create a private issue body snapshot"
  chmod 600 "$BODY_SNAPSHOT" || fail "unable to protect issue body snapshot"
  cat -- "$source" >"$BODY_SNAPSHOT" || fail "unable to snapshot issue body"
  [ -s "$BODY_SNAPSHOT" ] || fail "snapshotted issue body is empty"
}

# Record the exact bytes that will be published, read back from the private
# snapshot rather than the caller's mutable body file. Nothing binds the create
# route to a payload approved earlier, so a divergence between the approved text
# and the published one must at least be visible. Recording runs before the
# mutation boundary and fails closed, so nothing is created unrecorded.
record_published_payload() {
  local host=$1 repo=$2 title=$3
  shift 3
  local label header body_bytes
  # Never pass the body through command substitution: that strips every
  # trailing newline, so a body differing only in trailing bytes would be
  # recorded as one the user did approve. The bytes are copied out verbatim and
  # framed by a declared length, and exactly one newline separates them from the
  # end marker.
  body_bytes=$(wc -c <"$BODY_SNAPSHOT" | tr -d '[:space:]') ||
    fail "unable to record the payload being published"
  header="ISSUE_CREATE_PUBLISHING
Host: $host
Repository: $repo
Title: $title
Labels ($#):"
  for label in "$@"; do
    header="$header
- $label"
  done
  header="$header
Body ($body_bytes bytes):"
  # Chain with && so the first failure, not the last write, decides the status.
  {
    printf '%s\n' "$header" &&
      cat -- "$BODY_SNAPSHOT" &&
      printf '\n%s\n' 'ISSUE_CREATE_PUBLISHING_END'
  } >&3 || fail "unable to record the payload being published"
}

create_issue() {
  [ "$#" -ge 4 ] || usage
  local host=$1 repo=$2 title=$3 body_file=$4
  local issue_url issue_prefix issue_number capture_directory gh_status
  shift 4
  validate_payload "$host" "$repo" "$title" "$body_file" "$@"
  TARGET_HOST=$host
  TARGET_REPO=$repo
  snapshot_body "$body_file"
  record_published_payload "$host" "$repo" "$title" "$@"
  command -v gh >/dev/null 2>&1 || fail "GitHub CLI is unavailable"

  capture_directory=${TMPDIR:-/tmp}
  RESPONSE_STDOUT=$(mktemp "$capture_directory/issue-create-response.stdout.XXXXXX") ||
    fail "unable to create a private GitHub stdout capture"
  chmod 600 "$RESPONSE_STDOUT" || {
    rm -f -- "$RESPONSE_STDOUT"
    RESPONSE_STDOUT=""
    fail "unable to protect GitHub stdout capture"
  }
  RESPONSE_STDERR=$(mktemp "$capture_directory/issue-create-response.stderr.XXXXXX") || {
    rm -f -- "$RESPONSE_STDOUT"
    RESPONSE_STDOUT=""
    fail "unable to create a private GitHub stderr capture"
  }
  chmod 600 "$RESPONSE_STDERR" || {
    rm -f -- "$RESPONSE_STDOUT" "$RESPONSE_STDERR"
    RESPONSE_STDOUT=""
    RESPONSE_STDERR=""
    fail "unable to protect GitHub stderr capture"
  }

  local arguments=(issue create --repo "$repo" --title "$title" --body-file "$BODY_SNAPSHOT")
  for label in "$@"; do
    arguments+=(--label "$label")
  done
  gh_status=0
  # This is the logical mutation boundary. Everything before it is confirmed
  # local preparation; every interruption after it is conservatively unknown.
  CREATE_STATE="mutation_started"
  GH_HOST="$host" gh "${arguments[@]}" 3>&- \
    >"$RESPONSE_STDOUT" 2>"$RESPONSE_STDERR" ||
    gh_status=$?
  if [ "$gh_status" -ne 0 ]; then
    post_write_stop "unknown" 31 \
      "GitHub CLI returned nonzero after the create request; creation may have occurred."
  fi
  CREATE_STATE="validating_response"
  issue_url=$(cat -- "$RESPONSE_STDOUT") ||
    post_write_stop "created_response_unvalidated" 30 \
      "GitHub CLI reported success but its stdout could not be read."
  if has_control_character "$issue_url"; then
    post_write_stop "created_response_unvalidated" 30 \
      "GitHub CLI reported success but returned an unsafe issue URL."
  fi
  issue_prefix="https://$host/$repo/issues/"
  case "$issue_url" in
    "$issue_prefix"*) ;;
    *) post_write_stop "created_response_unvalidated" 30 \
         "GitHub CLI reported success but returned an unexpected issue URL." ;;
  esac
  issue_number=${issue_url#"$issue_prefix"}
  case "$issue_number" in
    "" | 0* | *[!0-9]*) post_write_stop "created_response_unvalidated" 30 \
      "GitHub CLI reported success but returned an invalid issue URL." ;;
  esac
  CREATE_STATE="reporting_success"
  if ! printf '%s\n' "$issue_url"; then
    post_write_stop "unknown" 31 \
      "The validated issue URL could not be delivered on stdout; creation occurred."
  fi
  CREATE_STATE="complete"
  if ! cleanup_response_captures; then
    printf '%s\n' \
      'Warning: issue URL validated, but one or more private response captures could not be removed.' >&2
  fi
}

[ "$#" -ge 1 ] || usage
command=$1
shift
case "$command" in
  create)
    CREATE_STATE="prewrite"
    create_issue "$@"
    ;;
  *) usage ;;
esac
