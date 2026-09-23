#!/usr/bin/env bash

set -u

fail() {
  echo "Error: $*" >&2
  exit 1
}

[ "$#" -eq 2 ] ||
  fail "worktree validation requires PREPARED_HEAD_OID and VALIDATION_COMMAND"
PREPARED_HEAD_OID=$1
VALIDATION_COMMAND=$2

case "$PREPARED_HEAD_OID" in
  ""|*[!0-9a-fA-F]*) fail "PREPARED_HEAD_OID is not a full Git object ID" ;;
esac
case "${#PREPARED_HEAD_OID}" in
  40|64) ;;
  *) fail "PREPARED_HEAD_OID is not a full Git object ID" ;;
esac
[ -n "$VALIDATION_COMMAND" ] || fail "VALIDATION_COMMAND must not be empty"

# Resolves in the main checkout and in a linked `git worktree add` checkout
# alike, and from any subdirectory of either.
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) ||
  fail "validation must run inside a Git checkout"
cd "$REPO_ROOT" || fail "failed to enter the Git checkout"

# Pin every reporting mode that configuration could otherwise relax:
# `status.showUntrackedFiles=no` hides untracked files, `submodule.<name>.ignore`
# hides a submodule checked out away from its gitlink, `core.fileMode=false`
# hides executable-bit drift, and `core.symlinks=false` lets a regular file stand
# in for a recorded symlink. A checkout whose filesystem cannot represent modes
# or symlinks fails loudly here rather than validating a differing tree.
checkout_status() {
  git -c core.fileMode=true -c core.symlinks=true status \
    --porcelain=v1 --untracked-files=all --ignore-submodules=none
}

# Skip-worktree (including sparse checkout) and assume-unchanged suppress status
# reporting for tracked paths, so a clean status cannot prove that the checkout
# matches HEAD. Populated submodules keep their own indices, and
# `git ls-files --recurse-submodules` supports only `--cached` and so cannot
# report flag letters; walk the submodules explicitly instead. Submodule paths
# are passed as printf arguments, never interpolated into a program text, so a
# path containing shell or expression metacharacters cannot break the scan.
hidden_index_entries() {
  local super submodules
  super=$(git ls-files -v) || return 1
  printf '%s\n' "$super" | grep -E '^([a-z]|S) ' || :
  submodules=$(git submodule foreach --quiet --recursive \
    'git ls-files -v | grep -E "^([a-z]|S) " |
       while IFS= read -r entry; do
         printf "%s %s/%s\n" "${entry%% *}" "$displaypath" "${entry#* }"
       done') || return 1
  [ -z "$submodules" ] || printf '%s\n' "$submodules"
}

require_no_hidden_entries() {
  local stage=$1 entries
  entries=$(hidden_index_entries) ||
    fail "failed to inspect index state $stage validation"
  [ -z "$entries" ] ||
    fail "index hides tracked-file drift $stage validation; clear skip-worktree
and assume-unchanged on these entries:
$entries"
}

git cat-file -e "$PREPARED_HEAD_OID^{commit}" ||
  fail "prepared validation commit is unavailable"
ACTUAL_HEAD_OID=$(git rev-parse HEAD) || fail "failed to read HEAD"
[ "$ACTUAL_HEAD_OID" = "$PREPARED_HEAD_OID" ] ||
  fail "worktree HEAD $ACTUAL_HEAD_OID is not the prepared commit $PREPARED_HEAD_OID"

require_no_hidden_entries before
STATUS_BEFORE=$(checkout_status) ||
  fail "failed to inspect worktree status"
[ -z "$STATUS_BEFORE" ] ||
  fail "worktree must be clean before validation"

/bin/sh -eu -c "$VALIDATION_COMMAND" ||
  fail "repository-selected validation failed"

# Repeat both gates: the validation command runs repository code with write
# access, so it can set these flags itself after the first scan.
require_no_hidden_entries after
STATUS_AFTER=$(checkout_status) ||
  fail "failed to inspect worktree status after validation"
[ -z "$STATUS_AFTER" ] ||
  fail "validation left the worktree dirty; ignore or remove these paths:
$STATUS_AFTER"
