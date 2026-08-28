#!/usr/bin/env bash

set -u
export LC_ALL=C

usage() {
    printf 'usage: stage-owned.sh PATH...\n' >&2
}

fail() {
    local code=$1
    shift
    printf '%s\n' "$*" >&2
    exit "$code"
}

if [ "$#" -eq 0 ]; then
    usage
    exit 2
fi

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || \
    fail 2 'not inside a Git worktree'
cd "$repo_root" || fail 2 'cannot enter the repository root'

declare -A authorized=()
declare -a authorized_order=()

for path in "$@"; do
    case "$path" in
        '' | /* | . | ./* | */./* | */. | .. | ../* | */../* | */.. | \
            */ | *//* | :* | *'*'* | *'?'* | *'['* | *']'*)
            usage
            fail 2 "invalid/non-exact repo-relative path: $path"
            ;;
    esac
    if [ -d "$path" ] && [ ! -L "$path" ]; then
        path_mode=$(git ls-files --stage -- "$path")
        path_mode=${path_mode%% *}
        [ "$path_mode" = 160000 ] || \
            fail 2 "directory path is not allowed: $path"
    fi
    if [[ -n ${authorized["$path"]+present} ]]; then
        fail 2 "duplicate authorized path: $path"
    fi
    authorized["$path"]=1
    authorized_order+=("$path")
done

before_file=$(mktemp "${TMPDIR:-/tmp}/stage-owned-before.XXXXXX") || \
    fail 5 'cannot create staging verification file'
status_file=$(mktemp "${TMPDIR:-/tmp}/stage-owned-status.XXXXXX") || {
    rm -f -- "$before_file"
    fail 5 'cannot create status verification file'
}
after_file=$(mktemp "${TMPDIR:-/tmp}/stage-owned-after.XXXXXX") || {
    rm -f -- "$before_file" "$status_file"
    fail 5 'cannot create post-stage verification file'
}
trap 'rm -f -- "$before_file" "$status_file" "$after_file"' EXIT

git diff --cached --name-only -z >"$before_file" || \
    fail 5 'cannot inspect the staged set'
while IFS= read -r -d '' staged_path; do
    if [[ -z ${authorized["$staged_path"]+present} ]]; then
        fail 3 "already-staged path is outside the authorized set: $staged_path"
    fi
done <"$before_file"

for path in "${authorized_order[@]}"; do
    git status --porcelain=v1 -z -- "$path" >"$status_file" || \
        fail 5 "cannot inspect requested path: $path"
    [ -s "$status_file" ] || fail 4 "requested path has no change: $path"
done

for path in "${authorized_order[@]}"; do
    git add -- "$path" || fail 5 "git add failed for authorized path: $path"
done

git diff --cached --name-only -z >"$after_file" || \
    fail 5 'cannot verify the staged set'

declare -A staged_set=()
while IFS= read -r -d '' staged_path; do
    if [[ -z ${authorized["$staged_path"]+present} ]]; then
        fail 5 "post-stage set contains an unauthorized path: $staged_path"
    fi
    staged_set["$staged_path"]=1
done <"$after_file"

for path in "${authorized_order[@]}"; do
    if [[ -z ${staged_set["$path"]+present} ]]; then
        fail 5 "post-stage set is missing an authorized path: $path"
    fi
done

while IFS= read -r -d '' staged_path; do
    printf '%s\n' "$staged_path"
done <"$after_file"
