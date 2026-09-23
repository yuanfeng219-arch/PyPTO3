#!/usr/bin/env bash
set -eu

usage() {
    echo "Usage: auto-pr-loop.sh classify KIND" >&2
    echo "       auto-pr-loop.sh guard ITERATION FINDING_KEY LEDGER" >&2
    exit 2
}

classify() {
    [ "$#" -eq 1 ] || usage

    case "$1" in
        ci-objective|correctness|style-policy)
            printf '%s\n' fix
            ;;
        architecture|product|judgment)
            printf '%s\n' defer
            ;;
        informational|resolved)
            printf '%s\n' ignore
            ;;
        *)
            printf '%s\n' defer
            ;;
    esac
}

guard() {
    [ "$#" -eq 3 ] || usage

    local iteration=$1
    local finding_key=$2
    local ledger=$3
    local ledger_directory ledger_name lock_directory temporary_ledger=""
    local lock_attempt=0 previous_count next_count

    case "$iteration" in
        ''|*[!0-9]*)
            echo "Error: iteration must be an integer from 1 through 8" >&2
            exit 21
            ;;
    esac
    while [ "${#iteration}" -gt 1 ] && [ "${iteration#0}" != "$iteration" ]; do
        iteration=${iteration#0}
    done
    case "$iteration" in
        1|2|3|4|5|6|7|8) ;;
        *)
            echo "Error: iteration must be an integer from 1 through 8" >&2
            exit 21
            ;;
    esac

    if [ -z "$finding_key" ] || [[ "$finding_key" == *$'\n'* ]] \
        || [[ "$finding_key" == *$'\r'* ]] \
        || [[ "$finding_key" == *$'\t'* ]]; then
        echo "Error: finding key must be one non-empty ledger field" >&2
        exit 22
    fi
    if [ -z "$ledger" ] || [[ "$ledger" == *$'\n'* ]]; then
        echo "Error: ledger path must be non-empty and single-line" >&2
        exit 22
    fi

    ledger_directory=$(dirname -- "$ledger")
    ledger_name=$(basename -- "$ledger")
    if [ ! -d "$ledger_directory" ]; then
        echo "Error: ledger directory does not exist: $ledger_directory" >&2
        exit 22
    fi

    lock_directory="${ledger}.lock"
    until mkdir -- "$lock_directory" 2>/dev/null; do
        lock_attempt=$((lock_attempt + 1))
        if [ "$lock_attempt" -ge 200 ]; then
            echo "Error: attempt ledger is locked: $ledger" >&2
            exit 22
        fi
        sleep 0.01
    done
    cleanup() {
        if [ -n "$temporary_ledger" ] && [ -e "$temporary_ledger" ]; then
            rm -f -- "$temporary_ledger" || true
        fi
        rmdir -- "$lock_directory" 2>/dev/null || true
    }
    trap cleanup EXIT
    trap 'cleanup; exit 22' HUP INT TERM

    previous_count=0
    if [ -e "$ledger" ]; then
        if [ ! -f "$ledger" ]; then
            echo "Error: attempt ledger is not a regular file: $ledger" >&2
            exit 22
        fi
        previous_count=$(awk -F '\t' -v key="$finding_key" '
            NF != 2 || $1 == "" || $2 !~ /^[12]$/ { invalid = 1 }
            $1 == key && ($2 + 0) > count { count = $2 + 0 }
            END {
                if (invalid) {
                    exit 1
                }
                print count + 0
            }
        ' "$ledger") || {
            echo "Error: attempt ledger contains an invalid record" >&2
            exit 22
        }
    fi

    if [ "$previous_count" -ge 2 ]; then
        echo "Error: finding already attempted twice: $finding_key" >&2
        exit 20
    fi
    next_count=$((previous_count + 1))

    temporary_ledger=$(mktemp \
        "$ledger_directory/.${ledger_name}.tmp.XXXXXX") || {
        echo "Error: failed to create temporary attempt ledger" >&2
        exit 22
    }
    if [ -f "$ledger" ]; then
        cat -- "$ledger" >"$temporary_ledger"
    fi
    printf '%s\t%s\n' "$finding_key" "$next_count" >>"$temporary_ledger"
    if ! mv -- "$temporary_ledger" "$ledger"; then
        echo "Error: failed to replace attempt ledger" >&2
        exit 22
    fi
    temporary_ledger=""
    cleanup
    trap - EXIT HUP INT TERM
}

[ "$#" -ge 1 ] || usage
subcommand=$1
shift

case "$subcommand" in
    classify)
        classify "$@"
        ;;
    guard)
        guard "$@"
        ;;
    *)
        usage
        ;;
esac
