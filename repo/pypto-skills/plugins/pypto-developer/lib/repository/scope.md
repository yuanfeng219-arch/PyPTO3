# Repository Scope Gate

Resolve this gate before the first repository, remote, or GitHub mutation in
any workflow. It decides whether a skill may act on its own authority or must
first obtain explicit user confirmation for this specific repository.

These workflows run automatically only in a PTO-family repository. They still
work anywhere else, but only after the user confirms that repository by name.
Read-only inspection is never gated; the gate governs mutation.

A fetch is a mutation for this purpose. It contacts a remote and writes objects,
remote-tracking refs, and `FETCH_HEAD` into the local repository, so it belongs
after the gate even though it changes nothing on the remote.

## Classify the checkout

Collect identity from the checkout rather than from the invocation:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "Error: the current directory is not in a Git worktree" >&2
  exit 1
}
cd "$REPO_ROOT" || exit 1

SCOPE_MARKER='(^|[^[:alnum:]])(py)?pto([^[:alnum:]]|$)'
SCOPE_DECLARATION='(^|[^[:alnum:]])pto-family-repository([^[:alnum:]]|$)'
REPO_SCOPE=foreign
SCOPE_DISCOVERY=ok

# Emit only the repository path of a remote URL. The host and any userinfo are
# never identity: `https://pto.example.com/acme/unrelated` names no family
# repository, and matching the whole URL would hand every repository on such a
# host an unearned `family`.
scope_url_path() {
  case "$1" in
    *://*)
      SCOPE_PATH=${1#*://}
      case "$SCOPE_PATH" in
        */*) SCOPE_PATH=${SCOPE_PATH#*/} ;;
        *) return 0 ;;
      esac
      ;;
    *:*)
      SCOPE_PATH=${1#*:}
      ;;
    *)
      SCOPE_PATH=$(basename -- "$1") || return 1
      ;;
  esac
  printf '%s\n' "$SCOPE_PATH"
}

collect_scope_identity() {
  basename -- "$REPO_ROOT" || return 1

  SCOPE_REMOTES=$(git remote) || return 1
  while IFS= read -r SCOPE_REMOTE; do
    [ -n "$SCOPE_REMOTE" ] || continue
    SCOPE_URLS=$(git remote get-url --all "$SCOPE_REMOTE") || return 1
    SCOPE_PUSH_URLS=$(git remote get-url --push --all "$SCOPE_REMOTE") ||
      return 1
    while IFS= read -r SCOPE_URL; do
      [ -n "$SCOPE_URL" ] || continue
      scope_url_path "$SCOPE_URL" || return 1
    done <<URLS
$SCOPE_URLS
$SCOPE_PUSH_URLS
URLS
  done <<EOF
$SCOPE_REMOTES
EOF
}

collect_scope_declarations() {
  SCOPE_FILES=$(git ls-files -- 'AGENTS.md' 'CLAUDE.md' '*/AGENTS.md' \
    '*/CLAUDE.md' 'README*' 'CONTRIBUTING*') || return 1
  while IFS= read -r SCOPE_FILE; do
    [ -n "$SCOPE_FILE" ] || continue
    [ -r "$SCOPE_FILE" ] || return 1
    grep -Eih "$SCOPE_DECLARATION" -- "$SCOPE_FILE" || [ $? -eq 1 ] || return 1
  done <<EOF
$SCOPE_FILES
EOF
}

SCOPE_IDENTITY=$(collect_scope_identity) || SCOPE_DISCOVERY=failed
SCOPE_DECLARATIONS=$(collect_scope_declarations) || SCOPE_DISCOVERY=failed

if [ "$SCOPE_DISCOVERY" = ok ] && {
     printf '%s\n' "$SCOPE_IDENTITY" | grep -Eiq "$SCOPE_MARKER" ||
     printf '%s\n' "$SCOPE_DECLARATIONS" | grep -Eiq "$SCOPE_DECLARATION"
   }; then
  REPO_SCOPE=family
fi
printf 'REPO_SCOPE=%s\n' "$REPO_SCOPE"
```

`family` means one of two things: the repository name or the repository path of
one of its configured fetch or push URLs carries the marker, or the repository
affirmatively declares itself by writing the literal token
`pto-family-repository` in one of its own governing instruction files.

Only the path is identity. A remote URL's host and userinfo are infrastructure,
not membership, so they are stripped before matching — a checkout hosted at
`pto.example.com` or fetched as `pto@host:acme/unrelated` stays `foreign`.

Identity and declaration are matched by different patterns on purpose. A name
or a URL path is an identifier, so the marker is enough. Prose is not: a README
that merely mentions the family — including a negative mention such as "not
compatible with PyPTO" — is a topic, not a claim of membership, and must never
classify a checkout as `family`. A family repository whose name carries no
marker opts in by adding the declaration token, never by talking about the
family.

Anything else is `foreign`, including a checkout whose evidence cannot be
collected: an unreadable remote, a missing identity, or a failed command
classifies as `foreign` rather than passing unclassified. Fail closed.

`REPO_SCOPE` therefore starts at `foreign` and is raised only by a complete
evidence sweep. Every identity command is status-checked, and one failure
abandons the whole sweep: a marker already emitted by the directory name or an
earlier remote must not carry a checkout whose remaining evidence never came
back. Never soften a command failure into an empty result.

Never widen the marker to make a repository pass, and never treat a familiar
directory, a previous session, or the user's usual project as evidence.

## Warn and require explicit confirmation

For `REPO_SCOPE=foreign`, stop before the first mutation and show exactly what
is at stake:

```text
Repository scope warning
  Repository: <REPO_ROOT> (<resolved identity, or "unresolved">)
  Remotes:    <remote name and URL per configured remote>
  Evidence:   <why the checkout classified as foreign>
  Skill:      <invoked skill>
  Intended:   <the exact first mutation and its targets>

These workflows act automatically only in a PTO-family repository. Confirm
explicitly to run <skill> against <repository> here.
```

Wait for a confirmation that names this repository. Then:

- Confirmation covers this repository and this invocation only. It does not
  transfer to another repository, another skill, a later invocation, or a
  later session.
- Approval given before this warning was shown is not confirmation, and
  neither is a confirmation recorded for a different repository.
- Standing authorization from an autonomous caller such as `auto-pr` does not
  cover this gate. An autonomous run in a foreign repository stops here and
  reports the gate as a blocker instead of proceeding.
- Confirmation authorizes the workflow to run here. It supplies no verification
  command, branch name, remote, or commit convention, and it overrides no
  ownership, policy, or stop rule.

Continue read-only work while waiting. Record the confirmation with the
repository it names, and report the gate result alongside the workflow's other
evidence.
