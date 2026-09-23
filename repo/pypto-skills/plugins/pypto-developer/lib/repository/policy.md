# Repository Policy Resolution

Use this contract before a workflow chooses verification, review, hooks,
documentation, branch naming, ownership, or commit-message behavior. Consumer
repository policy is authoritative; this shared library supplies no fallback
convention.

## Confirm repository scope first

Resolve the [repository scope gate](scope.md) before the first repository,
remote, or GitHub mutation. A workflow proceeds on its own authority only in a
PTO-family repository; anywhere else it warns, names the intended mutation, and
waits for explicit user confirmation. The discovery below is read-only and may
run before the gate resolves.

## Discover the repository and changed scope

Start with read-only commands from the current checkout:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel) || exit 1
cd "$REPO_ROOT" || exit 1
git status --porcelain=v1 --untracked-files=all
git diff --name-status
git diff --cached --name-status
git ls-files
git config --local --list --show-origin
git log -20 --format='%h%x09%s'
```

Use the status and diffs to list every changed path and the authorized subset as
exact repository-root-relative paths. Existing modifications are user-owned
unless the current task clearly created them. Inspect all changes, but edit,
stage, or commit only paths and hunks whose ownership is established.

Discover written instructions rather than assuming their names. Read files
named by governing instructions, then inspect the repository root and every
ancestor directory of every changed path for instruction and policy documents.
Useful discovery commands include:

```bash
find "$REPO_ROOT" -type f \( -name 'AGENTS.md' -o -name 'CLAUDE.md' \
  -o -name 'CONTRIBUTING*' -o -name 'README*' \) -print
git ls-files '.github/**' 'docs/**' '*CONTRIBUTING*' '*README*'
```

Resolve the complete root-to-leaf chain of applicable nested instruction files
for every changed path, including unrelated paths that will remain untouched. A
nearer file governs only its documented scope; it does not silently erase
compatible higher-level requirements.

## Apply precedence

Apply these levels in order:

1. **User instructions** and governing system instructions.
2. **Applicable repository instructions** for each path, including the resolved
   nested instruction chain.
3. **Documented workflow and configuration**, such as contribution guides,
   configured hooks, CI definitions, and repository-local workflow skills.
4. **Unambiguous local history**, only when written policy is absent and the
   relevant history is consistent.

Higher-precedence policy overrides lower-precedence evidence. History may
confirm missing details; it may not override or reinterpret written policy.

## Stop rules

Stop and ask the user when ownership cannot be separated at file or hunk
granularity, applicable sources conflict at the same precedence, a required
verification command is unavailable, or commit-message requirements remain
ambiguous. Show the conflicting or missing evidence and the exact decision
needed.

Never replace missing policy with a convention from this common repository, a
source repository, a tool default, or personal preference. Never invent a test
command, branch name, remote, commit prefix, trailer, or documentation rule.
