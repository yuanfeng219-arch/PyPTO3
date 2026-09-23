# Issues for hw-native-sys/pypto-skills

Downloaded: 2026-08-17T19:39:43.5522024+08:00
Total issues: 2

## #5 setup.md resolves LOCAL_REPO from the gh default repository, selecting the wrong push target in a fork checkout

- State: open
- URL: https://github.com/hw-native-sys/pypto-skills/issues/5
- Created: 2026-08-06T04:59:39Z
- Updated: 2026-08-06T04:59:39Z
- Closed: 

### Body

## Summary

`lib/github/setup.md` §2 derives `LOCAL_REPO` with `gh repo view --json nameWithOwner,url` (line 39). Invoked with no argument, `gh repo view` returns the **gh default repository**, which `gh repo set-default` stores in the consuming repository's own Git config as `remote.<name>.gh-resolved = base`. In a fork checkout whose default points at the parent, `LOCAL_REPO` becomes the parent — not "the repository represented by this checkout", which is how §2 defines it.

Every downstream derivation inherits the wrong identity. Observed in a fork checkout with `origin = Hzfengsy/pypto` and `upstream = hw-native-sys/pypto`:

| setup.md line | Variable | Expected (fork checkout) | Actual |
| --- | --- | --- | --- |
| 43 | `LOCAL_REPO` | `Hzfengsy/pypto` | `hw-native-sys/pypto` |
| 61 | `IS_FORK` | `true` | `false` |
| 63-67 | `PR_REPO` | `hw-native-sys/pypto` | `hw-native-sys/pypto` |
| 171 | `PUSH_REMOTE` | `origin` | `upstream` |
| 207-213 | `ROLE` / `PR_HEAD_PREFIX` | `fork` / `Hzfengsy:` | `owner` / *(empty)* |

The net effect is that the workflow is configured to push a feature branch **directly into the shared upstream repository** instead of the contributor's fork.

The failure is silent, and the existing guards do not catch it because they verify consistency against the already-wrong identity. `pr-context.sh guard-branch` compares `LOCAL_REPO` with `HEAD_REPO`; both are wrong in the same direction, so it passes. The same applies to the `remote_targets_repo "$PUSH_REMOTE" "$EXPECTED_PUSH_REPO"` check in `commit-and-push.md`, since `EXPECTED_PUSH_REPO` is derived from the same poisoned `LOCAL_REPO`.

## Motivation/Impact

This resolves toward the higher-blast-radius target: it silently redirects a push from a personal fork to a shared organization repository. A contributor who follows the documented workflow can create branches directly on the upstream repo without any step reporting the change of destination.

`gh repo set-default` is not an exotic configuration. `gh` interactively prompts for it the first time you run `gh pr list` / `gh issue list` in a checkout with more than one remote, so any fork clone with both `origin` and `upstream` is a candidate. The setting lives in the consuming repository's Git config, so it is invisible to anyone reading only the skill sources.

Reproduction, entirely read-only:

```bash
cd <fork-checkout>            # origin=<you>/<repo>, upstream=<org>/<repo>
git config --local --get-regexp 'remote.*gh-resolved'
# remote.upstream.gh-resolved base

gh repo view --json nameWithOwner --jq .nameWithOwner
# <org>/<repo>                <-- the parent, not this checkout

gh repo view --json nameWithOwner --jq .nameWithOwner --repo <you>/<repo>
# <you>/<repo>                <-- what setup.md §2 intends
```

Found while running `auto-pr` against `hw-native-sys/pypto`. It was caught only because the operator inspected `git remote -v` by hand before the push and noticed the mismatch; nothing in the workflow surfaced it.

## Acceptance Criteria

- `LOCAL_REPO` is derived from the checkout's own remotes (for example the remote the current branch tracks, else `origin`), not from an argument-less `gh repo view`.
- When the checkout-derived identity and the gh default repository disagree, the workflow stops with an actionable message naming both values, rather than silently preferring either one.
- A fork checkout with `remote.<upstream>.gh-resolved = base` resolves `ROLE=fork`, `PUSH_REMOTE=origin`, `PR_HEAD_PREFIX=<owner>:`, and `PR_REPO=<parent>`.
- A non-fork checkout with the same setting still resolves `ROLE=owner` correctly.
- `lib/github/setup.md` §2 documents the interaction with `gh repo set-default` and states that the gh default repository is not a source of checkout identity.
- Regression coverage under `tests/` (alongside `test_github_pr.py` / `test_commit_and_push.py`) for the fork-with-`gh-resolved`-parent topology.


---

## #6 Validation sandbox cannot run validation for native-build repositories, and its documented escape hatch has no mechanism

- State: closed
- URL: https://github.com/hw-native-sys/pypto-skills/issues/6
- Created: 2026-08-06T04:59:56Z
- Updated: 2026-08-06T11:33:47Z
- Closed: 2026-08-06T11:33:47Z

### Body

## Summary

`lib/github/commit-and-push.md` requires the push transaction to validate inside `scripts/validation-sandbox.sh`, forbids any credentialed fallback, and offers exactly one escape hatch: "stop and require an explicitly trusted project runner that enforces the same credential-free, network-denied boundary". No mechanism exists to declare such a runner, so for any repository whose validation needs a native build the workflow has no supported path to completion.

Measured against `hw-native-sys/pypto` (C++ extension built with CMake + nanobind, one Git submodule). Every element of that repository's documented validation fails inside the sandbox:

| Repository-defined validation | Sandbox result | Cause |
| --- | --- | --- |
| `cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo` | `Configuring incomplete, errors occurred!` | `find_package(nanobind CONFIG REQUIRED)` fails; nanobind lives in the host Python environment, which the sandbox correctly hides |
| C++ link step | would fail | `3rdparty/msgpack-c` is empty — `git archive` does not include submodule contents |
| `python -m pytest tests/ut/` | cannot run | `pytest` absent; no network to install it |
| `python tests/lint/check_headers.py` | `Error: '/workspace' is not a git repository` | the repo's own lint scripts enumerate via `git ls-files`; the sandbox strips Git metadata by design |

The last row is the notable one: even the pure-Python checks fail, so this is not merely "native builds are hard". Probed by invoking `validation-sandbox.sh` directly with a diagnostic command; `bwrap` 0.8.0 was present and working, so this is not a missing-runtime case.

Two distinguishable problems:

1. **No way to declare a trusted runner.** The contract names the escape hatch but provides no config key, file, or convention for supplying one, and correctly forbids selecting one from the worktree. The instruction is therefore unactionable.
2. **`git archive` silently drops submodule contents.** `validation-sandbox.sh` builds the snapshot with `git archive --format=tar "$PREPARED_HEAD_OID"`. For a repository with submodules the extracted tree is incomplete, and the omission is silent — validation could fail for a reason unrelated to the change, or in principle pass while never compiling the omitted code.

A third, ordering issue: sandbox feasibility is knowable before any repository mutation, but the blocker surfaces at the push gate. In the observed run a branch had already been created, changes staged, and a commit written before the workflow discovered it could not validate or push.

## Motivation/Impact

`pypto-developer` is published for `hw-native-sys/pypto`, and `auto-pr` / `github-pr` / `fix-pr` cannot complete their push transaction there. The operator's only options are to abandon the workflow or to override the documented boundary, and the second is what actually happened in the observed run — which is the outcome the boundary exists to prevent. An unactionable safety rule tends to get bypassed rather than obeyed.

The submodule gap is independently worth fixing: it silently changes what gets validated, for any consumer with submodules, whether or not the trusted-runner question is resolved.

## Acceptance Criteria

- A documented mechanism exists for a repository to declare a trusted validation runner, resolved from a location outside the consuming worktree so the existing threat model is preserved, and `commit-and-push.md` references it where it currently says "require an explicitly trusted project runner".
- `validation-sandbox.sh` produces a complete snapshot for a repository with submodules, or fails loudly when it cannot, rather than silently extracting an incomplete tree.
- The skills detect that validation cannot run in the available sandbox **before** creating a branch, staging, or committing, and report it while the worktree is still unmodified.
- The reported blocker names which specific validation requirement could not be met (missing toolchain, submodule contents, Git metadata, or network) instead of a generic failure.
- `tests/test_commit_and_push.py` covers the submodule snapshot case and the declared-runner path.


---

