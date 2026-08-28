# PyPTO Skills

This repository is the canonical source for portable skills shared across
PyPTO-related repositories. It publishes separate developer and user plugins
for Claude Code and Codex while keeping repository-specific workflows in their
own repositories.

## Plugins

### `pypto-developer`

Reusable repository development workflows:

- `clean-branches`
- `github-pr`
- `fix-pr`
- `git-commit`
- `create-issue`
- `fix-issue`
- `auto-pr`

These skills discover and honor the consumer repository's policies, test
commands, issue forms, and commit conventions.

### `pypto-user`

User-facing workflows shared by PyPTO and pypto-lib:

- `generate-ir-trace`
- `incore-profiling`
- `setup-and-run`

`setup-and-run` owns the sequence and the gates that take a new user from a
fresh checkout to a validated model run; the consumer repository still owns its
own setup, platform, and model documentation.

Repository-specific setup, testing, review, kernel-style, and model workflows
remain in their owning repositories.

## Installation

Add this repository as a marketplace, then install either or both plugins.

Codex:

```bash
codex plugin marketplace add hw-native-sys/pypto-skills
codex plugin add pypto-developer@pypto-skills
codex plugin add pypto-user@pypto-skills
```

Claude Code:

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-developer@pypto-skills
claude plugin install pypto-user@pypto-skills
```

## Layout

- `plugins/pypto-developer/` contains the developer plugin and its shared Git
  and GitHub references.
- `plugins/pypto-user/` contains the user plugin and its bundled profiling
  helpers.
- `.agents/plugins/marketplace.json` publishes the Codex marketplace.
- `.claude-plugin/marketplace.json` publishes the Claude Code marketplace.
- `skills` and `lib` are compatibility symlinks for existing submodule
  consumers of the developer bundle.
- `tests/` validates skill structure, plugin manifests, local links, and
  portability.

## Validation

Run the standard-library test suite with:

```bash
python -m unittest discover -s tests -v
```

Install the pinned CI tools and run the static checks with:

```bash
python -m pip install --requirement requirements-ci.txt
ruff check tests
ruff format --check tests
pyright
git ls-files -z -- '*.sh' | xargs -0 -r -n 1 bash -n
```

The push transaction runs repository-selected validation in the working
checkout at exactly the commit it is about to push. Executing that repository
code is governed by the harness permission controls that already gate the rest
of the workflow; there is no separate isolation boundary to configure.
