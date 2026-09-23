---
name: auto-pr
description: Use when publishing the current task as a pull request and autonomously repairing objective check or review failures until the PR is green or a bounded stop condition is reached.
---

# Automatic Pull Request

Orchestrate existing publication and repair skills. Own only current-PR scope,
finding decisions, attempt-ledger state, and termination; do not reproduce the
Git, GitHub, review, or verification mechanics owned by composed skills.

## Load the contracts

Resolve the [repository scope gate](../../lib/repository/scope.md) before any
delegation that mutates. A foreign repository is a blocker for this autonomous
workflow: warn, report the gate, and stop rather than confirming on the user's
behalf. Standing authorization from this invocation never covers the gate.

Read [repository policy](../../lib/repository/policy.md) before classifying
style or selecting verification. Then load the installed
[git-commit](../git-commit/SKILL.md), [github-pr](../github-pr/SKILL.md), and
[fix-pr](../fix-pr/SKILL.md) skills. If any contract is unavailable, stop
instead of recreating it.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## Publish and bind one PR

Delegate publication directly to `github-pr`. Do not invoke `git-commit`
directly: `github-pr` must select and validate the route, head, branch, and
write authority before it invokes `git-commit`. Retain the exact host,
repository, number, URL, base, and head that `github-pr` reports. This identity
is the current pull request; `git-commit` is only an indirect dependency here.

Supply `github-pr` standing authorization from this explicit `auto-pr`
invocation covering branch naming, the commit set and its message, and the
pull-request description. Publication runs unattended: never pause to have the
user approve a branch name, a commit count, or a description. Derive each from
the change and report them at the end. Standing authorization covers nothing
else; ownership separation, unavailable policy, failed verification, and every
other stop rule the composed skills own remain in force.

Require a reviewer-ready description. A body that only restates the title or
lists commit subjects is not one; direct `github-pr` to recompose it once, and
treat a second inadequate result as a delegated blocker.

Scope every lookup, delegated repair, recheck, and report to that exact
identity. Reject a missing, ambiguous, closed, or mismatched identity. Never
enumerate, inspect, update, or reuse state from another pull request. Never
invent a missing identity field or substitute an illustrative host,
repository, number, base, or head. Missing PR identity is terminal in every mode.
In a dry run where publication is intentionally not performed, report intended
actions and the unknown identity, then stop; it must not enter classification,
guard, or repair orchestration. In a live workflow, treat missing identity as
a delegated blocker.

Create a fresh task-private attempt ledger outside the consumer repository.
Resolve [the loop helper](scripts/auto-pr-loop.sh) relative to this file; never
substitute a helper from the consumer repository.

## Orchestrate the bounded loop

Run at most eight iterations. Before starting another iteration, stop if it
would be iteration 9. Do not reset the iteration number after a fix, push,
changed head, or reclassification.

For each iteration:

1. Delegate one read-only inventory and state recheck for the current pull
   request to `fix-pr`, covering inline threads, review bodies, conversation
   comments, and check states together. Require stable finding identifiers and
   available evidence; pending checks are not green. Any wait between
   iterations re-runs that whole inventory on each poll, never check states
   alone, and an inventory taken minutes after publication or a push is
   provisional until reconfirmed.
2. **Classify before editing.** Normalize each finding and invoke the helper's
   `classify` subcommand:
   - Use `ci-objective` for a reproducible check failure and `correctness` for
     an objective defect.
   - Use `style-policy` only when repository policy unambiguously requires the
     change. Otherwise treat style preference as `judgment`.
   - Use `architecture`, `product`, or `judgment` when a decision is absent,
     ambiguous, risky, or scope-expanding.
   - Use `informational` or `resolved` only when no repair remains.
   - Preserve the helper's conservative `defer` result for unknown kinds.
3. For every `fix` result, form one stable key from the finding identity and
   normalized root cause. Invoke `guard ITERATION FINDING_KEY LEDGER` before
   authorizing work. The same finding may be attempted at most twice. Exit 20
   means stop before a third attempt; never rename a key to evade the bound.
4. Supply `fix-pr` one composed-authorization envelope for every approved
   `fix`: the exact validated current-PR identity (host, repository, number,
   and head); the unchanged numbered inventory entry and stable finding ID;
   the normalized allowed kind (`ci-objective`, `correctness`, or
   `style-policy`); successful guard iteration and attempt evidence; and the
   standing authorization from the explicit `auto-pr` invocation.
5. Delegate exactly one repair iteration for only those findings to `fix-pr`.
   Let it independently validate the envelope and own evidence gathering,
   edits, repository-local commit and verification policy, publication,
   replies, resolution, and its final recheck. Do not widen the confirmed set.
6. **Rerun repository-required verification after each fix** through
   `fix-pr`, and require its verified current-head result before the next
   inventory. A failed or partial delegated step is a blocker, not success.

Make no code or policy decision for `defer`. Leave every deferred thread
unresolved and do not reply as though it were addressed. Make no repair for
`ignore`; retain it in state if it still affects readiness.

## Stop and report

Stop immediately on the first applicable condition:

- **Deferred judgment:** no objective repair remains, but any unresolved
  architecture, product, policy, or other judgment is required. Deferred judgment takes precedence over green success, even when every required check is green.
- **Green success:** all required checks completed successfully, no objective
  actionable finding remains, and no deferred judgment remains open. Green
  checks alone do not establish this; require a full feedback inventory taken
  after the last check completed.
- **Repeated failure:** the helper reports that one stable finding was already
  attempted twice. Do not make a third attempt.
- **Iteration exhaustion:** iteration 8 completes without reaching a prior
  terminal state. Do not start iteration 9.
- **Delegated blocker:** a composed skill reports ambiguity, unavailable
  policy, failed verification, partial publication, or identity mismatch.

Report the exact current PR identity, the head branch and commit subjects
`github-pr` published, completed iteration count, final check states, objective
fixes and verification evidence, and all still-open items.
Label the outcome as green success, deferred judgment, twice-repeated failure,
eight-iteration exhaustion, or delegated blocker. For every deferred item,
include its stable identifier, location and author when available, concise
decision needed, and confirmation that it remains unresolved. State that no
other pull request was touched.
