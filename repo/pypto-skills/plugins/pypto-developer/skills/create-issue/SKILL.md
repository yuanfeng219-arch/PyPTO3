---
name: create-issue
description: Use when drafting, checking, or filing a GitHub issue, including bug reports, feature requests, documentation reports, and repository issue forms.
---

# Create Issue

Draft and create one issue against the repository proven by the current Git
checkout. Keep discovery read-only and make the exact approved payload immutable.

## Bound system-test execution

Never run the full system-test suite locally. Run only system-test cases
directly relevant to the changed or requested scope; use CI for the full
system-test suite. If CI cannot run it, report that limitation instead of
substituting a local full-suite run.

## Resolve repository policy

Resolve the [repository scope gate](../../lib/repository/scope.md) before
filing. Creating an issue writes to a remote repository and notifies its
watchers, so a foreign checkout warns with the destination repository and the
intended issue, then waits for explicit confirmation.

Read the authoritative [repository policy](../../lib/repository/policy.md) first.
Discover all applicable repository instructions for the checkout and issue
workflow before choosing issue behavior, metadata, or follow-up mutations. Stop
on missing or conflicting required policy and show the evidence; do not replace
it with source-repository conventions or GitHub defaults.

## Resolve issue context

Read [issue context](../../lib/github/issue-context.md) and invoke the
[issue-context helper](../../lib/github/scripts/issue-context.sh) from that
shared-library path for its `repository`, `templates`, and `search` routes.
Retain `GITHUB_HOST`, `LOCAL_REPO`, `ISSUE_REPO`, and `DEFAULT_BRANCH` from the
returned JSON. Stop on ambiguous identity; do not use an ambient `gh` repository,
a fixed remote, or a familiar organization.

Run the `templates` route, then read [template interpretation](../../lib/github/issue-templates.md).
Select only from the discovered issue forms and legacy templates. Preserve the
repository's title prefix and labels. If none fits, use the documented fallback.

## Search and classify related work

Build focused keywords from the proposed issue, then run the `search` route. It
searches open and closed issues. Deep-read plausible candidates and classify the
result as exactly one of:

- `DUPLICATE #N`: the same request or root cause.
- `RELATED #N...`: overlapping context with a different request or root cause.
- `NO_MATCH`: no material overlap.

Stop only for `DUPLICATE` and report the existing issue. For `RELATED`, continue
and insert `Related: #N` for each related issue in the body. A closed issue or a
shared component is not by itself a duplicate.

## Gather a complete issue

Enumerate every YAML field whose validation contains `required: true`, or every
required legacy-template prompt. Gather each missing user fact explicitly.
Never invent form values, reproduction steps, expected/actual behavior, labels,
project metadata, assignees, or status. Do not present placeholders as complete.

Draft a concise title with the discovered prefix, the discovered labels, and a
body that preserves all selected-template fields in order. Keep the exact body
in a temporary file. Creating that local draft is preparation, not issue
confirmation. Include related references without replacing required content.

## Show the complete issue in text

Post the complete mutation directly in the reply as readable text: the host,
the repository, the title, every label on its own line, and the complete body
exactly as drafted. Show one label per line so a label containing a comma
cannot be read as several labels. Never summarize, truncate, or paraphrase the
body, and never present a draft that still holds a placeholder. Showing the
issue performs no GitHub call.

The text must be exactly what the create command will send. Pass that same
title, label list, and body file to the helper unchanged. If any fact, field,
target, label, or body byte changes afterwards, show the complete issue again
before creating it.

## Wait for explicit confirmation

Ask whether to create exactly the issue shown. Wait for an explicit yes after
the complete text. A request to draft, inspect, or edit is not confirmation. Do
not combine confirmation with an earlier incomplete or superseded draft.

## Create exactly the approved issue

After confirmation, run `scripts/issue-create.sh create HOST REPO TITLE
BODY_FILE LABEL...` with the unchanged values. The helper validates the target,
rejects a control character in the title or a label, and publishes a private
snapshot of the body file, so a later edit of that file cannot reach GitHub
mid-flight. It also rejects a label containing a comma: the CLI reads that as
several labels, so the name cannot be applied as discovered. Report such a
label instead of splitting or renaming it.

Before the mutation boundary it records the exact payload it is about to
publish between `ISSUE_CREATE_PUBLISHING` and `ISSUE_CREATE_PUBLISHING_END`,
read back from that snapshot. The body is copied out verbatim under a declared
`Body (N bytes):` length, and exactly one newline separates it from the end
marker, so trailing newlines are recorded as sent. Nothing binds this route to
the earlier text, so compare the recorded host, repository, title, labels, and
body with what you showed, and report any difference instead of presenting the
issue as approved. A payload that cannot be recorded stops the workflow before
GitHub is called.

Interpret its outcome marker conservatively:

- `ISSUE_CREATE_OUTCOME:confirmed_not_created` means validation stopped before
  the GitHub mutation was invoked.
- `ISSUE_CREATE_OUTCOME:created_response_unvalidated` means GitHub CLI reported
  success but its returned URL could not be validated. Treat the issue as
  created with an unvalidated response, not as a failed creation.
- `ISSUE_CREATE_OUTCOME:unknown` means the create request was invoked but the
  client returned nonzero, so server-side creation cannot be disproved.

The helper accepts only the exact canonical issue URL
`https://HOST/OWNER/REPO/issues/<positive-integer>` for the validated host and
repository. Extra path segments, query strings, fragments, user information,
ports, or lookalike repository names produce `created_response_unvalidated`.

Treat the helper's mutation boundary as authoritative. A signal before that
boundary is `confirmed_not_created`; a signal from the boundary through URL
validation and complete success output is `unknown`. This includes `SIGPIPE`
when the stdout consumer closes early. The helper checks the success write and
does not enter its complete state unless the full URL was delivered.

Do not infer success from a partial URL on stdout. Read outcome diagnostics from
the stderr descriptor saved when the helper started, even if command-local
stderr was redirected later. If that descriptor is also unavailable, the helper
best-effort appends the same sanitized diagnostics to the private mode-0600
stderr response capture without printing raw GitHub output. For every post-write
stop, use `ISSUE_CREATE_VERIFY_TARGET` and
`ISSUE_CREATE_RETRY:blocked...` markers.

For either post-write outcome, do not retry. Keep the helper's private mode-0600
stdout/stderr captures local and do not print or share raw response bytes. First
perform host- and repository-pinned read-only verification, for example:

```bash
GH_HOST="$GITHUB_HOST" gh issue list --repo "$ISSUE_REPO" --state all \
  --limit 100 --search "$EXACT_TITLE" \
  --json number,title,body,labels,url
```

Compare candidate title, complete body, and labels with the approved payload.
If a matching issue exists, report its URL and do not create another. If
eventual consistency or incomplete evidence prevents proving absence, stop and
recheck later or ask the user; do not retry. Only a validated absence plus fresh
explicit approval permits another create attempt. On ordinary success, report
the helper-validated URL.

Treat project metadata as a separate optional mutation. Perform it only when
applicable repository instructions explicitly name the project and every field
needed for the update. Preview and confirm that separate mutation; otherwise
skip it. If it fails after issue creation, report the issue as created and the
metadata update as failed.
