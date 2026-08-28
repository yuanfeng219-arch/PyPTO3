# GitHub Issue Context

Use this contract for issue workflows. Keep its validated identity vocabulary
aligned with [GitHub workflow setup](setup.md), but do not depend on pull-request
lookup or ambient GitHub CLI repository selection.

## Repository identity

Run `scripts/issue-context.sh repository` from the checkout. It delegates to
`scripts/repo-identity.sh`, the same identity resolver the pull-request
workflows use, so issue and pull-request identity cannot drift. That resolver
reads every Git remote fetch and push URL, normalizes the host and
`owner/name`, and rejects unrelated hosts or repositories. It queries each
candidate through host-pinned REST calls, maps a fork to its parent issue
repository, and queries that target for its default branch. It never uses
`gh repo view`, whose ambient base-repository selection answers a different
question and prefers a parent over the checkout's own fork.

The JSON output defines these exact values:

| JSON field | Shell name | Meaning |
| --- | --- | --- |
| `github_host` | `GITHUB_HOST` | Validated GitHub host from Git remotes |
| `local_repo` | `LOCAL_REPO` | Validated checkout `owner/name` |
| `issue_repo` | `ISSUE_REPO` | Validated repository that owns issues |
| `default_branch` | `DEFAULT_BRANCH` | Current default branch of `ISSUE_REPO` |

Stop if any value is empty, malformed, or inconsistent with the remote and API
evidence. Do not substitute an authenticated account's ambient repository.

## Duplicate candidates and templates

Run `scripts/issue-context.sh search HOST REPO QUERY`. It returns the complete
JSON array from a host-pinned, repository-pinned search of up to 1,000 open and
closed issues. Each candidate contains `number`, `title`, `body`, `state`,
`labels`, and `url`. Use bodies and root causes, not title similarity alone, to
classify candidates.

Run `scripts/issue-context.sh templates HOST REPO`. It returns a JSON array of
current YAML issue forms and legacy Markdown templates with `name`, `path`, and
`kind`. A missing template directory returns `[]`; authentication, host,
repository, malformed-data, and other API failures stop the workflow.

## Issue and related-work records

When a later workflow needs one issue, keep the read host- and repository-pinned
and retain this exact record shape: `number`, `title`, `body`, `state`, `labels`,
`assignees`, `url`, `project_items`, `project_status`, `linked_pull_requests`,
`conflicts`, and `approval_envelope`. Represent unknown assignment or status as
`null`, not a guessed value. Each linked pull-request record contains `number`,
`state`, `url`, `head_repository`, and `head_branch`.

For `fix-preflight`, the canonical `approval_envelope` is the only conflict
override payload. It binds the host, repository, issue number and state, the
configured in-progress value, the ordered conflict set, sorted assignee logins,
matching project item titles plus statuses, and sorted active pull-request
number/head/state records. An authoritative reread compares the complete
canonical envelope byte-for-byte after JSON key sorting. A replacement object
in the same conflict category does not inherit an earlier approval.

Treat issue reads, duplicate candidates, assignee/status discovery, and linked
pull-request discovery as outputs of this read-only context. Keep every write in
a separate helper with its own preview or approval boundary.
