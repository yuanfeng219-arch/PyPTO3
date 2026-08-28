# GitHub Issue Template Interpretation

Interpret only the forms and legacy templates returned by the target
repository's issue-context discovery. Never use a source repository's fixed
template table.

## Select and preserve repository metadata

Read candidate files from `ISSUE_REPO` on `DEFAULT_BRANCH` with the validated
`GITHUB_HOST`. Choose a template whose stated purpose matches the request. If
classification is ambiguous, show the candidates and ask the user.

Preserve the selected template's title prefix and every configured label
exactly. Do not invent a prefix or label, and do not silently replace a missing
label with a familiar one.

## Complete the body

For YAML forms, enumerate every body element whose validation mapping contains
`required: true`. Preserve its label, order, choices, and visible instructions.
Render each completed value as a Markdown section accepted by `gh issue create`.
For legacy templates, preserve their section order and required prompts.

Map only facts supplied by the user or established by read-only evidence. List
each missing required fact and ask for it; never turn a placeholder, generic
sentence, detected local value, or guess into user-provided evidence. Do not
produce a confirmation-ready preview while any required field is missing.

If no suitable repository template exists, render exactly these sections:

```markdown
## Summary

<concise statement>

## Motivation/Impact

<why this matters>

## Acceptance Criteria

<observable completion conditions>
```

Apply the same completeness rule to the fallback. Keep related work as evidence,
not as a reason to overwrite or omit required facts.
