# Common GitHub Workflow Issues

Use these diagnostics with the shared context from [setup](setup.md).

| Issue | Safe response |
| --- | --- |
| GitHub authentication fails | Stop and ask the user to run `gh auth login` |
| Repository identity is empty | Stop; verify the checkout remotes and GitHub host |
| No writable repository for the checkout | Stop; fork the base repository and add that fork as a remote, or request push access. Never retarget the push at a repository the account cannot write |
| Several writable fork remotes | Stop; keep only the authenticated account's fork remote for this checkout |
| No remote matches `PR_REPO` | Add or correct the base-repository remote, then rerun setup |
| No remote matches the writable repository | Add or correct the contributor remote, then rerun setup |
| Default branch cannot be read | Stop rather than substituting a branch name |
| Rebase conflicts | Resolve, stage, and continue; abort when the resolution is unclear |
| Push is rejected | Fetch and inspect the remote branch before retrying |
| Pull request is merged | Stop; do not update a merged pull request |
| Maintainer cannot update a fork | Ask the author to enable maintainer edits and provide the fork remote |

## Shell quoting beside GitHub API writes

Do not place user or review text directly into an unquoted command. Store it in
a variable and quote the expansion:

```bash
REPLY_BODY='Fixed — concise description'
gh api "repos/$PR_REPO/issues/$PR_NUMBER/comments" -f body="$REPLY_BODY"
```

Backticks, dollar signs, exclamation marks, spaces, and newlines in
`REPLY_BODY` remain data. Avoid constructing a double-quoted command string
and executing it.

## GraphQL variables beside GraphQL calls

Do not interpolate repository names or comment text into a GraphQL query.
Build a JSON request with `jq --arg` and pass it through standard input:

```bash
GRAPHQL_QUERY='query($repoId: ID!) { node(id: $repoId) { id } }'
jq -n --arg query "$GRAPHQL_QUERY" --arg repo_id "$REPOSITORY_NODE_ID" \
  '{query: $query, variables: {repoId: $repo_id}}' |
  gh api graphql --input -
```

This protects GraphQL dollar-prefixed variables from shell expansion and lets
`jq` encode values correctly.

## JSON output

Let `gh` or `jq` parse API output. Do not merge standard error into JSON input:

```bash
gh api "repos/$PR_REPO/pulls/$PR_NUMBER" --jq '.state'
gh api "repos/$PR_REPO/pulls/$PR_NUMBER" | jq '.head.ref'
```

If the API call fails, handle its nonzero exit status before parsing.

## Pagination

REST list endpoints and every GraphQL connection are paginated. Use
`gh api --paginate` for REST lists. For GraphQL, request
`pageInfo { hasNextPage endCursor }`, continue from each returned cursor, and
merge nodes by ID. A successful first page is not evidence that the list is
complete.
