# Fetch Pull-Request Feedback

A pull request carries feedback on three separate surfaces. Fetch all three;
otherwise broad review requests and conversation comments are silently lost.

| Surface | GraphQL connection | Resolvable? | Reply location |
| --- | --- | --- | --- |
| Inline review threads | `reviewThreads` | Yes | Inline comment reply |
| Review summary bodies | `reviews` | No | Pull-request conversation |
| Conversation comments | `comments` | No | Pull-request conversation |

Requires `PR_REPO` from [setup](setup.md) and `PR_NUMBER` from
[lookup-pr](lookup-pr.md).

## Fetch a page safely

Keep repository values in GraphQL variables. Building JSON with `jq --arg`
prevents shell expansion of GraphQL dollar-prefixed variables and correctly
quotes repository names.

```bash
case "$PR_REPO" in
  */*) ;;
  *)
    echo "Error: PR_REPO must be an owner/name identity" >&2
    exit 1
    ;;
esac
case "$PR_NUMBER" in
  ""|*[!0-9]*)
    echo "Error: PR_NUMBER must be numeric" >&2
    exit 1
    ;;
esac
PR_REPO_OWNER=${PR_REPO%%/*}
PR_REPO_NAME=${PR_REPO#*/}

FEEDBACK_PAGE=$(mktemp) || exit 1
THREADS_CURSOR=${THREADS_CURSOR:-}
REVIEWS_CURSOR=${REVIEWS_CURSOR:-}
COMMENTS_CURSOR=${COMMENTS_CURSOR:-}

GRAPHQL_QUERY='
query(
  $owner: String!,
  $name: String!,
  $number: Int!,
  $threadsCursor: String,
  $reviewsCursor: String,
  $commentsCursor: String
) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $threadsCursor) {
        nodes {
          id
          isResolved
          isOutdated
          comments(first: 50) {
            nodes {
              id
              databaseId
              body
              path
              line
              originalLine
              diffHunk
              author { login }
              createdAt
            }
            pageInfo { hasNextPage endCursor }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
      reviews(first: 100, after: $reviewsCursor) {
        nodes {
          id
          databaseId
          state
          body
          author { login }
          submittedAt
        }
        pageInfo { hasNextPage endCursor }
      }
      comments(first: 100, after: $commentsCursor) {
        nodes {
          id
          databaseId
          body
          author { login }
          createdAt
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}'

jq -n \
  --arg query "$GRAPHQL_QUERY" \
  --arg owner "$PR_REPO_OWNER" \
  --arg name "$PR_REPO_NAME" \
  --argjson number "$PR_NUMBER" \
  --arg threads_cursor "$THREADS_CURSOR" \
  --arg reviews_cursor "$REVIEWS_CURSOR" \
  --arg comments_cursor "$COMMENTS_CURSOR" \
  '{
    query: $query,
    variables: {
      owner: $owner,
      name: $name,
      number: $number,
      threadsCursor:
        ($threads_cursor | if . == "" then null else . end),
      reviewsCursor:
        ($reviews_cursor | if . == "" then null else . end),
      commentsCursor:
        ($comments_cursor | if . == "" then null else . end)
    }
  }' |
  gh api graphql --input - > "$FEEDBACK_PAGE" || {
    echo "Error: pull-request feedback query failed" >&2
    exit 1
  }
```

Check `.errors` before consuming `.data`. A nonempty error array makes the page
incomplete and must stop the workflow.

## Pagination is mandatory

For each of `reviewThreads`, `reviews`, and `comments`:

1. Read its `pageInfo.hasNextPage`.
2. When true, copy `pageInfo.endCursor` into the matching cursor variable and
   fetch another page.
3. Merge nodes by `id`; never replace an earlier page with a later one.
4. Stop only when every connection reports `hasNextPage: false`.

Each review thread has its own nested `comments` connection. If that
connection reports another page, query that thread by node ID and paginate its
comments independently. The `first` limits above are page sizes, not total
limits.

## Split and classify the surfaces

```bash
PR_PATH='.data.repository.pullRequest'

jq "${PR_PATH}.reviewThreads.nodes |
  map(select(.isResolved == false))" "$FEEDBACK_PAGE"
jq "${PR_PATH}.reviews.nodes |
  map(select(.body != \"\"))" "$FEEDBACK_PAGE"
jq "${PR_PATH}.comments.nodes" "$FEEDBACK_PAGE"
```

Only inline threads carry `isResolved`. Review bodies and conversation comments
remain in API results after they are handled, so keep a per-pull-request ledger
of their node IDs and exclude IDs already recorded. Deduplicate a summary that
only repeats inline findings.

Bot status blocks and walkthroughs with no request are informational. Classify
feedback by its technical content, not the author: short human requests,
out-of-diff findings, and bot comments with actionable items all require
review.

## An early fetch is provisional

Reviewers post asynchronously, and automated reviewers commonly submit several
minutes after a pull request is created or a new head is pushed. A fetch taken
seconds after either event can only report what had already arrived, so treat
it as provisional: an empty result then means "nothing yet", never "no
feedback". Re-fetch all three surfaces after that settling period, and again
whenever the head changes, before anyone treats the pull request as clean.

## Outputs needed by replies

- Inline threads: thread `id`, comment `databaseId`, path, line, and body.
- Review summaries: node `id`, state, author, body, and submission time.
- Conversation comments: node `id`, `databaseId`, author, body, and creation
  time.

Remove temporary page files after their nodes have been merged into the
caller's working data.
