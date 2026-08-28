# Reply to and Resolve Feedback

Use the three surfaces returned by [fetch-comments](fetch-comments.md). Every
actionable item gets a response. Only inline review threads can be resolved.

## Inline review thread: reply first

Use the REST `databaseId` of the comment, not the GraphQL thread ID. Keep reply
text in a variable and quote its expansion so shell metacharacters remain
data.

```bash
if [ -z "${COMMENT_DATABASE_ID:-}" ] || [ -z "${REPLY_BODY:-}" ]; then
  echo "Error: COMMENT_DATABASE_ID and REPLY_BODY are required" >&2
  exit 1
fi

gh api \
  "repos/$PR_REPO/pulls/$PR_NUMBER/comments/$COMMENT_DATABASE_ID/replies" \
  -f body="$REPLY_BODY" || {
    echo "Error: failed to reply to inline comment $COMMENT_DATABASE_ID" >&2
    exit 1
  }
```

Do not embed review text in an evaluated command string. Quoted
`body="$REPLY_BODY"` safely preserves backticks, dollar signs, exclamation
marks, spaces, and newlines.

## Inline review thread: resolve second

Use the thread's GraphQL `id`, not a comment `databaseId`. Send GraphQL
variables as JSON so shell expansion cannot alter the query.

```bash
if [ -z "${THREAD_ID:-}" ]; then
  echo "Error: THREAD_ID is required to resolve an inline thread" >&2
  exit 1
fi

GRAPHQL_MUTATION='
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { id isResolved }
  }
}'

IS_RESOLVED=$(jq -n \
  --arg query "$GRAPHQL_MUTATION" \
  --arg thread_id "$THREAD_ID" \
  '{query: $query, variables: {threadId: $thread_id}}' |
  gh api graphql --input - \
    --jq '.data.resolveReviewThread.thread.isResolved') || {
  echo "Error: failed to resolve review thread $THREAD_ID" >&2
  exit 1
}
if [ "$IS_RESOLVED" != "true" ]; then
  echo "Error: review thread $THREAD_ID remains unresolved" >&2
  exit 1
fi
```

Verify the result is `true`. A reply without the resolve mutation leaves the
thread unresolved; a resolve mutation without a reply hides the rationale.

Record every addressed thread ID and, after all mutations, fully paginate the
pull request's `reviewThreads` connection again. Require every recorded ID to
be present with `isResolved=true`; a mutation response alone is not final
verification. A missing ID, unresolved ID, failed reply, failed mutation, or
failed verification read is a blocker. Never report the iteration or task as
complete while any addressed thread lacks this verified state.

## Review bodies and conversation comments

These surfaces have no resolve mutation. Address their authors in one batched
pull-request conversation comment per iteration, then record each handled node
ID in the caller's ledger.

```bash
if [ -z "${REPLY_BODY:-}" ] || [ -z "${HANDLED_LEDGER:-}" ]; then
  echo "Error: REPLY_BODY and HANDLED_LEDGER are required" >&2
  exit 1
fi
gh pr comment "$PR_NUMBER" --repo "$PR_REPO" --body "$REPLY_BODY" || {
  echo "Error: failed to post pull-request conversation reply" >&2
  exit 1
}

for NODE_ID in $HANDLED_NODE_IDS; do
  printf '%s\n' "$NODE_ID" >> "$HANDLED_LEDGER"
done
```

Mention each reviewer by login and make each bullet identify the point being
answered. Informational feedback needs no reply, but its ID must still be
recorded so the next iteration does not present it again.

After replies, fetch all pages again. Continue until there is no unhandled
actionable feedback or until the consuming workflow's documented stuck limit
is reached.
