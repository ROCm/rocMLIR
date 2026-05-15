---
name: update-pr-review
description: Given fresh review findings (from a prior review skill) and a PR number, fetch previous Claude inline comments, cross-reference findings, and update the PR in a thread-aware way. Resolves addressed issues, replies to active threads, and posts only genuinely new findings as new inline comments. Never posts the same issue twice.
argument-hint: [PR-number]
agent: general-purpose
allowed-tools: Bash(gh *), Bash(jq *), Bash(grep *), Bash(head *), Read, Grep, Glob
---

# Update PR Review

You are the second phase of a two-phase PR review pipeline. The first phase (the
`review-rocmlir-pr` skill) already produced a fresh list of findings from the diff. Your
job is to reconcile those findings with the existing inline comment threads on the PR
and take the right action for each one.

**Hard rule**: Never post the same finding as a new inline comment if it was already
flagged in a previous Claude review. Each issue must appear at most once as an inline
comment.

---

## Step 1 -- Parse Inputs

- `$ARGUMENTS` is the PR number.
- The fresh review findings are in the conversation context above (output from the prior
  skill). Each finding has `path`, `line`, `side`, `severity`, and `body`. Extract them.

---

## Step 2 -- Fetch Previous Inline Review Comments

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
gh api --paginate "repos/$REPO/pulls/$ARGUMENTS/comments" \
  | jq -s 'add // []' > /tmp/prev_comments.json
```

From the JSON, build the following picture:

**Claude root comments** -- entries where all of these are true:

- `user.login == "claude[bot]"`
- `in_reply_to_id` is null (i.e. they are thread roots, not replies)

These represent issues flagged in previous Claude reviews.

**Thread replies** -- entries where `in_reply_to_id` is non-null. Group replies under the
root comment they belong to (follow `in_reply_to_id` chains to the root).

**Human replies to Claude** -- within a Claude-rooted thread, any reply where
`user.login != "claude[bot]"`. These are the developer's responses.

For each Claude root comment, record:

- `id` -- used for reactions and replies
- `path` -- file path
- `line` -- line number (may have shifted due to rebasing)
- `body` -- the text of the finding
- `human_replies` -- list of `{id, body}` for any human replies in the thread, ordered by
  `id` ascending (GitHub returns comments in creation order; IDs are monotonically
  increasing, so the last element is always the most recent reply)

---

## Step 3 -- Cross-Reference Findings

For each finding from the fresh review, check if it matches a previous Claude root
comment:

**Matching criteria** (use both together):

1. Same `path` (exact file path match)
2. Same logical issue -- compare the finding `body` to the Claude comment `body`
   semantically. Line numbers may have shifted due to rebasing or new commits, so do not
   require an exact line match.

Then determine which scenario applies and act accordingly:

---

### Scenario A -- Issue is addressed in the new diff AND the developer replied

The flagged code no longer exists or the problem is corrected in the new diff, and there
is at least one human reply in the thread (e.g. "Done", "Fixed", "Implemented").

```bash
# 1. React +1 on the most recent human reply (last element of human_replies -- highest id)
gh api "repos/$REPO/pulls/comments/$HUMAN_REPLY_ID/reactions" \
  -X POST -f content="+1"

# 2. Post a "Resolved" reply on Claude's original comment to close the thread
gh api "repos/$REPO/pulls/$ARGUMENTS/comments/$CLAUDE_COMMENT_ID/replies" \
  -X POST -f body="Resolved -- addressed in this revision."
```

Do NOT create a new inline comment for this finding.

---

### Scenario B -- Issue is addressed in the new diff, no developer reply

The problem is fixed in the new diff, but the developer did not reply to Claude's
comment.

```bash
gh api "repos/$REPO/pulls/$ARGUMENTS/comments/$CLAUDE_COMMENT_ID/replies" \
  -X POST -f body="Resolved -- addressed in this revision."
```

Do NOT create a new inline comment for this finding.

---

### Scenario C -- Issue is NOT fixed AND the developer replied

The flagged code still has the problem in the new diff, and the developer replied in the
thread.

```bash
gh api "repos/$REPO/pulls/$ARGUMENTS/comments/$CLAUDE_COMMENT_ID/replies" \
  -X POST -f body="<concise explanation of why the issue is still present and what needs to change>"
```

Do NOT create a new inline comment for this finding.

---

### Scenario D -- Issue is NOT fixed AND the developer did not reply

The problem remains and no one replied to Claude's original comment.

**Do nothing. Skip silently.** The original inline comment is still visible on the PR.

Do NOT create a new inline comment for this finding.

---

### Scenario E -- Genuinely new finding (no previous Claude comment matches)

This issue was not flagged in any prior review.

Return this finding to the caller so it can be posted as a new inline comment via the
`mcp__github_inline_comment__create_inline_comment` MCP tool. Include the full structured
record (`path`, `line`, `side`, `severity`, `body`).

---

## Step 4 -- Output

Return a structured list of actions taken and genuinely new findings:

```
## Thread Updates
- [path:line] <what was done> (Scenario A/B/C/D)
  ...

## New Findings (post as inline comments)
- path: <path>
  line: <line>
  side: <RIGHT|LEFT>
  severity: <Critical|Major|Minor>
  body: |
    <body>
  ...
```

The caller (workflow) will iterate the `New Findings` records and call
`mcp__github_inline_comment__create_inline_comment` once per record. Findings handled as
thread updates in Step 3 must NOT appear in the `New Findings` list.
