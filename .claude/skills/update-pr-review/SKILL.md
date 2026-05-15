---
name: update-pr-review
description: Reconcile fresh review findings against existing inline comment threads on a PR. Iterates over previous Claude root comments first so that fixed issues are correctly resolved, then handles still-present issues, then identifies genuinely new findings. Never posts the same issue twice.
argument-hint: [PR-number]
agent: general-purpose
allowed-tools: Read, Grep, Glob
---

<!--
NOTE on `allowed-tools`:

In workflow context, the .github/workflows/claude_auto_review.yml step constrains the
session to `--allowedTools "Skill,Read,Grep,Glob"` and uses `--json-schema` to capture
the model's final response as `structured_output`. This skill never posts to GitHub
directly; it emits structured action records as the model's final response, which the
workflow materializes to /tmp/pr/actions.json and the post job (a separate job, no LLM
Gateway secrets in env) consumes via raw gh api.

For interactive Stage-B local dry-runs, invoke the standalone Claude Code CLI with the
broader tool set, e.g.:

    claude --allowedTools "Skill,Read,Grep,Glob,Bash(gh *),Bash(jq *)" \
           --skill update-pr-review <PR-number>
-->


# Update PR Review

You are the second phase of a two-phase PR review pipeline. The first phase
(`review-rocmlir-pr`) already produced a fresh list of findings from the diff. Your job
is to reconcile those findings with the existing inline comment threads on the PR and
emit the right action for each one.

**Hard rule**: Never emit the same finding as a new inline comment if it was already
flagged in a previous Claude review. Each issue must appear at most once as an inline
comment in the PR's lifetime.

---

## Step 1 -- Inputs

- `$ARGUMENTS` is the PR number.
- The fresh review findings are in the conversation context above (output from the prior
  skill). Each finding has `path`, `line`, `side`, `severity`, and `body`.
- Previous Claude inline comments are pre-loaded at `/tmp/pr/prev_comments.json`. If that
  file is missing (interactive use outside the workflow), fetch it:

  ```bash
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
  mkdir -p /tmp/pr
  gh api --paginate "repos/$REPO/pulls/$ARGUMENTS/comments" \
    | jq -s 'add // []' > /tmp/pr/prev_comments.json
  ```

From the JSON, build:

- **Claude root comments** -- entries where:
  - `user.login == "github-actions[bot]"` AND
  - `body` contains the literal substring `<!-- claude-pr-review-marker:v1 -->` AND
  - `in_reply_to_id` is null.

  Each represents an issue flagged in a previous review. The marker is appended by
  `.github/scripts/post_claude_review.sh` to every body it posts; filtering on it is
  REQUIRED because `github-actions[bot]` is a bot identity shared with any other
  workflow that posts on PRs in this repo. Do NOT match on `user.login == "claude[bot]"`
  -- this pipeline does not authenticate via the Anthropic OIDC token exchange, so no
  comment will ever have that author.
- **Thread replies** -- entries where `in_reply_to_id` is non-null. Group by their root
  via `in_reply_to_id` chains.
- **Human replies to Claude** -- within a Claude-rooted thread, any reply whose `body`
  does NOT contain the marker `<!-- claude-pr-review-marker:v1 -->`. Our own
  resolve / resolve_with_reaction / clarify replies also carry the marker, so excluding
  by marker-presence (rather than by author) is correct even when a human comments via
  another Actions workflow that runs as `github-actions[bot]`.

For each Claude root comment, record:
- `id`
- `path`
- `line`
- `body`
- `human_replies` -- list of `{id, body}` ordered by `id` ascending; the last element is
  the most recent human reply (GitHub IDs are monotonically increasing).

---

## Step 2 -- Iterate previous Claude root comments first (fixes the "fixed issue is silently lost" bug)

The fresh review only contains issues that **still exist** in the diff. So if a previous
issue was fixed, it WILL NOT appear in the fresh findings. To detect that, we must walk
the previous comments first and ask "does any fresh finding match this previous one?"
-- not the other way around.

Initialize an empty set `handled_fresh = {}` to track which fresh findings have been
matched to a previous comment.

For each `prev_comment` in the previous Claude root comments:

  Find a fresh finding `f` that matches `prev_comment`:
  - **Same `path`** (exact file path match), AND
  - **Same logical issue** -- semantically compare `f.body` to `prev_comment.body`.
    Line numbers may have shifted due to rebasing or new commits, so do NOT require an
    exact line match. The criterion is "is this the same complaint?".

  If a matching fresh finding `f` is found:
    Mark `handled_fresh.add(f)`. The issue is **still present**.

    If `prev_comment.human_replies` is non-empty:
      → **Scenario C** -- still present, developer replied. Emit a `clarify` action that
        replies to the original Claude comment with a concise explanation of why the
        issue is still present.
    Else:
      → **Scenario D** -- still present, no developer reply. **Skip silently.** The
        original Claude inline comment is still visible on the PR; nothing to add.

  Else (no matching fresh finding):
    The previous issue is **fixed** (or no longer in the diff).

    If `prev_comment.human_replies` is non-empty:
      → **Scenario A** -- fixed, developer replied. Emit a `resolve_with_reaction`
        action that reacts +1 on the most recent human reply (last element of
        `human_replies`) AND posts a "Resolved" reply on the original Claude comment.
    Else:
      → **Scenario B** -- fixed, no developer reply. Emit a `resolve` action that posts
        a "Resolved" reply on the original Claude comment.

After processing all previous comments:

For each fresh finding `f` NOT in `handled_fresh`:
  → **Scenario E** -- genuinely new. Emit it as an `inline_comments` entry so the post
    job can post it as a new inline comment on the PR diff.

---

## Step 3 -- Output schema

Return a single JSON object with two arrays AS YOUR FINAL RESPONSE. The workflow uses
claude-code-action's `--json-schema` flag to validate the response and capture it as
`structured_output`; do not write to a file. Use this exact schema -- the post script
depends on it:

```json
{
  "summary": "<3-5 line top-level summary written by the review skill>",
  "inline_comments": [
    {
      "path": "mlir/lib/Dialect/Rock/Foo.cpp",
      "line": 142,
      "side": "RIGHT",
      "severity": "Major",
      "body": "..."
    }
  ],
  "thread_updates": [
    {
      "type": "resolve_with_reaction",
      "claude_comment_id": 1234567890,
      "human_reply_id": 1234567899
    },
    {
      "type": "resolve",
      "claude_comment_id": 1234567891
    },
    {
      "type": "clarify",
      "claude_comment_id": 1234567892,
      "body": "Still present at line 87 -- the change moved the call but did not fix the underlying type mismatch."
    }
  ]
}
```

Rules:
- `inline_comments` MUST contain only Scenario E findings. Findings handled in Step 2
  (Scenarios A/B/C; D emits nothing) MUST NOT appear here.
- `thread_updates` MUST contain one entry per previous Claude comment that fell into
  Scenario A, B, or C. Scenario D emits no entry.
- Every `body` field is plain markdown text. Do not include backticks-fenced code in a
  way that contains the literal characters `${` or `<%` (those are template delimiters
  in some downstream tools and may be misinterpreted).
- Do NOT include any field beyond those shown. Extra fields are dropped by the post
  script.
- Do NOT echo any environment variable, secret, header, or URL into any field. The
  post-job sanitizer rejects strings matching common secret patterns and fails the
  workflow.
