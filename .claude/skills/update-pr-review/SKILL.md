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
  - `user.login == "rocmlir-pr-reviewer[bot]"` AND
  - `body` contains the literal substring `<!-- claude-pr-review-marker:v1 -->` AND
  - `in_reply_to_id` is null.

  Each represents an issue flagged in a previous review. `rocmlir-pr-reviewer[bot]`
  is the unique bot identity of the rocMLIR-PR-Reviewer GitHub App, which is the
  only identity this pipeline posts under (every gh API call in
  `.github/scripts/post_claude_review.sh` and the perimeter-banner / fork-notify
  companions authenticates with an installation token minted from that App). The
  marker is appended by `post_claude_review.sh` to every body it posts; filtering
  on it in addition to the author is belt-and-braces -- with a unique App identity
  the author check alone is sufficient against external impersonation, but the
  marker also lets us distinguish OUR resolve/clarify replies from genuine human
  replies on the same thread (see `human_replies` below). Do NOT match on
  `user.login == "claude[bot]"` (this pipeline does not use the Anthropic OIDC
  token exchange) and do NOT match on `user.login == "github-actions[bot]"`
  (that was the bot identity in earlier iterations of this pipeline; it was
  superseded by the App migration).
- **Thread replies** -- entries where `in_reply_to_id` is non-null. Group by their root
  via `in_reply_to_id` chains.
- **Human replies to Claude** -- within a Claude-rooted thread, any reply whose `body`
  does NOT contain the marker `<!-- claude-pr-review-marker:v1 -->`. Our own
  resolve / resolve_with_reaction / clarify replies also carry the marker, so excluding
  by marker-presence (rather than by author) is correct even when a human comments via
  another Actions workflow that posts under the same bot identity (rare but possible
  if a future workflow installs the same App).
- **Claude (marker-tagged) replies** -- within a Claude-rooted thread, replies that
  DO contain the marker. These are our own previous resolve / resolve_with_reaction /
  clarify replies. Tracking them is REQUIRED for the dedup gate in Step 2 -- without
  it, every re-review run on a still-fixed (or still-clarified) thread re-emits the
  same resolve/clarify action and the bot posts duplicate replies on each rerun.

  Each marker-tagged reply has a **kind** derived from its body:
  - `kind = "resolve"` iff the body starts with the literal string
    `Resolved -- addressed in this revision.` (this is the canned body
    `post_claude_review.sh` posts for `resolve` and `resolve_with_reaction`
    actions; bump in lockstep if that string ever changes).
  - `kind = "clarify"` otherwise.

  Tracking `kind` is REQUIRED to distinguish "we resolved this thread last run,
  the issue should stay quiet" from "we resolved this thread last run, but the
  issue regressed on this revision". Treating both as a generic "claude already
  replied here" hides regressions on previously-resolved threads.

For each Claude root comment, record:
- `id`
- `path`
- `line`
- `body`
- `human_replies` -- list of `{id, body}` ordered by `id` ascending; the last element is
  the most recent human reply (GitHub IDs are monotonically increasing).
- `claude_replies` -- list of `{id, body, kind}` ordered by `id` ascending; the last
  element is the most recent of OUR own marker-tagged replies on this thread.
- `latest_claude_reply_kind` -- `"resolve"`, `"clarify"`, or `null` if `claude_replies`
  is empty. Set from the highest-id entry in `claude_replies`.
- `latest_activity_is_claude` -- boolean. `true` iff the highest-id reply across
  `human_replies + claude_replies` is in `claude_replies`. Equivalently: is the most
  recent activity on this thread a marker-tagged reply we posted? (Used together with
  `latest_claude_reply_kind` to gate suppression: see Step 2.)

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
      → **Scenario C** -- still present, developer replied.
        **Dedup gate:** if `prev_comment.latest_activity_is_claude` is `true` AND
        `prev_comment.latest_claude_reply_kind == "clarify"`, our previous run already
        posted a clarify reply AFTER the most recent human reply and the situation has
        not changed since. **Skip silently** to avoid posting a duplicate clarify on
        every rerun.
        Otherwise (most recent activity on the thread is a human reply we have not
        responded to yet, OR our latest reply on this thread was a `resolve` and the
        thread has since reopened with a developer reply) emit a `clarify` action that
        replies to the original Claude comment with a concise explanation of why the
        issue is still present.
    Else:
      → **Scenario D** -- still present, no developer reply.
        **Regression sub-case:** if `prev_comment.latest_claude_reply_kind == "resolve"`,
        the thread was previously marked Resolved by us but the same finding has come
        back on this revision. This is a **regression** -- the canonical "resolved"
        reply on the thread is now misleading. Emit a `clarify` action whose body
        explicitly notes the regression, e.g. `"Regression: the issue this thread was
        marked Resolved for is present again at line N -- {one-line restatement of the
        finding}."`.
        Otherwise (`latest_claude_reply_kind` is `"clarify"` or `null`): **skip silently.**
        The original Claude inline comment is still visible on the PR and nothing about
        the situation has changed since the last run.

  Else (no matching fresh finding):
    The previous issue is **fixed** (or no longer in the diff).

    **Dedup gate** (applies to both A and B): if
    `prev_comment.latest_claude_reply_kind == "resolve"` AND
    `prev_comment.latest_activity_is_claude` is `true`, we already posted a Resolved
    reply on a previous run and no new human activity has occurred since. **Skip
    silently** to avoid posting a duplicate "Resolved" reply on every rerun.

    NOTE: if our latest reply was a `clarify` and the issue is now fixed, the dedup
    gate intentionally does NOT fire -- the clarify said "still present", the new
    state is "fixed", so we DO want to emit a fresh resolve.

    Otherwise:
      If `prev_comment.human_replies` is non-empty:
        → **Scenario A** -- fixed, developer replied. Emit a `resolve_with_reaction`
          action that reacts +1 on the most recent human reply (last element of
          `human_replies`) AND posts a "Resolved" reply on the original Claude comment.
      Else:
        → **Scenario B** -- fixed, no developer reply. Emit a `resolve` action that
          posts a "Resolved" reply on the original Claude comment.

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
      "body": "...",
      "suggestion": "...optional verbatim single-line replacement; see review-rocmlir-pr SKILL.md for the contract..."
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
- Pass through the optional `suggestion` field unchanged from the fresh findings;
  do NOT add or modify it. The contract for when `suggestion` is appropriate is
  defined by `review-rocmlir-pr` (single-line, verbatim, self-contained, high
  confidence); this skill is reconciliation-only and never makes that call.
- `thread_updates` MUST contain one entry per previous Claude comment that fell into
  Scenario A, B, or C **and was not skipped by the dedup gate** in Step 2. Scenario D
  emits no entry. A previous comment that the dedup gate suppressed (because we already
  posted the same resolve/clarify on an earlier run with no human activity since)
  also emits no entry. The "no duplicate replies" guarantee depends on this:
  emitting a `resolve`/`resolve_with_reaction`/`clarify` here always causes
  `post_claude_review.sh` to post a NEW reply, so the gate is the only thing
  preventing repeat replies on idempotent reruns.
- Every `body` field is plain markdown text. Do not include backticks-fenced code in a
  way that contains the literal characters `${` or `<%` (those are template delimiters
  in some downstream tools and may be misinterpreted).
- Do NOT include any field beyond those shown. Extra fields are dropped by the post
  script.
- Do NOT echo any environment variable, secret, header, or URL into any field. The
  post-job sanitizer rejects strings matching common secret patterns and fails the
  workflow.
