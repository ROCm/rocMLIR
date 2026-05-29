#!/usr/bin/env bash
# Post Claude review actions to a GitHub PR.
#
# Runs in the post job, which has GITHUB_TOKEN but NO LLM Gateway secrets in its
# environment. Reads /tmp/pr/actions.json (downloaded as an artifact from the review
# job) and the PR head SHA from /tmp/pr/meta.json, then issues GitHub API calls.
#
# Required env:
#   GH_TOKEN                  -- rocMLIR-PR-Reviewer App installation token
#                                (minted in the workflow via
#                                actions/create-github-app-token; carries
#                                the App's installation permissions for
#                                pull-request comments + reactions +
#                                replies). Posts authored under this token
#                                appear as `rocmlir-pr-reviewer[bot]`.
#   GITHUB_REPOSITORY         -- owner/repo (auto-set by GitHub Actions)
#   PR_NUMBER                 -- PR number to comment on
#
# Files (relative to $PR_DIR, default /tmp/pr):
#   actions.json              -- output of the review/update skills (validated)
#   meta.json                 -- pre-fetched PR metadata; used for headRefOid
#
# Exit codes:
#   0 -- all actions succeeded (or were skipped with a warning)
#   1 -- missing input or transient API failure that prevented any posting

set -euo pipefail

PR_DIR="${PR_DIR:-/tmp/pr}"
ACTIONS_FILE="${PR_DIR}/actions.json"
META_FILE="${PR_DIR}/meta.json"

# Hidden HTML-comment marker appended to EVERY body this script posts. Re-review
# detection (in the workflow prompt and the update-pr-review skill) filters previous
# inline comments by:
#
#   user.login == "rocmlir-pr-reviewer[bot]" AND body contains MARKER
#                                            AND in_reply_to_id == null
#
# `rocmlir-pr-reviewer[bot]` is the unique bot identity of the rocMLIR-PR-Reviewer
# GitHub App, and is the only identity this pipeline posts under. Author-only
# filtering would technically suffice, but the marker is kept as belt-and-braces
# for two purposes:
#   - It lets the update-pr-review skill distinguish OUR marker-tagged
#     resolve/clarify replies from genuine human replies on the same thread
#     (the dedup gate that prevents repeat resolve/clarify on idempotent reruns
#     keys on this).
#   - If a future workflow in this repo ever installs the same App, the marker
#     gives us a path to filter our own comments from theirs without re-doing
#     the whole pipeline.
#
# Bump the suffix when changing the body format if you need to invalidate previously
# tagged comments.
MARKER='<!-- claude-pr-review-marker:v1 -->'

# Per-action sub-markers, appended on REPLIES only (root inline comments and
# the formal-review body do not get one -- those carry only MARKER). The
# update-pr-review skill's Step 1 uses these sub-markers to disambiguate the
# kind ("resolve" vs "clarify") of OUR own previous replies on a thread when
# computing the dedup gate. Without a sub-marker, the skill has to fall back
# to body-prefix matching against the canned resolved-body string just below,
# which is fragile if the model ever emits a clarify body that legitimately
# starts with that exact prefix.
ACTION_MARKER_RESOLVE='<!-- claude-pr-review-action:resolve -->'
ACTION_MARKER_CLARIFY='<!-- claude-pr-review-action:clarify -->'

with_marker() {
  printf '%s\n\n%s' "$1" "$MARKER"
}

# As `with_marker`, but additionally appends the action-kind sub-marker on
# its own line. Used for thread-update replies (resolve / resolve_with_reaction
# / clarify), which the skill must be able to attribute to a specific action
# kind on a future run.
with_marker_and_action() {
  local body="$1" action_marker="$2"
  printf '%s\n\n%s\n%s' "$body" "$MARKER" "$action_marker"
}

# Tracks whether ANY non-skippable failure happened during posting. We do not exit on
# the first failure -- a single bad inline comment must not lose the other postings or
# the formal review. Instead we record it here and exit non-zero at the end so the
# GitHub Actions job fails visibly.
HAD_FAILURE=0

: "${GH_TOKEN:?GH_TOKEN must be set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"
: "${PR_NUMBER:?PR_NUMBER must be set}"

if [[ ! -s "$ACTIONS_FILE" ]]; then
  echo "::error::$ACTIONS_FILE missing or empty"
  exit 1
fi
if [[ ! -s "$META_FILE" ]]; then
  echo "::error::$META_FILE missing or empty"
  exit 1
fi

HEAD_SHA=$(jq -r '.headRefOid' "$META_FILE")
if [[ -z "$HEAD_SHA" || "$HEAD_SHA" == "null" ]]; then
  echo "::error::Could not extract headRefOid from $META_FILE"
  exit 1
fi

REPO="$GITHUB_REPOSITORY"

post_inline_comments() {
  local count
  count=$(jq '.inline_comments | length' "$ACTIONS_FILE")
  echo "::group::Posting ${count} inline comments"

  local i path line side body suggestion
  for ((i = 0; i < count; i++)); do
    path=$(jq -r ".inline_comments[$i].path" "$ACTIONS_FILE")
    line=$(jq -r ".inline_comments[$i].line" "$ACTIONS_FILE")
    side=$(jq -r ".inline_comments[$i].side" "$ACTIONS_FILE")
    body=$(jq -r ".inline_comments[$i].body" "$ACTIONS_FILE")
    # Optional verbatim single-line replacement. When present, wrap it in a
    # fenced ```suggestion block at the bottom of the comment body. GitHub
    # renders this as a "Commit suggestion" button in the PR UI, letting the
    # developer apply Claude's fix with one click. Schema constraint enforced
    # by the action's --json-schema + the sanitizer: suggestion is a non-empty
    # string when present. The suggestion block is placed BEFORE the hidden
    # marker so re-review detection still finds the marker as the body's
    # last token.
    suggestion=$(jq -r ".inline_comments[$i].suggestion // empty" "$ACTIONS_FILE")
    if [[ -n "$suggestion" ]]; then
      body+=$'\n\n```suggestion\n'"$suggestion"$'\n```'
    fi
    body=$(with_marker "$body")

    # Non-2xx response handling:
    #   - 422 specifically due to "line not part of the diff" is the
    #     ONE soft failure: it just means the model picked a line that
    #     isn't in the PR diff (lag between fresh-finding generation
    #     and posting, or a finding on a context line that GitHub
    #     refuses), which doesn't justify failing the whole posting
    #     job. Log a warning and continue.
    #   - All other 422s (invalid/stale commit_id, malformed path/
    #     side, abuse/spam validation, schema errors) indicate either
    #     a bug in the reviewer output or a corrupted trusted input.
    #     Those MUST set HAD_FAILURE so the job fails visibly --
    #     silently dropping a real finding because GitHub returned
    #     422-for-a-different-reason is exactly the kind of failure
    #     mode this script must NOT have.
    #   - Every other non-2xx response is a hard failure too. We keep
    #     posting the remaining comments so a single bad entry
    #     doesn't drop the rest, then exit non-zero at the end.
    # The "part of the diff" substring is GitHub's stable error string
    # for line-not-in-diff (matches both the historical
    # "pull_request_review_thread.line must be part of the diff" and
    # newer variants like "is not part of the diff"). Using it as a
    # required substring -- in addition to the 422 status check --
    # narrows the suppression to the specific case we mean.
    #
    # Build the JSON request body with jq and feed it to `gh api
    # --input -`, rather than passing each field via -f / -F. Both
    # produce a JSON request body in practice (gh -f also JSON-
    # encodes its values for POST/PATCH/PUT), so the wire-level
    # outcome is the same -- but the explicit jq construction:
    #   - removes any ambiguity about how multi-line bodies and
    #     special characters (backticks, embedded quotes, $, etc.)
    #     are encoded;
    #   - matches GitHub's documented `gh api` recipe for endpoints
    #     that take a JSON body;
    #   - keeps the `line` value typed as a JSON number via
    #     `tonumber`, mirroring the previous -F behavior without a
    #     dedicated typed-field flag.
    if ! jq -n \
              --arg ci "$HEAD_SHA" \
              --arg p  "$path" \
              --arg ln "$line" \
              --arg sd "$side" \
              --arg b  "$body" \
              '{commit_id:$ci, path:$p, line:($ln|tonumber), side:$sd, body:$b}' \
         | gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments" \
              -X POST --input - \
              >/dev/null 2>/tmp/inline_err; then
      is_422=0
      if grep -q '"status":[[:space:]]*"422"' /tmp/inline_err 2>/dev/null \
         || grep -q "HTTP 422" /tmp/inline_err 2>/dev/null; then
        is_422=1
      fi
      is_line_not_in_diff=0
      if grep -qiE 'part of the diff|line.*(not|must) (be|in) (part|the) (of )?(the )?diff' \
            /tmp/inline_err 2>/dev/null; then
        is_line_not_in_diff=1
      fi
      if (( is_422 == 1 && is_line_not_in_diff == 1 )); then
        echo "::warning::Skipping inline comment ${path}:${line} (line not in diff)"
      else
        echo "::error::Failed to post inline comment ${path}:${line}"
        cat /tmp/inline_err
        HAD_FAILURE=1
      fi
    fi
  done

  echo "::endgroup::"
}

post_thread_updates() {
  local count
  count=$(jq '.thread_updates | length' "$ACTIONS_FILE")
  echo "::group::Processing ${count} thread updates"

  # Every reply body is wrapped with the marker so the next re-review can
  # recognise our own replies (vs. genuine human replies) inside a Claude
  # thread. Carries both the master marker (so the skill recognizes it as
  # "ours") and the resolve sub-marker (so a future skill run can tell this
  # is a resolve-kind reply, not a clarify-kind reply, when computing the
  # dedup / regression gate).
  local resolved_body
  resolved_body=$(with_marker_and_action "Resolved -- addressed in this revision." \
                                         "$ACTION_MARKER_RESOLVE")

  local i type cid hrid body
  for ((i = 0; i < count; i++)); do
    type=$(jq -r ".thread_updates[$i].type" "$ACTIONS_FILE")
    cid=$(jq -r ".thread_updates[$i].claude_comment_id" "$ACTIONS_FILE")

    # All three reply paths build the JSON request body via jq and
    # feed it through `gh api --input -`. See the inline-comments
    # POST above for the rationale (parity with -f, but with explicit
    # encoding semantics for multi-line / special-char bodies). The
    # +1 reaction stays on -f because its body is the fixed enum
    # string "+1" with no shell metacharacters.
    case "$type" in
      resolve)
        if ! jq -n --arg b "$resolved_body" '{body:$b}' \
             | gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
                  -X POST --input - \
                  >/dev/null 2>/tmp/thread_err; then
          echo "::error::Failed to post Resolved reply on comment ${cid}"
          cat /tmp/thread_err
          HAD_FAILURE=1
        fi
        ;;
      resolve_with_reaction)
        hrid=$(jq -r ".thread_updates[$i].human_reply_id" "$ACTIONS_FILE")
        # +1 reaction on developer's reply (best-effort: warn but don't fail the run
        # if the reaction can't be set; the Resolved reply is the important part).
        if ! gh api "repos/${REPO}/pulls/comments/${hrid}/reactions" \
             -X POST -f "content=+1" \
             >/dev/null 2>/tmp/thread_err; then
          echo "::warning::Failed +1 reaction on reply ${hrid}"
          cat /tmp/thread_err
        fi
        if ! jq -n --arg b "$resolved_body" '{body:$b}' \
             | gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
                  -X POST --input - \
                  >/dev/null 2>/tmp/thread_err; then
          echo "::error::Failed to post Resolved reply on comment ${cid}"
          cat /tmp/thread_err
          HAD_FAILURE=1
        fi
        ;;
      clarify)
        body=$(jq -r ".thread_updates[$i].body" "$ACTIONS_FILE")
        body=$(with_marker_and_action "$body" "$ACTION_MARKER_CLARIFY")
        if ! jq -n --arg b "$body" '{body:$b}' \
             | gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
                  -X POST --input - \
                  >/dev/null 2>/tmp/thread_err; then
          echo "::error::Failed to post clarification reply on comment ${cid}"
          cat /tmp/thread_err
          HAD_FAILURE=1
        fi
        ;;
      *)
        echo "::error::Unknown thread_update type: ${type}"
        HAD_FAILURE=1
        ;;
    esac
  done

  echo "::endgroup::"
}

post_review() {
  # Submit a formal pull-request review via `gh pr review --comment`.
  # Runs after post_inline_comments / post_thread_updates so the inline
  # comments exist when the review is submitted; they are NOT batched
  # into the review's `comments[]` array (a per-comment 422 would fail
  # the whole batched review atomically).
  #
  # Security policy: every verdict is submitted as `--comment` so the
  # bot's reviews stay advisory -- a bot `--approve` would satisfy
  # branch protection, and a bot `--request-changes` becomes a sticky
  # merge block with no automated recovery (the bug github/gh-aw#27655
  # + PR #27662 had to solve). No runtime opt-in; full threat model in
  # CLAUDE_AUTO_REVIEW.md §13.
  local verdict summary
  verdict=$(jq -r '.verdict // empty' "$ACTIONS_FILE")
  summary=$(jq -r '.summary // empty' "$ACTIONS_FILE")

  # `downgrade_note` is appended to the body header so a maintainer sees
  # both the model's verdict and the fact that it was submitted as
  # COMMENT. The bot always submits as COMMENT; a model verdict of
  # COMMENT therefore needs no annotation (the body header would
  # otherwise read `Verdict: COMMENT -- submitted as COMMENT` which is
  # just noise).
  local downgrade_note=""
  case "$verdict" in
    APPROVE|REQUEST_CHANGES)
      downgrade_note=" -- submitted as COMMENT (automated reviews are advisory)"
      ;;
    COMMENT) ;;
    *)
      # Sanitizer already validates the enum; reaching here means the
      # sanitizer was bypassed or the schema regressed. Do NOT echo
      # the raw value (model-controlled).
      echo "::error::post_review: unknown .verdict in actions.json; sanitizer should have rejected this"
      # `return 0` (not 1) for two reasons: (a) we MUST skip the body-
      # build / submit below -- otherwise the bypassed bad verdict
      # would land verbatim in the rendered review header (the very
      # leak this defensive arm exists to prevent); (b) failures are
      # propagated via HAD_FAILURE so a single bad action doesn't
      # abort the other postings, with the script exiting non-zero at
      # the end.
      HAD_FAILURE=1
      return 0
      ;;
  esac

  # Counts are over the FINAL inline_comments (post-reconciliation), so
  # on a re-review they reflect only genuinely-new (Scenario E) findings.
  # The header label switches to "New findings" on any re-review run,
  # sourced from `is_re_review` on meta.json (set by the pre-fetch step
  # using the exact same filter the prompt's Step 1 uses: prior root
  # comments by BOT_LOGIN carrying the marker). The earlier heuristic of
  # `thread_updates.length > 0` misread the steady-state of an unfixed
  # PR: when every prior thread was dedup-gated AND no genuinely-new
  # findings emerged, both arrays are [] and `Findings: 0` would read as
  # "the PR was always clean" -- exactly the misread this switch exists
  # to prevent. Defaulting to `false` if the field is absent (e.g. a
  # workflow rollback between runs) preserves the prior behaviour.
  local critical major minor total findings_label is_re_review
  critical=$(jq '[.inline_comments[] | select(.severity == "Critical")] | length' "$ACTIONS_FILE")
  major=$(jq    '[.inline_comments[] | select(.severity == "Major")]    | length' "$ACTIONS_FILE")
  minor=$(jq    '[.inline_comments[] | select(.severity == "Minor")]    | length' "$ACTIONS_FILE")
  total=$((critical + major + minor))
  is_re_review=$(jq -r '.is_re_review // false' "$META_FILE")
  findings_label="Findings"
  if [[ "$is_re_review" == "true" ]]; then
    findings_label="New findings"
  fi

  # Body header is computed here (deterministic, no model input). The
  # body that follows is the model's `summary` (## Scope / ## Findings
  # / ## Notes / ## CI status per the skill). Marker for attribution.
  local tmp
  tmp=$(mktemp)
  {
    printf '**Verdict:** %s%s &nbsp;\xc2\xb7&nbsp; **%s:** %d (%d Critical, %d Major, %d Minor)\n' \
        "$verdict" "$downgrade_note" "$findings_label" "$total" "$critical" "$major" "$minor"
    printf '\n---\n\n'
    printf '%s\n' "$summary"
    printf '\n%s\n' "$MARKER"
  } > "$tmp"

  echo "::group::Submitting formal review (verdict=${verdict}; submitted as COMMENT; ${total} new inline comments tallied)"
  if ! gh pr review "$PR_NUMBER" --repo "$REPO" --comment --body-file "$tmp" \
       2>/tmp/review_err; then
    echo "::error::Failed to submit review on PR #${PR_NUMBER}"
    cat /tmp/review_err
    HAD_FAILURE=1
  fi
  rm -f "$tmp"
  echo "::endgroup::"
}

post_inline_comments
post_thread_updates
post_review

if (( HAD_FAILURE != 0 )); then
  echo "::error::One or more posting actions failed; see logs above."
  exit 1
fi

echo "Done. Posted to PR #${PR_NUMBER} on ${REPO} at ${HEAD_SHA}."
