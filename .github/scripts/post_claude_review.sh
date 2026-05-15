#!/usr/bin/env bash
# Post Claude review actions to a GitHub PR.
#
# Runs in the post job, which has GITHUB_TOKEN but NO LLM Gateway secrets in its
# environment. Reads /tmp/pr/actions.json (downloaded as an artifact from the review
# job) and the PR head SHA from /tmp/pr/meta.json, then issues GitHub API calls.
#
# Required env:
#   GH_TOKEN                  -- GitHub token with pull_requests:write permission
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
#   user.login == "github-actions[bot]" AND body contains MARKER
#                                       AND in_reply_to_id == null
#
# Without the marker we would misclassify any unrelated workflow that comments on PRs
# as "github-actions[bot]" as a previous Claude review root, which would either:
#   - make us silently skip an initial review (N>0 -> re-review mode -> nothing to
#     reconcile against -> all fresh findings dropped), or
#   - try to reply/react to comments we did not author.
# It also lets us identify our own resolve/clarify replies and exclude them from
# the human-replies set the update skill walks.
#
# Bump the suffix when changing the body format if you need to invalidate previously
# tagged comments.
MARKER='<!-- claude-pr-review-marker:v1 -->'

with_marker() {
  printf '%s\n\n%s' "$1" "$MARKER"
}

# Tracks whether ANY non-skippable failure happened during posting. We do not exit on
# the first failure -- a single bad inline comment must not lose the other postings or
# the summary. Instead we record it here and exit non-zero at the end so the GitHub
# Actions job fails visibly.
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

  local i path line side body
  for ((i = 0; i < count; i++)); do
    path=$(jq -r ".inline_comments[$i].path" "$ACTIONS_FILE")
    line=$(jq -r ".inline_comments[$i].line" "$ACTIONS_FILE")
    side=$(jq -r ".inline_comments[$i].side" "$ACTIONS_FILE")
    body=$(jq -r ".inline_comments[$i].body" "$ACTIONS_FILE")
    body=$(with_marker "$body")

    # 422 (line not in diff) is non-fatal: log a warning and continue. Every other
    # non-2xx response is a hard failure -- we record it in HAD_FAILURE so the job
    # fails at the end, but we keep posting the remaining comments so a single bad
    # entry doesn't drop the rest.
    if ! gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments" \
         -X POST \
         -f "commit_id=${HEAD_SHA}" \
         -f "path=${path}" \
         -F "line=${line}" \
         -f "side=${side}" \
         -f "body=${body}" \
         >/dev/null 2>/tmp/inline_err; then
      if grep -q '"status":[[:space:]]*"422"' /tmp/inline_err 2>/dev/null \
         || grep -q "HTTP 422" /tmp/inline_err 2>/dev/null; then
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
  # thread.
  local resolved_body
  resolved_body=$(with_marker "Resolved -- addressed in this revision.")

  local i type cid hrid body
  for ((i = 0; i < count; i++)); do
    type=$(jq -r ".thread_updates[$i].type" "$ACTIONS_FILE")
    cid=$(jq -r ".thread_updates[$i].claude_comment_id" "$ACTIONS_FILE")

    case "$type" in
      resolve)
        if ! gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
             -X POST -f "body=${resolved_body}" \
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
        if ! gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
             -X POST -f "body=${resolved_body}" \
             >/dev/null 2>/tmp/thread_err; then
          echo "::error::Failed to post Resolved reply on comment ${cid}"
          cat /tmp/thread_err
          HAD_FAILURE=1
        fi
        ;;
      clarify)
        body=$(jq -r ".thread_updates[$i].body" "$ACTIONS_FILE")
        body=$(with_marker "$body")
        if ! gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
             -X POST -f "body=${body}" \
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

post_summary() {
  local summary
  summary=$(jq -r '.summary' "$ACTIONS_FILE")
  if [[ -z "$summary" || "$summary" == "null" ]]; then
    echo "::warning::No summary in actions.json; skipping top-level comment"
    return 0
  fi

  echo "::group::Posting top-level summary"
  # Use --body-file so multiline content with shell metacharacters is safe.
  # Append the marker so a future re-review can attribute this top-level
  # comment to us as well (not strictly required for the inline-comment
  # detection logic, but kept consistent across every body we post).
  local tmp
  tmp=$(mktemp)
  printf '%s\n\n%s\n' "$summary" "$MARKER" > "$tmp"
  if ! gh pr comment "$PR_NUMBER" --repo "$REPO" --body-file "$tmp" 2>/tmp/summary_err; then
    echo "::error::Failed to post top-level summary"
    cat /tmp/summary_err
    HAD_FAILURE=1
  fi
  rm -f "$tmp"
  echo "::endgroup::"
}

post_inline_comments
post_thread_updates
post_summary

if (( HAD_FAILURE != 0 )); then
  echo "::error::One or more posting actions failed; see logs above."
  exit 1
fi

echo "Done. Posted to PR #${PR_NUMBER} on ${REPO} at ${HEAD_SHA}."
