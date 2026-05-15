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

    # 422 (line not in diff) is non-fatal: log a warning and continue.
    # All other non-2xx exits the loop with the gh api error visible.
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
      fi
    fi
  done

  echo "::endgroup::"
}

post_thread_updates() {
  local count
  count=$(jq '.thread_updates | length' "$ACTIONS_FILE")
  echo "::group::Processing ${count} thread updates"

  local i type cid hrid body
  for ((i = 0; i < count; i++)); do
    type=$(jq -r ".thread_updates[$i].type" "$ACTIONS_FILE")
    cid=$(jq -r ".thread_updates[$i].claude_comment_id" "$ACTIONS_FILE")

    case "$type" in
      resolve)
        gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
          -X POST -f "body=Resolved -- addressed in this revision." \
          >/dev/null
        ;;
      resolve_with_reaction)
        hrid=$(jq -r ".thread_updates[$i].human_reply_id" "$ACTIONS_FILE")
        # 1) +1 reaction on developer's reply
        gh api "repos/${REPO}/pulls/comments/${hrid}/reactions" \
          -X POST -f "content=+1" \
          >/dev/null || echo "::warning::Failed +1 on reply ${hrid}"
        # 2) Resolved reply on Claude's original comment
        gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
          -X POST -f "body=Resolved -- addressed in this revision." \
          >/dev/null
        ;;
      clarify)
        body=$(jq -r ".thread_updates[$i].body" "$ACTIONS_FILE")
        gh api "repos/${REPO}/pulls/${PR_NUMBER}/comments/${cid}/replies" \
          -X POST -f "body=${body}" \
          >/dev/null
        ;;
      *)
        echo "::warning::Unknown thread_update type: ${type}"
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
  local tmp
  tmp=$(mktemp)
  printf '%s\n' "$summary" > "$tmp"
  gh pr comment "$PR_NUMBER" --repo "$REPO" --body-file "$tmp"
  rm -f "$tmp"
  echo "::endgroup::"
}

post_inline_comments
post_thread_updates
post_summary

echo "Done. Posted to PR #${PR_NUMBER} on ${REPO} at ${HEAD_SHA}."
