#!/usr/bin/env bash
# Validate /tmp/pr/actions.json before posting to GitHub.
#
# This step runs in the SAME job as the Claude review (which has the LLM Gateway
# secrets in its environment). It is the last gate before the artifact is uploaded
# and consumed by the post job. If Claude was prompt-injected into trying to leak a
# secret via the JSON payload, this script must catch it and fail the job.
#
# Inputs:
#   $1 -- path to actions.json (default /tmp/pr/actions.json)
# Exit codes:
#   0 -- JSON is valid and contains no suspicious patterns
#   1 -- malformed JSON or missing required keys
#   2 -- suspected secret/credential pattern in payload
#   3 -- payload exceeds size limits

set -euo pipefail

# Pattern definitions live next to this script so the execution-log sanitizer
# can share them. When this script is run from /tmp/trusted/ at runtime, the
# workflow snapshots secret_patterns.sh into the same directory (see the
# "Snapshot trusted sanitizers" step in claude_auto_review.yml).
# shellcheck source=secret_patterns.sh
source "$(dirname "$0")/secret_patterns.sh"

ACTIONS_FILE="${1:-/tmp/pr/actions.json}"
MAX_BYTES=${MAX_BYTES:-262144}             # 256 KiB cap on the whole payload
MAX_INLINE_COMMENTS=${MAX_INLINE_COMMENTS:-50}
MAX_THREAD_UPDATES=${MAX_THREAD_UPDATES:-100}
MAX_BODY_BYTES=${MAX_BODY_BYTES:-8192}     # 8 KiB cap per comment body

if [[ ! -s "$ACTIONS_FILE" ]]; then
  echo "::error::actions.json is missing or empty at $ACTIONS_FILE"
  exit 1
fi

actual_bytes=$(wc -c < "$ACTIONS_FILE")
if (( actual_bytes > MAX_BYTES )); then
  echo "::error::actions.json is ${actual_bytes} bytes, exceeds cap ${MAX_BYTES}"
  exit 3
fi

if ! jq -e . "$ACTIONS_FILE" >/dev/null; then
  echo "::error::actions.json is not valid JSON"
  exit 1
fi

# Required top-level keys
for key in summary inline_comments thread_updates; do
  if ! jq -e "has(\"$key\")" "$ACTIONS_FILE" >/dev/null; then
    echo "::error::actions.json missing required key: $key"
    exit 1
  fi
done

# Schema sanity for arrays
if ! jq -e '.inline_comments | type == "array"' "$ACTIONS_FILE" >/dev/null; then
  echo "::error::inline_comments must be an array"
  exit 1
fi
if ! jq -e '.thread_updates | type == "array"' "$ACTIONS_FILE" >/dev/null; then
  echo "::error::thread_updates must be an array"
  exit 1
fi

# Count caps
inline_count=$(jq '.inline_comments | length' "$ACTIONS_FILE")
thread_count=$(jq '.thread_updates | length' "$ACTIONS_FILE")
if (( inline_count > MAX_INLINE_COMMENTS )); then
  echo "::error::inline_comments has ${inline_count} entries, exceeds cap ${MAX_INLINE_COMMENTS}"
  exit 3
fi
if (( thread_count > MAX_THREAD_UPDATES )); then
  echo "::error::thread_updates has ${thread_count} entries, exceeds cap ${MAX_THREAD_UPDATES}"
  exit 3
fi

# Per-body size cap
oversized=$(jq -r --argjson cap "$MAX_BODY_BYTES" '
  [.summary, (.inline_comments[]?.body), (.thread_updates[]?.body // empty)]
  | map(select(. != null) | select((. | length) > $cap))
  | length
' "$ACTIONS_FILE")
if (( oversized > 0 )); then
  echo "::error::${oversized} body field(s) exceed ${MAX_BODY_BYTES} bytes"
  exit 3
fi

# Required fields per inline comment
bad_inline=$(jq -r '
  .inline_comments
  | map(select(
      (.path|type)!="string" or (.path|length)==0 or
      (.line|type)!="number" or
      (.side|type)!="string" or (.side != "RIGHT" and .side != "LEFT") or
      (.body|type)!="string" or (.body|length)==0
    ))
  | length
' "$ACTIONS_FILE")
if (( bad_inline > 0 )); then
  echo "::error::${bad_inline} inline_comments entries are missing required fields or have wrong types"
  exit 1
fi

# Required fields per thread update
bad_thread=$(jq -r '
  .thread_updates
  | map(select(
      .type as $t |
      ($t != "resolve" and $t != "resolve_with_reaction" and $t != "clarify") or
      (.claude_comment_id|type)!="number" or
      ($t == "resolve_with_reaction" and (.human_reply_id|type)!="number") or
      ($t == "clarify" and ((.body|type)!="string" or (.body|length)==0))
    ))
  | length
' "$ACTIONS_FILE")
if (( bad_thread > 0 )); then
  echo "::error::${bad_thread} thread_updates entries are missing required fields or have wrong types"
  exit 1
fi

# Secret/credential pattern scan over every string in the document. Patterns are
# defined in secret_patterns.sh (sourced above) and shared with the execution-log
# sanitizer.
hits=$(jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" \
        | grep -E "$SUSPICIOUS_PATTERNS" || true)
if [[ -n "$hits" ]]; then
  echo "::error::Suspected secret/credential pattern in actions.json. Refusing to post."
  echo "::error::Matched (redacted) preview:"
  echo "$hits" | head -3 | sed -E 's/[A-Za-z0-9_-]/x/g'
  exit 2
fi

# Also scan for echoes of the env var NAMES (possible exfil attempts even without the value)
name_hits=$(jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" \
              | grep -E "$ENV_VAR_NAMES" || true)
if [[ -n "$name_hits" ]]; then
  echo "::error::actions.json mentions an LLM-Gateway env var name. Refusing to post."
  echo "$name_hits" | head -3
  exit 2
fi

echo "Sanitizer OK: ${inline_count} inline comments, ${thread_count} thread updates, ${actual_bytes} bytes."
