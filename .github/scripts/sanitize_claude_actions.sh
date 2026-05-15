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

# Secret/credential pattern scan over every string in the document.
# These patterns are deliberately generous: a few false positives that block a build
# are far better than leaking a key. Patterns are kept as a single-line ERE alternation
# (not PCRE extended mode) to maximise grep portability.
#
# Patterns covered:
#   sk-ant-api##-...          real Anthropic API key
#   sk-[30+ chars]            generic OpenAI-style sk- key (excludes the dummy literal
#                             "sk-ant-dummy-gateway-key" which is too short to match)
#   Bearer <token>            HTTP Bearer auth tokens
#   Ocp-Apim-Subscription-Key: <value>
#   ghp_/gho_/ghu_/ghs_/ghr_/github_pat_   GitHub PATs and the GHA installation
#                             token (ghs_) that actions/checkout writes to .git/config
#                             when persist-credentials is on. We disable
#                             persist-credentials in the workflow, but this is the
#                             belt-and-braces backstop.
#   eC1hY2Nlc3MtdG9rZW46Z2hz  base64 of "x-access-token:ghs" -- the prefix of
#                             actions/checkout's basic-auth header. Catches an attacker
#                             that leaks the entire AUTHORIZATION value.
#   xox[baprs]-...            Slack tokens
#   AKIA<16>                  AWS access key id
#   -----BEGIN ... PRIVATE KEY-----
suspicious_patterns='sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{30,}|[Bb]earer[[:space:]]+[A-Za-z0-9_.-]{20,}|[Oo]cp-[Aa]pim-[Ss]ubscription-[Kk]ey[[:space:]]*:[[:space:]]*[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|eC1hY2Nlc3MtdG9rZW46Z2hz|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN[[:space:]]+([A-Z]+[[:space:]]+)?PRIVATE[[:space:]]+KEY-----'

hits=$(jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" \
        | grep -E "$suspicious_patterns" || true)
if [[ -n "$hits" ]]; then
  echo "::error::Suspected secret/credential pattern in actions.json. Refusing to post."
  echo "::error::Matched (redacted) preview:"
  echo "$hits" | head -3 | sed -E 's/[A-Za-z0-9_-]/x/g'
  exit 2
fi

# Also scan for echoes of the env var NAMES (possible exfil attempts even without the value)
env_var_names='ANTHROPIC_BASE_URL|LLM_GATEWAY_KEY|USER_NTID|ANTHROPIC_CUSTOM_HEADERS|ANTHROPIC_API_KEY'
name_hits=$(jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" \
              | grep -E "$env_var_names" || true)
if [[ -n "$name_hits" ]]; then
  echo "::error::actions.json mentions an LLM-Gateway env var name. Refusing to post."
  echo "$name_hits" | head -3
  exit 2
fi

echo "Sanitizer OK: ${inline_count} inline comments, ${thread_count} thread updates, ${actual_bytes} bytes."
