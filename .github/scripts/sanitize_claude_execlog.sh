#!/usr/bin/env bash
# Scan Claude's execution transcript (the JSONL output written by claude-code's
# --print mode and surfaced via the action's `execution_file` output) BEFORE
# uploading it as a workflow artifact.
#
# Why this exists, when actions.json is already sanitized:
#   actions.json only contains what Claude chose to surface to GitHub. The
#   execution log additionally contains every Read/Grep/Glob/Skill tool I/O,
#   which could include the contents of any file Claude was prompt-injected
#   into reading (.git/config, env-dumped to a tempfile by an injected Skill,
#   secrets pasted from PR diff text, etc.). If we only gate on actions.json,
#   the execution log can leak material that never appeared in the final JSON.
#
# Inputs:
#   $1 -- path to execution log file
# Exit codes:
#   0 -- file is clean
#   1 -- file is missing or empty (treated as failure -- if claude-code-action
#        produced an execution_file output we expect the file to exist)
#   2 -- secret/credential pattern matched

set -euo pipefail

# Same source path strategy as sanitize_claude_actions.sh -- when this script is
# executed from /tmp/trusted/ at runtime, secret_patterns.sh is alongside it.
# shellcheck source=secret_patterns.sh
source "$(dirname "$0")/secret_patterns.sh"

LOG_FILE="${1:-}"
if [[ -z "$LOG_FILE" ]]; then
  echo "::error::sanitize_claude_execlog.sh requires a file path as first arg"
  exit 1
fi
if [[ ! -s "$LOG_FILE" ]]; then
  echo "::error::Execution log missing or empty at: $LOG_FILE"
  exit 1
fi

bytes=$(wc -c < "$LOG_FILE")

# Scan the raw bytes of the file. This is a superset of "JSON-extracted strings"
# (which is what sanitize_claude_actions.sh does) -- a literal secret embedded
# anywhere in the transcript, including in JSON-escaped form, will appear as a
# substring of the raw bytes for almost all alphanumeric token shapes our
# patterns target. Bytes are simpler and faster than re-decoding the JSONL.
hits=$(grep -E "$SUSPICIOUS_PATTERNS" -- "$LOG_FILE" | head -3 || true)
if [[ -n "$hits" ]]; then
  echo "::error::Suspected secret/credential pattern in Claude execution log."
  echo "::error::Refusing to upload execution-log artifact."
  echo "::error::First matches (redacted preview):"
  echo "$hits" | sed -E 's/[A-Za-z0-9_-]/x/g'
  exit 2
fi

name_hits=$(grep -E "$ENV_VAR_NAMES" -- "$LOG_FILE" | head -3 || true)
if [[ -n "$name_hits" ]]; then
  echo "::error::Execution log mentions an LLM-Gateway env var name."
  echo "::error::Refusing to upload execution-log artifact."
  echo "::error::First matches (env-var-name visibility means likely exfil attempt):"
  echo "$name_hits"
  exit 2
fi

echo "Execution-log sanitizer OK: ${bytes} bytes scanned, no secret patterns matched."
