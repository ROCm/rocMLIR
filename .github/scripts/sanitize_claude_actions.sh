#!/usr/bin/env bash
# Validate /tmp/pr/actions.json before posting to GitHub.
#
# This step runs in the SAME job as the Claude review (which has the LLM Gateway
# secrets in its environment). It is the last gate before the artifact is uploaded
# and consumed by the post job. If Claude was prompt-injected into trying to leak
# a secret via the JSON payload, this script must catch it and fail the job.
#
# Note: claude-code-action validates the model's response against the workflow's
# `--json-schema` flag BEFORE this script ever runs. That covers the outer JSON
# shape, required keys, array types, and per-element field types/enums. This
# script only adds checks the schema cannot easily express:
#   - whole-payload size cap
#   - per-array length caps
#   - per-string byte cap, applied INDIVIDUALLY to .summary, each
#     inline_comments[].body, each inline_comments[].suggestion, and each
#     thread_updates[].body. Worst-case POSTED body is ~2x this cap (body
#     + suggestion + framing + marker) which is well inside GitHub's
#     ~65 KiB PR-comment limit -- the per-string cap dominates in practice.
#   - conditional thread_update requirements (resolve_with_reaction needs
#     human_reply_id; clarify needs a non-empty body)
#   - inline_comments[].suggestion single-line contract (no LF/CR, no
#     triple-backtick) -- multi-line suggestions silently mismatch our
#     single-line API call, and embedded ``` would close the wrapping
#     ```suggestion fence early
#   - secret/credential pattern scan over every string (covers .suggestion
#     automatically via `[.. | strings]`)
#   - LLM-Gateway env-var-name scan (same -- covers all strings)
#
# Inputs:
#   $1 -- path to actions.json (default /tmp/pr/actions.json)
# Exit codes:
#   0 -- payload is within limits and contains no suspicious patterns
#   1 -- malformed JSON, missing required keys, or bad conditional fields
#   2 -- suspected secret/credential pattern in payload
#   3 -- payload exceeds size or count limits

set -euo pipefail

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
  echo "::error::actions.json is not valid JSON (claude-code-action --json-schema should have caught this)"
  exit 1
fi

# Count caps. The action's --json-schema validates that these are arrays;
# we only need to bound their length here.
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

# Per-string size cap, applied INDIVIDUALLY to .summary, each
# inline_comments[].body, each inline_comments[].suggestion, and each
# thread_updates[].body. This is NOT a cap on the total posted comment.
#
# Concretely: each inline_comments[i] is posted as
#     body[i] + framing(~50B) + suggestion[i] + marker(~50B)
# so a worst-case posted body is ~2 * MAX_BODY_BYTES + ~100B framing
# (~16 KiB at MAX_BODY_BYTES=8192). GitHub's PR-comment limit is ~65 KiB
# so a 2x cap is comfortable. We keep the cap per-string instead of
# combined because (a) it gives the model a clearer error per-field if
# something is oversized, and (b) the combined cap is dominated by the
# per-string cap in practice -- a 2x bloat in a comment that's already
# under the per-string cap is still well inside GitHub's limit.
#
# Use `utf8bytelength`, NOT `length`. jq's `length` on a string returns
# the count of Unicode code points; the byte limits we care about
# (artifact size, GitHub API request size) are bytes-on-the-wire after
# UTF-8 encoding. A body containing multi-byte code points (CJK,
# emoji, accented Latin in code-comment quotes, etc.) would otherwise
# pass an N-codepoint check but exceed N bytes downstream.
oversized=$(jq -r --argjson cap "$MAX_BODY_BYTES" '
  [.summary,
   (.inline_comments[]?.body),
   (.inline_comments[]?.suggestion // empty),
   (.thread_updates[]?.body // empty)]
  | map(select(. != null) | select((. | utf8bytelength) > $cap))
  | length
' "$ACTIONS_FILE")
if (( oversized > 0 )); then
  echo "::error::${oversized} string field(s) exceed ${MAX_BODY_BYTES} bytes"
  exit 3
fi

# Conditional thread-update requirements that JSON Schema can't express
# concisely:
#   - type == "resolve_with_reaction" requires human_reply_id (integer)
#   - type == "clarify"               requires body (non-empty string)
bad_thread=$(jq -r '
  .thread_updates
  | map(select(
      .type as $t |
      ($t == "resolve_with_reaction" and (.human_reply_id|type)!="number") or
      ($t == "clarify" and ((.body|type)!="string" or (.body|length)==0))
    ))
  | length
' "$ACTIONS_FILE")
if (( bad_thread > 0 )); then
  echo "::error::${bad_thread} thread_updates entries violate the type-specific field requirements"
  exit 1
fi

# Inline-comment suggestion contract: must be a single line, no fence
# breakouts.
#   - LF/CR -> would create a multi-line suggestion. We do not pass
#     start_line/start_side to GitHub, so the API would replace only the
#     single line at `line` with all the multi-line content -- almost
#     certainly not what the developer wants when they click "Commit
#     suggestion".
#   - "```" -> would close the wrapping ```suggestion ... ``` fence early
#     (post_claude_review.sh wraps the suggestion verbatim) and the rest of
#     the suggestion would render as comment text, not as a suggested change.
# JSON Schema's `pattern` already excludes \r and \n on the action's side
# (defense-in-depth), but we re-check here because pattern doesn't catch
# triple-backtick (lookaround is not portably supported in JSON Schema
# validators) and because we want this last gate to be self-contained.
bad_suggestion=$(jq -r '
  [.inline_comments[]?.suggestion // empty]
  | map(select(test("[\r\n]") or contains("```")))
  | length
' "$ACTIONS_FILE")
if (( bad_suggestion > 0 )); then
  echo "::error::${bad_suggestion} inline_comments[].suggestion violate the single-line contract (contain LF/CR or triple-backtick). Fix: keep suggestions to one line, no embedded fences."
  exit 1
fi

# Reject ```suggestion fences in any prose-body field. The structured
# inline_comments[].suggestion field above is the ONLY sanctioned channel
# for commit suggestions: it has the strict single-line/no-fence/high-
# confidence contract enforced just above, and post_claude_review.sh
# wraps it in a controlled ```suggestion fence at a known position.
# A ```suggestion fence inside a free-form body field bypasses every
# part of that contract -- a multi-line, attacker-controlled fence
# would render as a one-click "Commit suggestion" UI in GitHub and let
# a maintainer commit attacker-supplied bytes verbatim.
# We split each body on \n so per-line ^/$ anchors work correctly, and
# we accept any fence with 3+ backticks plus optional surrounding
# whitespace -- GitHub renders ` ```suggestion `, ` ````suggestion `,
# and ` ``` suggestion ` all as the commit-suggestion UI. The literal
# word "suggestion" is the language tag GitHub keys on; other tags
# (```diff, ```python, ...) render as plain code blocks and are fine.
bad_fence=$(jq -r '
  [.summary,
   (.inline_comments[]?.body),
   (.thread_updates[]?.body // empty)]
  | map(select(. != null))
  | [.[] | split("\n")[] | select(test("^[ \t]*`{3,}[ \t]*suggestion[ \t]*$"))]
  | length
' "$ACTIONS_FILE")
if (( bad_fence > 0 )); then
  echo "::error::${bad_fence} body field(s) (.summary / .inline_comments[].body / .thread_updates[].body) contain a \`\`\`suggestion fence. Use the structured inline_comments[].suggestion field instead -- it has the single-line/no-fence contract enforced and is wrapped safely by post_claude_review.sh."
  exit 1
fi

# Anti-spoofing: reject any model-supplied string that contains the literal
# substring "<!-- claude-pr-review-". `post_claude_review.sh` appends the
# master marker (<!-- claude-pr-review-marker:v1 -->) to every body it posts
# AND appends a per-action sub-marker (<!-- claude-pr-review-action:resolve
# --> or :clarify -->) to every thread-update reply. The update-pr-review
# skill uses both markers to attribute prior comments and replies as "ours"
# and to determine each reply's kind (resolve vs clarify) when computing the
# dedup gate.
#
# A prompt-injected model response that included a marker would let an
# attacker:
#   - mint a "fake" Claude root comment that the next reconciliation run
#     thinks WE posted, suppressing a real finding (Scenario E -> dropped
#     because "we already covered this");
#   - mint a "fake" resolve sub-marker on a clarify body so the next run
#     suppresses a regression (Scenario D dedup gate);
#   - mint a "fake" clarify sub-marker on a resolve body to keep the dedup
#     gate inverted forever.
#
# The post script appends its markers AFTER the model's body, so a
# legitimate body never needs to contain the literal "<!-- claude-pr-review-"
# prefix. Reject any string in the model output that does, in any field,
# regardless of position.
bad_marker=$(jq -r '
  [.. | strings]
  | map(select(contains("<!-- claude-pr-review-")))
  | length
' "$ACTIONS_FILE")
if (( bad_marker > 0 )); then
  echo "::error::${bad_marker} string field(s) contain the literal \"<!-- claude-pr-review-\" marker prefix. The pipeline reserves <!-- claude-pr-review-* --> markers for post_claude_review.sh; the model is told to never emit them. A string containing this prefix is either prompt-injected or a programming bug -- refusing to post."
  exit 1
fi

# Secret/credential pattern scan over every string in the document. Patterns are
# defined in secret_patterns.sh.
hits=$(jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" \
        | grep -E "$SUSPICIOUS_PATTERNS" || true)
if [[ -n "$hits" ]]; then
  echo "::error::Suspected secret/credential pattern in actions.json. Refusing to post."
  echo "::error::Matched (redacted) preview:"
  echo "$hits" | head -3 | sed -E 's/[A-Za-z0-9_-]/x/g'
  exit 2
fi

# Also scan for echoes of the env var NAMES (possible exfil attempts even
# without the value). Redact the matched line in the same way we redact
# credential-pattern matches above: the matched line may include the
# env var's VALUE on the same line (e.g. "ANTHROPIC_BASE_URL=https://..."
# or "Ocp-Apim-Subscription-Key: <key>"), and printing it verbatim into
# the GitHub Actions log would defeat the very secret-protection this
# sanitizer exists for. Mask all word characters with x; structural
# punctuation stays so the maintainer can still tell what shape of
# string matched.
name_hits=$(jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" \
              | grep -E "$ENV_VAR_NAMES" || true)
if [[ -n "$name_hits" ]]; then
  echo "::error::actions.json mentions an LLM-Gateway env var name. Refusing to post."
  echo "::error::Matched (redacted) preview:"
  echo "$name_hits" | head -3 | sed -E 's/[A-Za-z0-9_-]/x/g'
  exit 2
fi

# Scan for the LITERAL VALUE of ANTHROPIC_BASE_URL. The env-var-NAME
# scan above catches a model that mentions the variable by name (a
# common exfil-confirmation pattern), but a sufficiently determined
# prompt-injection attempt could exfiltrate the URL by VALUE alone --
# without ever using the env var name. We pass the configured base URL
# in via the env (the sanitizer step in claude_auto_review.yml exports
# ANTHROPIC_BASE_URL). The check is a fixed-string substring match
# rather than a regex (no false positives from the model writing about
# Anthropic in general), and matches are reported with full redaction
# so the URL itself never lands in the log.
if [[ -n "${ANTHROPIC_BASE_URL:-}" ]]; then
  url_hits=$(jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" \
                | grep -F "$ANTHROPIC_BASE_URL" || true)
  if [[ -n "$url_hits" ]]; then
    echo "::error::actions.json contains the ANTHROPIC_BASE_URL value. Refusing to post."
    echo "::error::Matched ${#url_hits} bytes (full content REDACTED to avoid leaking the URL into the log)."
    exit 2
  fi
fi

# =====================================================================
# Generic URL allow-list. The prompt instructs the model to avoid URLs
# entirely except permalinks back to github.com (this PR / repo / docs);
# this block is the enforcement side of that contract. Without it, the
# only url-shaped check above is the ANTHROPIC_BASE_URL fixed-string
# match, which leaves a wide range of attacks open if the model is
# prompt-injected:
#   - phishing / click-tracker URLs posted under the bot's identity (the
#     rocMLIR-PR-Reviewer App is a verified org-installed identity, so
#     reviewers click its links with elevated trust)
#   - URL-shaped exfil where the secret is encoded in path/query (the
#     model has no network egress at runtime, but a maintainer who later
#     clicks the URL becomes the egress channel)
#   - typo-squat / lookalike domains (anthrop1c.com, githab.com, ...)
#     that wouldn't trip any of the above content scans
#   - userinfo-bypass: https://github.com@evil.example/path -- in this
#     URL, "github.com" is the userinfo (RFC 3986 §3.2.1: everything in
#     the authority before the LAST @ is userinfo), and the actual host
#     is "evil.example" -- so a naive regex that stops at @ and treats
#     "github.com" as the host would pass a phishing URL
#   - protocol-relative URLs in Markdown link destinations:
#     [click](//evil.example/path) -- GitHub renders this with href
#     "//evil.example/path", which the browser resolves against the
#     page's protocol (https) -- so the actual destination is
#     https://evil.example/path. A regex that requires the literal
#     "https?://" prefix never sees these.
#   - non-http(s) Markdown link schemes: [click](mailto:...),
#     [click](ftp://...), [click](javascript:...), [click](data:...).
#     Even if GitHub's HTML sanitizer strips javascript:, mailto: and
#     ftp: still produce clickable affordances; the pipeline has no
#     legitimate use for any of them.
#
# The allow-list of HOSTS is intentionally tiny: github.com only (and
# its subdomains: gist.github.com, raw.githubusercontent.com,
# objects.githubusercontent.com, etc.). Code review bodies can always
# reference in-repo files by path/line without a URL; the few cases
# that genuinely need a link (cross-repo PR refs, GitHub-hosted gists)
# are all on github.com. If a future legitimate use case needs another
# host, add it here AND in the prompt's "Hard constraints" block in
# claude_auto_review.yml -- keep the two in sync so the contract the
# model is told about matches what the sanitizer actually enforces.

# Re-used host-allow-list regex. A host is allowed iff it is exactly
# `github.com` OR ends in `.github.com` / `.githubusercontent.com`.
# Used by both Layer 1 (bare URL host check) and Layer 2c (protocol-
# relative Markdown destination host check).
ALLOWED_HOST_RE='^(github\.com|[A-Za-z0-9._-]+\.(github\.com|githubusercontent\.com))$'

# All model-supplied strings, one per line, used by all three layers
# below. Pre-extracting them avoids re-running jq four times.
strings_tmp=$(mktemp)
trap 'rm -f "$strings_tmp"' EXIT
jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" > "$strings_tmp"

# ---------------------------------------------------------------------
# Layer 1: bare http(s) URLs anywhere in any string (prose, code, etc.).
#
# Detection: scheme + authority including userinfo and any URL char up
# to the path/query/fragment delimiter.
#   - `tr 'A-Z' 'a-z'` runs IMMEDIATELY after the case-insensitive grep
#     and BEFORE the sed extractions. Otherwise an uppercase scheme
#     like "HTTP://github.com" would survive grep -oiE, fail the case-
#     sensitive sed scheme-strip, and reach the host check as
#     "http://github.com" -- correctly rejecting a legitimate URL.
#   - The first sed extracts the authority (everything between "//" and
#     the first /, ?, #, or end-of-string). The char class includes `@`
#     so the authority can contain userinfo.
#   - The second sed strips userinfo: per RFC 3986 §3.2.1 the userinfo
#     is everything in the authority before the LAST `@`, so a greedy
#     `.*@` removal yields the bare host[:port]. This is the fix for
#     [click](https://github.com@evil.example/path), where without the
#     strip the host check would see "github.com" instead of the
#     attacker-controlled "evil.example".
#   - The third sed strips a trailing port.
disallowed_hosts=$(grep -oiE 'https?://[A-Za-z0-9._~:@/-]+' "$strings_tmp" \
  | tr 'A-Z' 'a-z' \
  | sed -E 's|^https?://([^/?#]+).*|\1|' \
  | sed -E 's|.*@||' \
  | sed -E 's|:[0-9]+$||' \
  | sort -u \
  | grep -vE "$ALLOWED_HOST_RE" \
  || true)
if [[ -n "$disallowed_hosts" ]]; then
  echo "::error::actions.json contains URLs to disallowed hosts:"
  printf '%s\n' "$disallowed_hosts" | head -10 | sed 's/^/  - /'
  echo "::error::Only github.com (and *.github.com / *.githubusercontent.com) URLs are allowed in review bodies. See the URL allow-list in sanitize_claude_actions.sh and the matching contract in the prompt's Hard constraints block in .github/workflows/claude_auto_review.yml."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 2: Markdown link destinations.
#
# Layer 1 only matches "https?://...". That misses two attack shapes
# inside Markdown link destinations:
#   (a) protocol-relative URLs: [click](//evil.example/x). GitHub
#       renders these against https://, so the actual destination is
#       https://evil.example/x.
#   (b) non-http(s) schemes: [click](mailto:..), [click](ftp://..),
#       [click](javascript:..), [click](data:..), [click](file:..).
#       Even when sanitized by the renderer (e.g. javascript: is
#       stripped), several of these remain clickable; the pipeline
#       has zero legitimate use for any of them.
# Both inline links `](dest)` and reference-style definitions
# `[ref]: dest` are extracted. Title text is stripped by the
# whitespace boundary in the destination-extraction regex.

# Inline destinations: `](<dest>` or `](dest`, optional surrounding
# whitespace and angle brackets.
inline_dests=$(grep -oE '\]\([ \t]*<?[^[:space:]<>)]+' "$strings_tmp" \
  | sed -E 's|^\]\([ \t]*<?||' \
  || true)
# Reference-style destinations: lines like `  [ref]:   <dest>  "title"`.
ref_dests=$(grep -E '^[ \t]*\[[^]]+\]:[ \t]+' "$strings_tmp" \
  | sed -E 's|^[ \t]*\[[^]]+\]:[ \t]+<?([^[:space:]<>]+).*|\1|' \
  || true)
# Combine, drop empties, lowercase for scheme/host comparisons.
md_dests=$( { printf '%s\n' "$inline_dests"; printf '%s\n' "$ref_dests"; } \
  | grep -v '^$' \
  | tr 'A-Z' 'a-z' \
  || true)

# Layer 2a: any non-http(s) scheme in a Markdown destination -> reject.
# Schemes match RFC 3986 §3.1: ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ).
# We accept http: and https: (those are subject to Layer-1's host check)
# and reject everything else. Plain paths and fragment anchors don't
# match the scheme regex, so they pass through.
bad_scheme=$(printf '%s\n' "$md_dests" \
  | grep -E '^[a-z][a-z0-9+.-]*:' \
  | grep -vE '^https?:' \
  | sort -u \
  || true)
if [[ -n "$bad_scheme" ]]; then
  echo "::error::actions.json contains Markdown link destinations with non-http(s) schemes:"
  printf '%s\n' "$bad_scheme" | head -10 | sed 's/^/  - /'
  echo "::error::Only http(s)://github.com URLs (allow-listed by Layer 1), in-repo paths, and fragment anchors are valid link destinations. mailto:, ftp:, javascript:, data:, file:, vbscript: etc. are rejected."
  exit 2
fi

# Layer 2b: protocol-relative destinations (//host/...). Treat as if
# they were https://host/... and run the host through the Layer-1
# allow-list. //github.com/foo passes; //evil.example/x rejects.
bad_proto_rel=$(printf '%s\n' "$md_dests" \
  | grep -oE '^//[^/?#]+' \
  | sed -E 's|^//||' \
  | sed -E 's|.*@||' \
  | sed -E 's|:[0-9]+$||' \
  | sort -u \
  | grep -vE "$ALLOWED_HOST_RE" \
  || true)
if [[ -n "$bad_proto_rel" ]]; then
  echo "::error::actions.json contains protocol-relative Markdown link destinations to disallowed hosts:"
  printf '%s\n' "$bad_proto_rel" | head -10 | sed 's/^/  - /'
  echo "::error::Protocol-relative destinations (//host/...) resolve to the page's protocol; on a github.com page, //evil.example/x becomes https://evil.example/x. Only //github.com/... (and *.github.com / *.githubusercontent.com) is allowed."
  exit 2
fi

echo "Sanitizer OK: ${inline_count} inline comments, ${thread_count} thread updates, ${actual_bytes} bytes."
