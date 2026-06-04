#!/usr/bin/env bash
# Validate /tmp/pr/actions.json before posting to GitHub.
#
# Runs in the SAME job as the Claude review (which holds the LLM Gateway
# secrets) -- the last gate before the artifact is uploaded and consumed by
# the post job. If Claude was prompt-injected into leaking a secret via the
# JSON payload, this script must catch it and fail the job.
#
# claude-code-action's --json-schema validates the outer JSON shape, required
# keys, array types, and per-element field types/enums BEFORE this script
# runs. This script adds the checks the schema cannot easily express, in this
# order (see CLAUDE_AUTO_REVIEW.md §10 for the full design + bypass catalog):
#   - whole-payload size cap (MAX_BYTES); per-array length caps
#     (MAX_INLINE_COMMENTS, MAX_THREAD_UPDATES); per-string byte cap
#     (MAX_BODY_BYTES) applied individually to .summary, each
#     inline_comments[].body/.suggestion, each thread_updates[].body.
#   - schema-defense-in-depth re-check on .verdict / .summary / each
#     body+suggestion string type (a non-string would partially leak via
#     jq's truncated error before later scans see it).
#   - conditional thread_update field requirements (resolve_with_reaction
#     needs human_reply_id; clarify needs a non-empty body).
#   - thread_updates[] reference cross-check vs PREV_COMMENTS_FILE: every
#     claude_comment_id must be a Claude *root* (BOT_LOGIN + CLAUDE_MARKER +
#     in_reply_to_id == null); every non-null human_reply_id must be a human
#     reply in the SAME thread (else a prompt-injected ID could drop +1 on
#     any review comment in the repo or spoof a "resolved" reply under a
#     human's thread under the bot's identity).
#   - inline_comments[].suggestion single-line contract (no LF/CR, no
#     triple-backtick fence breakout); body-field ```suggestion fence
#     rejection (the only sanctioned commit-suggestion channel is the
#     structured field above).
#   - marker anti-spoof: reject any "<!-- claude-pr-review-" substring (the
#     pipeline reserves that marker namespace for attribution + resolve/
#     clarify dedup; an injected marker spoofs prior bot comments).
#   - secret/credential pattern scan over every string (raw + entity-decoded
#     + TAB/LF/CR-stripped views); LLM-Gateway env-var-NAME scan;
#     LLM-Gateway env-var-VALUE scan (fixed-string match against the runtime
#     values of ANTHROPIC_BASE_URL, LLM_GATEWAY_KEY, USER_NTID -- catches
#     by-VALUE exfil that never names the env var).
#   - URL allow-list, 6 progressive layers. Allowed hosts: bare `github.com`
#     / `llvm.org` plus any subdomain of `github.com` / `githubusercontent.com`
#     / `llvm.org`. Bypass classes (bare URL, Markdown destination, HTML
#     href/src, bracketed-IP, percent-encoded authority, LF/CR/TAB-split
#     host) are catalogued per-layer in §10.
#
# All rejection diagnostics redact the matched bytes: matched content is
# model-controlled, and the public Actions log of a public-repo PR would
# itself become a leak channel if the trigger were printed verbatim.
#
# Inputs:
#   $1 -- path to actions.json (default /tmp/pr/actions.json)
# Exit codes:
#   0 -- payload is within limits and contains no suspicious patterns
#   1 -- malformed JSON, missing required keys, or bad conditional fields
#   2 -- suspected secret/credential pattern in payload
#   3 -- payload exceeds size or count limits

set -euo pipefail

# shellcheck source-path=SCRIPTDIR
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

# Schema-defense-in-depth: re-check the string-typed and enum-valued fields
# the action's --json-schema is supposed to have validated already. If a
# schema regression slipped a non-string through, later operations
# (`utf8bytelength`, `test`, `contains`, `split`) would error with jq's
# truncated value preview (~10 chars) landing in the public Actions log
# before any secret-pattern layer ran. Do NOT echo raw .verdict / .summary
# / field values for the same reason -- all model-controlled.
verdict=$(jq -r '.verdict // empty' "$ACTIONS_FILE")
case "$verdict" in
  APPROVE|REQUEST_CHANGES|COMMENT) ;;
  *)
    echo "::error::actions.json has missing or invalid .verdict; must be APPROVE, REQUEST_CHANGES, or COMMENT (raw value not printed -- model-controlled content)"
    exit 1
    ;;
esac

if ! jq -e '.summary | type == "string" and length > 0' "$ACTIONS_FILE" >/dev/null; then
  echo "::error::actions.json .summary must be a non-empty string (--json-schema type:string + minLength:1 should have rejected this)"
  exit 1
fi

if ! jq -e '
  all(
    ( .inline_comments[]?.body,
      .inline_comments[]?.suggestion,
      .thread_updates[]?.body );
    . == null or type == "string")
' "$ACTIONS_FILE" >/dev/null; then
  echo "::error::actions.json has a non-string value in inline_comments[].body, inline_comments[].suggestion, or thread_updates[].body (--json-schema type:string should have rejected this; raw value not printed -- model-controlled content)"
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

# Per-string byte cap on .summary, each inline_comments[].body /
# .suggestion, and each thread_updates[].body. Worst-case posted body is
# ~2x this cap + ~100B framing, well inside GitHub's ~65 KiB comment limit.
# Uses `utf8bytelength` (not `length`, which is codepoints) so multi-byte
# content (CJK, emoji, accented Latin) is bounded in wire bytes.
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

# Two-arm contract on thread_updates[] references vs PREV_COMMENTS_FILE:
#   - claude_comment_id must be a Claude *root* (user.login == $BOT_LOGIN,
#     body contains $CLAUDE_MARKER, in_reply_to_id == null) -- same
#     definition the prefetch step's re-review N-count uses (doc §15
#     sync table). A human reviewer's root would otherwise let the bot
#     post a "resolved" reply under that human's thread.
#   - human_reply_id (when non-null) must be a human reply in THAT SAME
#     thread (user.login != $BOT_LOGIN AND in_reply_to_id ==
#     claude_comment_id). Without the tie-back, a prompt-injected ID
#     could drop a +1 reaction on any review comment in the repo
#     (the reaction endpoint POST /repos/<repo>/pulls/comments/<cid>/
#     reactions has no PR scope).
# BOT_LOGIN / CLAUDE_MARKER are env-input-with-defaults so the test
# harness can swap; production workflow passes the same values it uses
# in prefetch + prompt (doc §15). Fail-closed if PREV_COMMENTS_FILE is
# missing/unparseable while thread_updates is non-empty.
PREV_COMMENTS_FILE="${PREV_COMMENTS_FILE:-/tmp/pr/prev_comments.json}"
BOT_LOGIN="${BOT_LOGIN:-rocmlir-pr-reviewer[bot]}"
CLAUDE_MARKER="${CLAUDE_MARKER:-<!-- claude-pr-review-marker:v1 -->}"
if (( thread_count > 0 )); then
  if [[ ! -s "$PREV_COMMENTS_FILE" ]]; then
    echo "::error::actions.json has ${thread_count} thread_updates entries but PREV_COMMENTS_FILE ($PREV_COMMENTS_FILE) is missing or empty; cannot validate referenced comment IDs"
    exit 1
  fi
  if ! jq -e . "$PREV_COMMENTS_FILE" >/dev/null; then
    echo "::error::PREV_COMMENTS_FILE ($PREV_COMMENTS_FILE) is not valid JSON"
    exit 1
  fi
  # Build a lookup table keyed by comment id (stringified, so jq's
  # `from_entries` can index it) with the two booleans + the
  # in_reply_to_id the predicate needs. Then count thread_updates
  # entries that fail either arm of the contract above. A null
  # claude_comment_id is BAD (we can't validate the human_reply_id
  # thread tie-back without it); a null human_reply_id is fine
  # (resolve_with_reaction is the only type that requires it, and
  # the bad_thread check above already enforces that).
  bad_refs=$(jq -r \
      --slurpfile prev "$PREV_COMMENTS_FILE" \
      --arg bot "$BOT_LOGIN" \
      --arg marker "$CLAUDE_MARKER" '
    ($prev[0] | map({
        key: (.id | tostring),
        value: {
          in_reply_to_id: .in_reply_to_id,
          is_claude_root: (
            .user.login == $bot
            and ((.body // "") | contains($marker))
            and .in_reply_to_id == null
          ),
          is_human: (.user.login != $bot)
        }
      }) | from_entries) as $byid
    | .thread_updates
    | map(select(
        # claude_comment_id arm: missing or not a Claude root.
        ((.claude_comment_id == null)
         or
         (($byid[(.claude_comment_id | tostring)] // null) as $c
          | $c == null or ($c.is_claude_root | not)))
        or
        # human_reply_id arm: if set, must be a human reply IN the
        # same Claude thread.
        (.human_reply_id != null and
         (($byid[(.human_reply_id | tostring)] // null) as $h
          | $h == null
            or ($h.is_human | not)
            or ($h.in_reply_to_id != .claude_comment_id)))
      ))
    | length
  ' "$ACTIONS_FILE")
  if (( bad_refs > 0 )); then
    echo "::error::${bad_refs} thread_updates entries reference comment IDs outside the model's own Claude threads. Contract: claude_comment_id must be a Claude *root* comment (user.login == ${BOT_LOGIN}, body contains the Claude marker, in_reply_to_id == null) present in PREV_COMMENTS_FILE (${PREV_COMMENTS_FILE}); human_reply_id, when set, must be a *human* reply (user.login != ${BOT_LOGIN}) whose in_reply_to_id == claude_comment_id (i.e. in that same thread). Raw IDs not printed -- model-controlled content."
    exit 1
  fi
fi

# Inline-comment suggestion contract: single line, no fence breakouts.
# LF/CR would silently replace only the single line at `line` with all
# the multi-line content (we don't pass start_line/start_side). A "```"
# would close the wrapping ```suggestion fence early. JSON Schema's
# `pattern` already excludes \r and \n; we re-check here because the
# triple-backtick check needs lookarounds and we want this gate self-
# contained.
bad_suggestion=$(jq -r '
  [.inline_comments[]?.suggestion // empty]
  | map(select(test("[\r\n]") or contains("```")))
  | length
' "$ACTIONS_FILE")
if (( bad_suggestion > 0 )); then
  echo "::error::${bad_suggestion} inline_comments[].suggestion violate the single-line contract (contain LF/CR or triple-backtick). Fix: keep suggestions to one line, no embedded fences."
  exit 1
fi

# Reject ```suggestion fences in any prose body field -- the structured
# inline_comments[].suggestion is the only sanctioned commit-suggestion
# channel (contract above + safe fence wrap in post_claude_review.sh).
# An injected fence in a body field would render as a one-click "Commit
# suggestion" UI, bypassing the single-line/no-fence/high-confidence
# contract. Split on \n so ^/$ anchors work per-line; accept 3+ backticks
# with optional whitespace (GitHub renders `` ```suggestion ``,
# `` ````suggestion ``, and `` ``` suggestion `` all as the UI). Other
# language tags (```diff, ```python, ...) render as plain code blocks
# and are fine.
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

# Marker anti-spoof: reject any "<!-- claude-pr-review-" substring. The
# pipeline reserves that prefix for post_claude_review.sh's master and
# per-action sub-markers, which update-pr-review keys on for "ours" vs
# "theirs" attribution and resolve/clarify dedup. A model-emitted marker
# would spoof a prior bot comment (suppressing real findings) or flip the
# resolve/clarify dedup gate. The post script appends markers AFTER the
# model's body, so a legitimate body never contains this prefix.
bad_marker=$(jq -r '
  [.. | strings]
  | map(select(contains("<!-- claude-pr-review-")))
  | length
' "$ACTIONS_FILE")
if (( bad_marker > 0 )); then
  echo "::error::${bad_marker} string field(s) contain the literal \"<!-- claude-pr-review-\" marker prefix. The pipeline reserves <!-- claude-pr-review-* --> markers for post_claude_review.sh; the model is told to never emit them. A string containing this prefix is either prompt-injected or a programming bug -- refusing to post."
  exit 1
fi

# =====================================================================
# Build the THREE string views every downstream scan consumes (full
# rationale: doc §10 "Why three string views"):
#
#   $strings_tmp                  raw bytes as emitted. Used by the
#                                 marker-spoof check above (an entity-
#                                 encoded marker would not match update-
#                                 pr-review's later raw-byte body match
#                                 against the GitHub-stored body, so the
#                                 attribution attack only works on raw).
#   $strings_decoded_tmp          entity-decoded, per-line structure
#                                 preserved. GitHub's renderer entity-
#                                 decodes link destinations, href / src,
#                                 and body text, so an encoded payload
#                                 (e.g. `https&#x3A;//evil/x`) renders
#                                 as the unencoded form. Decoding only
#                                 ADDS matches, never removes them.
#                                 Used by Layer 1 (bare URL), Layer 2
#                                 (Markdown destinations -- needs line
#                                 structure for the `^[ref]:` anchor),
#                                 and the secret / env-name / env-value
#                                 scans.
#   $strings_decoded_oneline_tmp  entity-decoded AND with intra-string
#                                 ASCII tab/LF/CR stripped (WHATWG URL
#                                 §4.4: the URL parser strips exactly
#                                 those bytes; HTML attribute syntax
#                                 permits them, so a `<a href="//evil\n
#                                 host/x">` resolves to evilhost/x in
#                                 the browser, defeating per-line
#                                 anchors). Used by Layer 3 (HTML href/
#                                 src), Layer 4 (bracketed-IP), Layer 5
#                                 (percent-encoded authority); Layer 6
#                                 host-checks destinations sourced from
#                                 it (closes the allowed-host-prefix-of-
#                                 longer-disallowed-host residual gap
#                                 that Layer 1's per-line view cannot
#                                 detect).
#
# Layers 1 and 2 intentionally keep the per-line view: their regexes
# already truncate at whitespace, so an LF-split URL like
# `https://evil\nhost/x` reduces to `evil` (rejected by Layer 1) and a
# stripped view would over-trigger on legitimate prose
# (`See: https://github.com\ncontinue ...`). Bare-prose LF-split URLs
# aren't autolinked by the markdown renderer (it stops at LF), so they
# aren't a reachable phishing vector. Routing summary below.
#
# python3 is preinstalled on every runner; html.unescape handles named,
# decimal, and hex entities per the WHATWG list.
strings_tmp=$(mktemp)
strings_decoded_tmp=$(mktemp)
strings_decoded_oneline_tmp=$(mktemp)
trap 'rm -f "$strings_tmp" "$strings_decoded_tmp" "$strings_decoded_oneline_tmp"' EXIT
jq -r '[.. | strings] | .[]' "$ACTIONS_FILE" > "$strings_tmp"
python3 -c 'import html, sys; sys.stdout.write(html.unescape(sys.stdin.read()))' \
    < "$strings_tmp" > "$strings_decoded_tmp"
# $strings_decoded_oneline_tmp: walk every string in the JSON document,
# entity-decode it, strip ASCII tab / LF / CR (matches WHATWG URL §4.4),
# write each as a single output line. Reads $ACTIONS_FILE directly (not
# $strings_tmp) so we can iterate string-by-string and preserve string
# boundaries -- $strings_tmp's separator newlines and any in-string
# newlines look identical once written, so we cannot distinguish them
# after the fact.
python3 -c '
import html, json, sys
strip_table = {0x09: None, 0x0A: None, 0x0D: None}
def walk(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)
with open(sys.argv[1]) as f:
    data = json.load(f)
out = sys.stdout
for s in walk(data):
    out.write(html.unescape(s).translate(strip_table))
    out.write("\n")
' "$ACTIONS_FILE" > "$strings_decoded_oneline_tmp"

# Secret/credential pattern scan (patterns in secret_patterns.sh). Scans
# all three views so entity-encoded and LF-split secrets are caught. The
# rejection diagnostic redacts alphanumerics and `_-` (preserving
# structural punctuation for triage); structured printing uses `awk
# 'NR<=N'` rather than `head -N` because head closes stdin early and
# triggers SIGPIPE (rc 141) under `set -euo pipefail`, masking the
# intended `exit 2`. Same awk pattern in every "preview" pipeline below.
hits=$(cat "$strings_tmp" "$strings_decoded_tmp" "$strings_decoded_oneline_tmp" \
        | grep -E "$SUSPICIOUS_PATTERNS" || true)
if [[ -n "$hits" ]]; then
  echo "::error::Suspected secret/credential pattern in actions.json. Refusing to post."
  echo "::error::Matched (redacted) preview:"
  echo "$hits" | awk 'NR<=3' | sed -E 's/[A-Za-z0-9_-]/x/g'
  exit 2
fi

# Echo of an env var NAME (possible exfil attempt even without the value).
# Same dual-view scan + same redaction policy.
name_hits=$(cat "$strings_tmp" "$strings_decoded_tmp" "$strings_decoded_oneline_tmp" \
              | grep -E "$ENV_VAR_NAMES" || true)
if [[ -n "$name_hits" ]]; then
  echo "::error::actions.json mentions an LLM-Gateway env var name. Refusing to post."
  echo "::error::Matched (redacted) preview:"
  echo "$name_hits" | awk 'NR<=3' | sed -E 's/[A-Za-z0-9_-]/x/g'
  exit 2
fi

# Env-var VALUE scan: fixed-string match the runtime values of
# ANTHROPIC_BASE_URL (gateway URL), LLM_GATEWAY_KEY (APIM subscription
# key), USER_NTID (org user id). The env-var-NAME scan above catches the
# common name-mentioning exfil; this catches by-VALUE exfil that never
# names the var (a bare APIM key or NTID is otherwise indistinguishable
# from random text until matched as a fixed substring). The sanitize
# step in claude_auto_review.yml exports all three. Diagnostic fully
# redacts (the value would leak via the log otherwise).
for var in ANTHROPIC_BASE_URL LLM_GATEWAY_KEY USER_NTID; do
  val="${!var:-}"
  if [[ -z "$val" ]]; then
    continue
  fi
  if cat "$strings_tmp" "$strings_decoded_tmp" "$strings_decoded_oneline_tmp" \
       | grep -F -q -- "$val"; then
    echo "::error::actions.json contains the ${var} value. Refusing to post."
    echo "::error::Full content REDACTED to avoid leaking the secret/identity into the log."
    exit 2
  fi
done

# =====================================================================
# URL allow-list. Full design (purpose, threat model, bypass classes per
# layer, host-set rationale) in CLAUDE_AUTO_REVIEW.md §10. Host set is
# github.com / llvm.org and their allowed subdomains (plus
# githubusercontent.com subdomains for raw GitHub content); adding a
# host requires updating ALLOWED_HOST_RE here AND the prompt's "Hard
# constraints" block in claude_auto_review.yml (doc §15 sync table).
#
# Routing of the three string views constructed above:
#   secret / env-name / env-value : cat $strings_tmp $strings_decoded_tmp
#                                       $strings_decoded_oneline_tmp
#   Layer 1 (bare URL)            : $strings_decoded_tmp
#   Layer 2 (Markdown dests)      : $strings_decoded_tmp (inline)
#                                 + $strings_decoded_oneline_tmp (inline,
#                                       LF-split)
#                                 + $strings_tmp + per-dest decode/strip
#                                       (ref-style; raw because an entity-
#                                       encoded LF in `[1]: github.com&#10;
#                                       .evil/x` decodes pre-extractor and
#                                       splits the line)
#   Layer 3 (HTML href/src)       : $strings_decoded_oneline_tmp
#   Layer 4 (bracketed-IP)        : $strings_decoded_oneline_tmp
#   Layer 5 (percent auth)        : $strings_decoded_oneline_tmp +
#                                       $md_dests + $attr_dests
#                                       (authority isolation; byte-level
#                                       scan over the whole doc false-
#                                       positives on %XX in path/query)
#   Layer 6 (abs http(s) dest host): $md_dests + $attr_dests
#                                       (post-Layer-5 so 4/5 fire with
#                                       their more-specific diagnostics;
#                                       6 then catches the github.com-
#                                       prefix-of-longer-disallowed-host
#                                       bypass that Layer 1's per-line
#                                       view cannot see).

# Host is allowed iff it is exactly `github.com` / `llvm.org` OR ends in
# `.github.com` / `.githubusercontent.com` / `.llvm.org`. Used by Layers
# 1 / 2b / 3b / 6.
ALLOWED_HOST_RE='^(github\.com|llvm\.org|[A-Za-z0-9._-]+\.(github\.com|githubusercontent\.com|llvm\.org))$'

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
disallowed_hosts=$(grep -oiE 'https?://[A-Za-z0-9._~:@/-]+' "$strings_decoded_tmp" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -E 's|^https?://([^/?#]+).*|\1|' \
  | sed -E 's|.*@||' \
  | sed -E 's|:[0-9]+$||' \
  | sort -u \
  | grep -vE "$ALLOWED_HOST_RE" \
  || true)
if [[ -n "$disallowed_hosts" ]]; then
  # Redaction policy for every URL-rejection diagnostic in this file
  # (Layers 1, 2a/b, 3a/b, 4, 5, 6): mask alphanumerics and `_-` with
  # `x`, keep structural punctuation. Model-supplied destinations can
  # encode secrets in subdomains / local-parts; printing verbatim would
  # turn the public Actions log into a leak channel. Maintainer keeps
  # enough shape (3-label host, port, brackets, ...) for triage; full
  # content stays in the uploaded actions.json artifact.
  echo "::error::actions.json contains URLs to disallowed hosts (redacted):"
  printf '%s\n' "$disallowed_hosts" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Only github.com / llvm.org (and *.github.com / *.githubusercontent.com / *.llvm.org) URLs are allowed in review bodies. See sanitize_claude_actions.sh / prompt's Hard constraints / doc §10."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 2: Markdown link destinations (inline + reference-style). Catches
# protocol-relative (`//evil/x`) and non-http(s) schemes (`mailto:`,
# `javascript:`, ...) that Layer 1's `https?://` pattern misses. Title
# text is stripped by the whitespace boundary in the destination regex.

# Inline destinations `](<dest>` or `](dest`. Extract from both the per-
# line view (well-formed inline) and the LF-stripped oneline view (LF-
# split inline whose host reassembles to a disallowed form -- per-line
# alone would truncate `[c](https://github.com\n.evil/x)` to a github.com
# prefix that Layer 1 allow-lists).
inline_dests=$(grep -oE '\]\([ \t]*<?[^[:space:]<>)]+' "$strings_decoded_tmp" \
  | sed -E 's|^\]\([ \t]*<?||' \
  || true)
inline_dests_oneline=$(grep -oE '\]\([ \t]*<?[^[:space:]<>)]+' "$strings_decoded_oneline_tmp" \
  | sed -E 's|^\]\([ \t]*<?||' \
  || true)
# Reference-style destinations `  [ref]:   <dest>  "title"`. Extract from
# $strings_tmp (RAW) so an entity-encoded LF inside the destination
# (`[1]: github.com&#10;.evil/x`) survives as a single token; the per-
# line decoded view would split on the decoded LF and Layer 6 would
# only see the allowed-host-prefix half. Two-step: line-anchored regex on
# raw, then per-dest html.unescape + ASCII tab/LF/CR strip. (Literal-LF
# ref-style is invalid CommonMark and renders as a benign first-half
# destination; the bypass is entity-encoded, where entities are TEXT in
# the markdown source and only the HTML render side decodes them.)
ref_dests_raw=$(grep -E '^[ \t]*\[[^]]+\]:[ \t]+' "$strings_tmp" \
  | sed -E 's|^[ \t]*\[[^]]+\]:[ \t]+<?([^[:space:]<>]+).*|\1|' \
  || true)
ref_dests=$(printf '%s\n' "$ref_dests_raw" \
  | python3 -c '
import html, sys
strip_table = {0x09: None, 0x0A: None, 0x0D: None}
for line in sys.stdin:
    line = line.rstrip("\n")
    if line:
        sys.stdout.write(html.unescape(line).translate(strip_table))
        sys.stdout.write("\n")
' \
  || true)
# Combine, drop empties, dedupe, lowercase for scheme/host comparisons.
md_dests=$( { printf '%s\n' "$inline_dests"; \
              printf '%s\n' "$inline_dests_oneline"; \
              printf '%s\n' "$ref_dests"; } \
  | grep -v '^$' \
  | tr '[:upper:]' '[:lower:]' \
  | sort -u \
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
  echo "::error::actions.json contains Markdown link destinations with non-http(s) schemes (redacted):"
  printf '%s\n' "$bad_scheme" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Only http(s)://github.com / http(s)://llvm.org URLs and allowed subdomains (allow-listed by Layer 1), in-repo paths, and fragment anchors are valid link destinations. mailto:, ftp:, javascript:, data:, file:, vbscript: etc. are rejected."
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
  echo "::error::actions.json contains protocol-relative Markdown link destinations to disallowed hosts (redacted):"
  printf '%s\n' "$bad_proto_rel" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Protocol-relative destinations (//host/...) resolve to the page's protocol; on a github.com page, //evil.example/x becomes https://evil.example/x. Only //github.com/... / //llvm.org/... (and *.github.com / *.githubusercontent.com / *.llvm.org) is allowed."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 3: HTML attribute destinations (href= and src=) on `<a>` and
# `<img>` (the only URL-bearing elements GitHub's HTML allow-list lets
# through for comments). Catches protocol-relative, non-http(s) schemes,
# bare-http(s), and tracking-pixel `<img src>` -- all of which bypass
# Layers 1/2 if not extracted attribute-aware. Reads
# $strings_decoded_oneline_tmp so entity-encoded and LF/CR/TAB-split
# attribute values are caught (HTML5 permits those bytes in attribute
# syntax; WHATWG URL §4.4 strips them in the browser parse).
# Handles double-, single-, and unquoted attribute syntax.
attr_dests=$(grep -oiE '(href|src)[[:space:]]*=[[:space:]]*("[^"]*"|'"'"'[^'"'"']*'"'"'|[^[:space:]>]+)' "$strings_decoded_oneline_tmp" \
  | sed -E 's|^[^=]*=[[:space:]]*||' \
  | sed -E 's|^"(.*)"$|\1|' \
  | sed -E "s|^'(.*)'$|\1|" \
  | grep -v '^$' \
  | tr '[:upper:]' '[:lower:]' \
  || true)

# Layer 3a: any non-http(s) scheme in an href/src -> reject.
attr_bad_scheme=$(printf '%s\n' "$attr_dests" \
  | grep -E '^[a-z][a-z0-9+.-]*:' \
  | grep -vE '^https?:' \
  | sort -u \
  || true)
if [[ -n "$attr_bad_scheme" ]]; then
  echo "::error::actions.json contains href= or src= attributes with non-http(s) schemes (redacted):"
  printf '%s\n' "$attr_bad_scheme" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Only http(s)://github.com / http(s)://llvm.org (and *.github.com / *.githubusercontent.com / *.llvm.org) URLs are valid href/src destinations. mailto:, ftp:, javascript:, data:, file:, vbscript: etc. are rejected."
  exit 2
fi

# Layer 3b: protocol-relative href/src -> check host against allow-list.
# Same regex as Layer 2b but against the attribute destinations.
attr_bad_proto_rel=$(printf '%s\n' "$attr_dests" \
  | grep -oE '^//[^/?#]+' \
  | sed -E 's|^//||' \
  | sed -E 's|.*@||' \
  | sed -E 's|:[0-9]+$||' \
  | sort -u \
  | grep -vE "$ALLOWED_HOST_RE" \
  || true)
if [[ -n "$attr_bad_proto_rel" ]]; then
  echo "::error::actions.json contains protocol-relative href= or src= attributes to disallowed hosts (redacted):"
  printf '%s\n' "$attr_bad_proto_rel" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Protocol-relative href/src (//host/...) resolves to the page's protocol; on a github.com page, //evil.example/x becomes https://evil.example/x. Only //github.com/... / //llvm.org/... (and *.github.com / *.githubusercontent.com / *.llvm.org) is allowed."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 4: bracketed-IP-literal hosts (RFC 3986 §3.2.2 IP-literal =
# "[" (IPv6 / IPvFuture) "]"). Categorical reject -- the host allow-list
# cannot apply to an IP literal (`[2606:...]` could be GitHub Pages or
# attacker-pinned), and allowed hosts are never reached via a raw IP.
# The single regex `(https?:)?//\[[^]]+\]` covers bare URLs, Markdown /
# protocol-relative destinations, HTML href/src, IPv6 / IPv4-mapped
# IPv6 / IPvFuture; entity-encoded brackets (`&#x5B;::1&#x5D;`) and
# LF-split brackets are caught via the decoded+stripped view. The `//`
# immediately followed by `[` requirement means bracketed PATH segments
# (`https://github.com/[::1]/x`) correctly don't trigger.
bracketed_hosts=$(grep -oiE '(https?:)?//\[[^]]+\]' "$strings_decoded_oneline_tmp" \
  | sort -u \
  || true)
if [[ -n "$bracketed_hosts" ]]; then
  echo "::error::actions.json contains URLs with bracketed-IP-literal hosts (redacted):"
  printf '%s\n' "$bracketed_hosts" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::IPv6 / IPvFuture URL authorities (https://[2606:...]/x, //[2606:...]/x, etc.) are categorically rejected. Allowed hosts are never reached via a raw IP literal; reference resources by hostname so the host allow-list can apply."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 5: percent-encoded authorities. Per WHATWG URL the host is
# percent-decoded before resolution -- so `%65vil/x` renders as `evil/x`,
# `github.com%2eevil/x` becomes a subdomain of evil. Categorical reject:
# allowed hosts are written as literal ASCII hostnames and never need
# %XX in the authority. Detection extracts AUTHORITIES from three contexts (bare
# URL, $md_dests, $attr_dests) then rejects any containing `%`. The
# AUTHORITY-only scope avoids false positives on valid %XX in path /
# query / fragment (an earlier byte-level scan over the whole document
# rejected legitimate `github.com/.../file%20with%20spaces.txt`).
# Bare-URL extraction uses a wider char class than Layer 1's (includes
# `%`) so percent-encoded-authority URLs are seen at all.
bare_pct_auths=$(grep -oiE 'https?://[A-Za-z0-9._~:@/%-]+' "$strings_decoded_oneline_tmp" \
  | sed -E 's|^https?://([^/?#]*).*|\1|' \
  || true)
# Markdown-destination authorities. $md_dests is one entry per line,
# already lowercased + entity-decoded. Destinations without `//`
# (in-repo paths, fragment anchors) yield no match and are dropped.
md_pct_auths=$(printf '%s\n' "$md_dests" \
  | grep -oE '^(https?:)?//[^/?#]*' \
  | sed -E 's|^(https?:)?//||' \
  || true)
# HTML attribute authorities (same shape; $attr_dests is one entry per
# line, lowercased, sourced from the LF-stripped oneline view).
attr_pct_auths=$(printf '%s\n' "$attr_dests" \
  | grep -oE '^(https?:)?//[^/?#]*' \
  | sed -E 's|^(https?:)?//||' \
  || true)
# Reject if any extracted authority contains `%`.
pct_authorities=$( { printf '%s\n' "$bare_pct_auths" "$md_pct_auths" "$attr_pct_auths"; } \
  | grep -v '^$' \
  | grep -F '%' \
  | sort -u \
  || true)
if [[ -n "$pct_authorities" ]]; then
  echo "::error::actions.json contains URLs with percent-encoded authorities (redacted):"
  printf '%s\n' "$pct_authorities" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Per the WHATWG URL spec the host component is percent-decoded before resolution, so https://%65vil.example/x renders as https://evil.example/x and https://github.com%2eevil.example/x renders as a subdomain of evil.example. Allowed hosts are written as literal ASCII hostnames and never legitimately need percent-encoding in the authority; all percent-encoded authorities are categorically rejected. Reference resources by literal hostname instead. (Percent-encoding in the URL path / query / fragment is unaffected -- only the authority is checked.)"
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 6: absolute http(s):// destination host check on $md_dests +
# $attr_dests (both sourced from the LF-stripped oneline view). Closes
# the LF/CR/TAB-split-host bypass that Layer 1's per-line view cannot
# see when the truncated form is an allowed-host PREFIX of a longer
# disallowed host (`<a href="https://github.com\nfoo.evil/x">`: per-line
# host extracts to `github.com` and passes; browser resolves to
# `github.com.foo.evil/x`). Runs AFTER Layers 4 (bracketed-IP) and 5
# (percent-auth) so those layers' more-specific diagnostics fire for
# their bypass classes; Layer 6 sees only vanilla "host not in allow-
# list" cases. Authority-extraction sed chain matches Layer 1's: strip
# scheme, strip userinfo (RFC 3986 §3.2.1 -- last @), strip port. The
# `^https?://` filter scopes to absolutes; in-repo paths / fragment
# anchors / non-http(s) schemes are handled by Layer 2a, and proto-
# relative by Layer 2b / 3b.
abs_disallowed=$( { printf '%s\n' "$md_dests" "$attr_dests"; } \
  | grep -v '^$' \
  | grep -E '^https?://' \
  | sed -E 's|^https?://([^/?#]+).*|\1|' \
  | sed -E 's|.*@||' \
  | sed -E 's|:[0-9]+$||' \
  | sort -u \
  | grep -vE "$ALLOWED_HOST_RE" \
  || true)
if [[ -n "$abs_disallowed" ]]; then
  echo "::error::actions.json contains http(s):// Markdown link destinations or HTML href= / src= attributes to disallowed hosts (redacted):"
  printf '%s\n' "$abs_disallowed" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Only http(s)://github.com / http(s)://llvm.org (and *.github.com / *.githubusercontent.com / *.llvm.org) URLs are allowed as Markdown link destinations or HTML href / src attribute values. Layer 1 catches the same shape in bare-URL prose; this layer makes the rejection destination-aware and closes the LF/CR/TAB-split bypass that Layer 1's per-line view cannot detect when the truncated host is an allowed-host prefix of a longer disallowed host (see WHATWG URL §4.4)."
  exit 2
fi

echo "Sanitizer OK: ${inline_count} inline comments, ${thread_count} thread updates, ${actual_bytes} bytes."
