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
#   - LLM-Gateway env-var-NAME scan (same -- covers all strings)
#   - LLM-Gateway env-var-VALUE scan: fixed-string-greps the runtime
#     values of ANTHROPIC_BASE_URL, LLM_GATEWAY_KEY, and USER_NTID over
#     all model strings (raw + entity-decoded views), redacting the
#     diagnostic so the value itself never lands in the log. Catches the
#     bypass where a prompt-injected response exfiltrates a known secret
#     by VALUE without ever naming the env var
#   - URL allow-list with 6 progressive layers (bare http(s) URLs;
#     protocol-relative URLs; markdown link destinations; HTML href / src
#     attribute destinations; backslash / percent-encoded / IDN-confusable
#     authority forms; explicit http(s):// hosts) -- the only URLs the
#     model may emit in any string are `*.github.com` and
#     `*.githubusercontent.com`; everything else fails closed
#   - anti-spoofing scan for the literal substring `<!-- claude-pr-review-`
#     in any model-supplied string. `post_claude_review.sh` reserves the
#     `<!-- claude-pr-review-* -->` marker space (master marker + per-
#     action sub-markers) to attribute "our" comments and classify
#     replies; a model body that contained one would let an attacker
#     mint a fake bot comment that the next reconciliation run treats
#     as ours (suppressing real findings, or flipping the resolve /
#     clarify dedup gate). Reject any occurrence anywhere
#   - thread_updates[] reference cross-check against
#     /tmp/pr/prev_comments.json: every claude_comment_id must resolve
#     to a Claude root review comment (BOT_LOGIN + master marker +
#     in_reply_to_id == null) and every non-null human_reply_id must be
#     a human reply IN THAT SAME thread. Without this, a prompt-
#     injected ID could drop a `+1` reaction on any review comment in
#     the entire repo, or post a "resolved" reply under a human
#     reviewer's thread under the bot's identity. Fails closed if
#     PREV_COMMENTS_FILE is missing/unparseable while thread_updates is
#     non-empty
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

# Verdict must be one of the three allowed strings. The action's
# --json-schema already enforces this; the re-check here is
# defense-in-depth against a schema regression. Do NOT echo the raw
# .verdict value: this sanitizer runs in the secret-bearing review
# job and a prompt-injected payload could place a secret-shaped
# string in .verdict, which would land in Actions logs before the
# secret-pattern scan layer below gets to catch it.
verdict=$(jq -r '.verdict // empty' "$ACTIONS_FILE")
case "$verdict" in
  APPROVE|REQUEST_CHANGES|COMMENT) ;;
  *)
    echo "::error::actions.json has missing or invalid .verdict; must be APPROVE, REQUEST_CHANGES, or COMMENT (raw value not printed -- model-controlled content, see CLAUDE_AUTO_REVIEW.md secret-redaction policy)"
    exit 1
    ;;
esac

# Summary must be a non-empty string. The action's --json-schema enforces
# `type: string` + `minLength: 1`; the re-check here is defense-in-depth
# against a schema regression. The post script prepends a body header
# (verdict + finding counts) and appends the marker, bracketing the
# summary, so an empty / missing / non-string summary would still
# produce a structurally non-empty body -- but a review whose model-
# authored portion is missing or malformed is meaningless and a strong
# signal that something is wrong upstream. Fail closed.
#
# `jq -e` exits non-zero when the filter result is false, null, or
# empty, so the combined predicate rejects: missing key, null,
# non-string type (number / boolean / array / object), and empty
# string. This matches the schema's `{type: "string", minLength: 1}`
# constraint exactly.
if ! jq -e '.summary | type == "string" and length > 0' "$ACTIONS_FILE" >/dev/null; then
  echo "::error::actions.json .summary must be a non-empty string (--json-schema type:string + minLength:1 should have rejected this)"
  exit 1
fi

# Defense-in-depth: every other field that the later byte-length /
# fence / marker scans dereference as a string must actually BE a
# string (or null, for optional fields). The schema enforces `type:
# "string"` on each; the re-check here closes the same redaction-
# bypass class as the verdict / summary checks above:
#
#   If a schema regression slipped, say, an object into
#   inline_comments[].body, the later `utf8bytelength` scan would
#   error with `jq: error (at ...): object ({"hidden":"sk-secret...
#   only strings have UTF-8 byte length` -- jq truncates the value
#   at ~10 characters, but that partial-secret prefix would land in
#   the public Actions log before the secret-pattern scan ever ran.
#   The `test` / `contains` / `split` operations in the suggestion-
#   contract, body-field fence-guard, and marker-spoof checks have
#   the same property.
#
# Predicate: every value of the stream must be null (optional fields
# missing) or a string. `jq -e` exits non-zero on false / null /
# empty, failing the sanitizer with a fixed-text error so no model-
# controlled content reaches the log.
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

# Validate every thread_updates[] reference is on the model's OWN
# Claude thread, not someone else's. Without this, the loose "ID
# exists in prev_comments.json" check would still let a prompt-
# injected payload reach human reviewer threads, because
# prev_comments.json is every inline review comment on the PR (bot +
# humans). Two-arm contract:
#
#   - claude_comment_id MUST resolve to a Claude *root* review
#     comment in prev_comments.json: user.login == $BOT_LOGIN,
#     body contains $CLAUDE_MARKER, in_reply_to_id == null. This is
#     the same "Claude root" definition the workflow's re-review
#     N-count uses (see the prefetch step in claude_auto_review.yml)
#     and the prompt's Step 1. A claude_comment_id that points at a
#     human reviewer's root would otherwise let the bot post a
#     "resolved" reply under that human's thread under the bot's own
#     identity.
#   - human_reply_id (when non-null) MUST resolve to a human reply
#     IN THAT SAME Claude thread: user.login != $BOT_LOGIN AND
#     in_reply_to_id == claude_comment_id. Without the thread tie-
#     back, a prompt-injected integer could drop a `+1` reaction on
#     any review comment in the PR -- or, given the reaction
#     endpoint's repo scope
#         POST /repos/<repo>/pulls/comments/<cid>/reactions
#     (no PR number in the path), any review comment in the entire
#     repo.
#
# BOT_LOGIN and CLAUDE_MARKER are inputs so the test harness can swap
# them; defaults match production. Both are duplicated across files
# (CLAUDE_AUTO_REVIEW.md §15 sync table); the workflow's sanitize
# step passes the same values it uses in the prefetch step + prompt.
# Fail-closed if PREV_COMMENTS_FILE is missing/unparseable while
# thread_updates is non-empty (we have nothing to validate against).
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

# =====================================================================
# Build the THREE views of the model's strings up front so EVERY
# downstream scan (secret/credential, env-var name, env-var value, URL
# allow-list, Markdown destination, HTML attribute, bracketed-IP-literal,
# percent-encoded authority) can consume them without re-running jq for
# each layer.
#
#   - $strings_tmp                 : raw bytes as the model emitted
#                                    them. Used for byte-exact checks
#                                    (e.g. matching what
#                                    update-pr-review will later
#                                    string-match against the GitHub-
#                                    stored `body` field).
#   - $strings_decoded_tmp         : the same text after HTML-entity
#                                    decoding, with per-line structure
#                                    preserved. GitHub's markdown
#                                    renderer entity-decodes link
#                                    destinations, href / src attribute
#                                    values, AND inline body text before
#                                    display. That means a payload like
#                                      `sk&#45;ant-1234abc...`
#                                      `Ocp&#45;Apim&#45;Subscription&#45;Key`
#                                      `https&#x3A;//evil.example/x`
#                                      `[click](&#x2F;&#x2F;evil.example/x)`
#                                      `<a href="&#x2F;&#x2F;evil.example/x">x</a>`
#                                    would RENDER as the unencoded form
#                                    to anyone reading the comment, but
#                                    a literal-bytes pattern check on
#                                    the raw model output never sees the
#                                    unencoded form. Decoding only ADDS
#                                    matches (an entity-encoded secret
#                                    becomes the plain secret after
#                                    decode), never removes them, so
#                                    concatenating raw + decoded is a
#                                    strict strengthening of the pre-
#                                    existing "raw only" checks.
#                                    Used by Layer 1 (bare URL), Layer 2
#                                    (Markdown destinations -- needs the
#                                    line structure for the `^[...]: `
#                                    ref-style anchor), the secret /
#                                    env-name / env-value scans, and
#                                    everywhere else that benefits from
#                                    line-by-line grep.
#   - $strings_decoded_oneline_tmp : entity-decoded AND with intra-string
#                                    ASCII tab / LF / CR stripped, one
#                                    JSON string per output line. Used
#                                    by Layer 3 (HTML href / src), Layer
#                                    4 (bracketed-IP-literal), and Layer
#                                    5 (percent-encoded authority).
#
#                                    Why a third view? grep is line-
#                                    oriented, and the WHATWG URL parser
#                                    is not. Per the URL Standard §4.4,
#                                    ASCII tab, LF, and CR are stripped
#                                    from URL strings during parsing.
#                                    GitHub's HTML sanitizer (cmark +
#                                    sanitization-filter) does not strip
#                                    those bytes from quoted attribute
#                                    values -- they are valid in HTML 5
#                                    attribute syntax. So a model output
#                                    of `<a href="//evil\nhost.com/x">`
#                                    or its entity-encoded twin
#                                    `<a href="//evil&#10;host.com/x">`
#                                    is rendered verbatim into the
#                                    `href` attribute, the browser then
#                                    strips the LF, and the URL parser
#                                    resolves the protocol-relative form
#                                    against the page scheme to
#                                    `https://evilhost.com/x`. Against
#                                    the per-line view, every URL layer
#                                    misses this:
#                                      Layer 1: no `https://` on either
#                                      half-line.
#                                      Layer 2: no `](` on either half-
#                                      line.
#                                      Layer 3: the unquoted attribute
#                                      alternative `[^[:space:]>]+`
#                                      matches `"//evil` on the first
#                                      line (the leading `"` is captured
#                                      because the closing quote is on
#                                      the next line and the on-same-
#                                      line `^"(.*)"$` unquote sed never
#                                      fires). The leading `"` then
#                                      defeats Layer 3a/3b's `^[a-z]`
#                                      and `^//` anchors.
#                                      Layer 4: no `//[` on either line.
#                                    The fix: align the view that the
#                                    URL layers see with the view the
#                                    browser sees, by stripping ASCII
#                                    tab / LF / CR per-string before
#                                    they run. Per-string (rather than
#                                    file-global) preserves the
#                                    inter-string newline that the
#                                    secret/env scans and Layer 2's
#                                    ref-style anchor still need on
#                                    $strings_decoded_tmp. Bare-URL and
#                                    Markdown-destination layers (1 and
#                                    2) intentionally keep the per-line
#                                    view: their regexes already
#                                    truncate at whitespace, so a
#                                    newline-split URL like
#                                    `https://evil\nhost.com/x` becomes
#                                    `https://evil` -- which fails the
#                                    host allow-list and is rejected --
#                                    and a stripped view would create a
#                                    false positive against perfectly
#                                    legitimate prose like
#                                    `See: https://github.com\ncontinue
#                                    reading.` (joins to
#                                    `https://github.comcontinue` whose
#                                    host is no longer github.com).
#
#                                    The strip is deliberately limited
#                                    to ASCII tab / LF / CR -- exactly
#                                    the three bytes WHATWG URL §4.4
#                                    drops -- to keep the contract
#                                    aligned with the browser. We do
#                                    NOT strip e.g. NUL or zero-width
#                                    chars: those have their own threat
#                                    model and are out of scope for
#                                    this view.
#
#                                    Caveat on Layer 1's per-line view:
#                                    the host-truncation argument
#                                    above closes the LF-split bypass
#                                    only for hosts whose TRUNCATED
#                                    form fails the allow-list (e.g.
#                                    `https://evil\nhost.com/x` -> `evil`
#                                    -- not in the allow-list). It
#                                    does NOT close the case where the
#                                    truncated host is a github.com
#                                    PREFIX of a longer disallowed
#                                    host (e.g.
#                                    `<a href="https://github.com\n.evil.com/x">`,
#                                    where Layer 1's per-line view
#                                    sees `https://github.com` and
#                                    happily allow-lists it, while the
#                                    browser-side strip resolves the
#                                    href to `https://github.com.evil.com/x`).
#                                    That residual gap is closed by
#                                    Layer 6 below (explicit http(s)://
#                                    destination host check on
#                                    md_dests + attr_dests, both of
#                                    which are sourced from the
#                                    oneline view): once destinations
#                                    are extracted from the LF-stripped
#                                    view, the full reassembled host
#                                    is run through the same authority
#                                    + ALLOWED_HOST_RE pipeline as
#                                    Layer 1, so a github.com prefix
#                                    is no longer a free pass.
#                                    Plain-prose bare URLs that span
#                                    LF (`Visit https://github.com\n
#                                    .evil.com/x for ...`) are not
#                                    autolinked by the markdown render
#                                    -- the render stops at LF -- so
#                                    that residual case is not a
#                                    reachable phishing vector and
#                                    Layer 1 stays on the per-line
#                                    view to keep the false-positive
#                                    note above intact.
#
# `python3` is preinstalled on every github-hosted runner; `html.unescape`
# handles all named, decimal, and hex entities per the WHATWG list.
#
# The marker-prefix anti-spoof check ABOVE intentionally stays on raw
# JSON via jq: an entity-encoded marker (`&lt;!-- claude-pr-review- &gt;`)
# would NOT match update-pr-review's later raw-byte string-match in the
# `body` field that GitHub stores, so it cannot affect dedup attribution
# in the way an unencoded marker would.
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

# Secret/credential pattern scan over every string in the document.
# Patterns are defined in secret_patterns.sh.
# Scans BOTH the raw and entity-decoded views so an entity-encoded secret
# (e.g. `sk&#45;ant-...`, `Ocp&#45;Apim&#45;Subscription&#45;Key`) does
# not slip past while still rendering as a real secret in the comment.
# Redact ALL alphanumerics and `_-` in the match preview: the matched
#
# Diagnostic-preview note (applies to this AND every later "preview"
# pipeline in this file): we use `awk 'NR<=N'` rather than `head -N`
# to truncate the preview output. This script runs under `set -euo
# pipefail`, and `head -N` closes its stdin after N lines -- if the
# upstream `echo "$hits"` / `printf '%s\n' "$..."` still has more to
# write, it dies with SIGPIPE (rc 141). Pipefail propagates that 141
# as the pipeline's exit, set -e fires, and the script aborts BEFORE
# the intended `exit 2` runs. The visible failure mode is the
# sanitizer occasionally returning exit 141 instead of exit 2 on
# multi-match inputs and skipping its own diagnostic. `awk 'NR<=N'`
# prints lines 1..N but reads input to EOF (no early close, no
# SIGPIPE upstream); the rest of the redaction pipeline is unchanged.
# line may include the secret VALUE on the same line, and printing it
# verbatim into the public Actions log would defeat the very secret-
# protection this sanitizer exists for. Structural punctuation (.,:/=
# etc.) stays so the maintainer can still tell what shape of string
# matched.
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

# Scan for the LITERAL VALUES of every sensitive env var the LLM Gateway
# step exposes to the model:
#   - ANTHROPIC_BASE_URL : the gateway URL itself.
#   - LLM_GATEWAY_KEY    : the APIM subscription key (HTTP header value).
#   - USER_NTID          : the org-internal user identifier.
# The env-var-NAME scan above catches a model that mentions the variable
# by name (the most common exfil-confirmation pattern), but a sufficiently
# determined prompt-injection attempt could exfiltrate any of these by
# VALUE alone -- without ever using the env var name. The value-shape
# patterns in secret_patterns.sh catch generic key formats, but a bare
# APIM subscription key or a bare NTID value is essentially indistin-
# guishable from any other random-ish string until it's matched as a
# fixed substring. The sanitize step in claude_auto_review.yml exports
# all three as env vars; we loop over them, fixed-string-grep BOTH the
# raw and entity-decoded views (so e.g. `s&#x6B;-ant-...` is caught), and
# fully redact the diagnostic so the value itself never lands in the log.
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
#   - raw HTML attribute destinations: GitHub's comment renderer accepts
#     a sanitized subset of HTML, including <a href="..."> and
#     <img src="...">. <a href="//evil.example/x">click</a>,
#     <a href="https://evil.example/x">click</a>,
#     <a href="mailto:t@evil">click</a>, and <img src="//evil/track.png">
#     all render as live links / auto-fetched images and bypass any
#     check that only looks at bare URLs and Markdown link syntax.
#   - entity-encoded variants of any of the above: GitHub's renderer
#     entity-decodes link destinations and href/src attribute values
#     before resolving them, so `https&#x3A;//evil.example/x`,
#     `[click](&#x2F;&#x2F;evil.example/x)`, and
#     `<a href="&#x2F;&#x2F;evil.example/x">click</a>` all render as
#     live links to evil.example yet would slip past a literal-bytes
#     pattern check. The decode pre-pass below ($strings_decoded_tmp)
#     normalizes these before the URL layers run.
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
# Used by Layer 1 (bare URL host check), Layer 2b (protocol-relative
# Markdown destination host check), Layer 3b (protocol-relative HTML
# href / src destination host check), and Layer 6 (explicit http(s)://
# Markdown / HTML destination host check, post-Layer-5).
ALLOWED_HOST_RE='^(github\.com|[A-Za-z0-9._-]+\.(github\.com|githubusercontent\.com))$'

# $strings_tmp / $strings_decoded_tmp / $strings_decoded_oneline_tmp
# are constructed near the top of the file (right after the marker-
# spoof check) so EVERY scan in this file can consume them.
#
# Routing summary (see the construction block above for full rationale):
#   - secret / env-name / env-value scans:
#       cat $strings_tmp $strings_decoded_tmp $strings_decoded_oneline_tmp
#       (raw + entity-decoded + LF/CR/TAB-stripped; the union strictly
#       strengthens detection).
#   - Layer 1 (bare URL):              $strings_decoded_tmp
#   - Layer 2 (Markdown destinations): $strings_decoded_tmp
#                                      + $strings_decoded_oneline_tmp
#                                      (inline destinations only)
#                                      + $strings_tmp + per-destination
#                                      entity-decode + LF/CR/TAB strip
#                                      (reference-style destinations)
#       (Layer 2's ref-style `^[ \t]*\[[^]]+\]:` extractor is line-
#       anchored. The decoded per-line view ($strings_decoded_tmp)
#       is unsafe to use here: an entity-encoded LF/CR/TAB inside
#       the destination (`[1]: https://github.com&#10;.evil.com/x`)
#       decodes to a real LF before the line-extractor runs and
#       splits the destination across lines, truncating the captured
#       host to a github.com prefix that Layer 6 then incorrectly
#       allow-lists -- a renderable bypass because the HTML
#       attribute parser entity-decodes the destination on the
#       render side, and WHATWG URL §4.4 then strips the LF in the
#       browser, resolving the href to the longer disallowed host.
#       The fix is to extract ref-style destinations from
#       $strings_tmp (RAW, where entity-encoded LF/CR/TAB are still
#       text and don't split lines), then entity-decode and strip
#       ASCII LF/CR/TAB on each captured destination -- aligning
#       the sanitizer's parse with the browser's. Inline
#       destinations `[txt](dest)` use both decoded views: per-line
#       catches well-formed inline links, the oneline view catches
#       LF-split inline destinations. Layer 6 host-checks the
#       union of all three sources, closing every LF/CR/TAB-split
#       bypass we know about in renderable destination contexts.)
#   - Layer 3 (HTML href/src):         $strings_decoded_oneline_tmp
#   - Layer 4 (bracketed-IP-literal):  $strings_decoded_oneline_tmp
#       (Layers 3 and 4 look for URL-shaped tokens that, when they
#       appear inside an HTML attribute value, can have ASCII tab /
#       LF / CR embedded by the model -- see WHATWG URL §4.4 -- and
#       the per-line view defeats their leading anchors. The oneline
#       view aligns the sanitizer's parse with the browser's.)
#   - Layer 5 (percent-encoded auth):  $strings_decoded_oneline_tmp
#                                      (bare-URL extraction)
#                                      $md_dests + $attr_dests
#                                      (computed in Layers 2 and 3)
#       (Layer 5 needs to isolate the AUTHORITY component of each URL
#       before checking for `%`, so it consumes already-extracted
#       destinations from Layers 2 and 3 plus a separate bare-URL
#       extraction with `%` allowed in the char class. A previous
#       byte-level scan that matched `(https?:)?//[^/?#]*%` anywhere
#       gave false positives for `%XX` in the URL path / query /
#       fragment; see Layer 5 commentary for details.)
#   - Layer 6 (explicit http(s)://     $md_dests + $attr_dests
#     destination host check):
#       (Closes the LF/CR/TAB-split-host bypass that Layer 1's per-
#       line view cannot detect when the truncated form is a
#       github.com PREFIX of a longer disallowed host. Runs AFTER
#       Layer 4 (bracketed-IP) and Layer 5 (percent-encoded
#       authority) so those layers' more-specific diagnostics fire
#       first for their respective bypass classes; Layer 6 then
#       host-checks the residual "vanilla" disallowed-host case on
#       the unioned md + attr destinations.)

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
  # Redaction policy for ALL URL-rejection diagnostics in this file
  # (Layer 1 here, Layer 2a/b, Layer 3a/b, Layer 4, Layer 5, Layer 6 below): mask all
  # alphanumerics and `_-` with `x`, leave structural punctuation
  # (`.`, `:`, `/`, `[`, `]`, `=`) intact. The host or destination we
  # rejected is MODEL-SUPPLIED content that, in a prompt-injection
  # scenario, can carry sensitive data in the host/subdomain (e.g.
  # `https://<encoded-secret>.evil.example`) or in the rest of the
  # destination (e.g. `mailto:secret@target` -- the local-part is the
  # leak). The sanitizer rejects the payload but the public Actions
  # log of a public-repo PR is itself a leak channel; printing the
  # destination verbatim would defeat the very secret-protection this
  # sanitizer exists for. Same redaction policy as the secret-pattern
  # and env-var-name match previews higher up. The maintainer can
  # still see the SHAPE of what got rejected (3-label host, port,
  # bracketed authority, etc.) for triage; the full content is in the
  # uploaded actions.json artifact for deeper inspection.
  echo "::error::actions.json contains URLs to disallowed hosts (redacted):"
  printf '%s\n' "$disallowed_hosts" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
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
# whitespace and angle brackets. Extract from BOTH the per-line view
# (catches well-formed inline destinations) and the LF/CR/TAB-stripped
# oneline view (catches LF-split inline destinations whose host is
# reassembled to a disallowed form by the renderer/browser per WHATWG
# URL §4.4). Per-line alone is insufficient: a payload like
# `[c](https://github.com\n.evil.com/x)` truncates on per-line to
# `https://github.com` (a github.com PREFIX that Layer 1 allow-lists),
# while the oneline view yields `https://github.com.evil.com/x` -- the
# real, disallowed host.
inline_dests=$(grep -oE '\]\([ \t]*<?[^[:space:]<>)]+' "$strings_decoded_tmp" \
  | sed -E 's|^\]\([ \t]*<?||' \
  || true)
inline_dests_oneline=$(grep -oE '\]\([ \t]*<?[^[:space:]<>)]+' "$strings_decoded_oneline_tmp" \
  | sed -E 's|^\]\([ \t]*<?||' \
  || true)
# Reference-style destinations: lines like `  [ref]:   <dest>  "title"`.
#
# Read from $strings_tmp (RAW), NOT $strings_decoded_tmp, so that an
# entity-encoded LF / CR / TAB inside the destination (e.g.
# `[1]: https://github.com&#10;.evil.com/x`) is still seen as a
# SINGLE line at extraction time. The decoded per-line view would
# have already split such a destination across two lines (`&#10;`
# decodes to a literal LF before the line-anchored regex runs), and
# the per-line ref-style match would only see the first half --
# `https://github.com`, a github.com PREFIX that Layer 6 then
# incorrectly allow-lists. The renderer/browser, by contrast, sees
# the entity-encoded form survive into the href attribute, the HTML
# attribute parser entity-decodes it into a real LF, and the URL
# parser strips ASCII LF/CR/TAB per WHATWG URL §4.4 -- so the
# resolved href is the longer disallowed host
# `https://github.com.evil.com/x`. Layer 6 catches this on a
# correctly-extracted ref-style destination, but ONLY if the
# destination arrives at md_dests with the entity-encoded LF intact
# (as a single token, before the line split). The two-step
# extraction below does that:
#   (1) line-anchored regex on the RAW view yields the destination
#       as a single string with any entity-encoded LF/CR/TAB still
#       in entity form (so it's a single token, not split);
#   (2) Python `html.unescape` then resolves the entities to real
#       bytes, and `translate(strip_table)` removes ASCII tab / LF /
#       CR -- aligning the captured destination with the post-
#       browser-strip URL the renderer ultimately resolves.
#
# CommonMark requires `[label]:` at column 0 (up to 3 spaces indent)
# and forbids LF in the destination, so a literal-LF case is invalid
# CommonMark and renders as a destination of just the first half --
# benign, like the bare-prose LF-split case (Layer 1 / Layer 6
# both correctly accept). The entity-encoded case is the bypass we
# close here: entity references are TEXT in the markdown source, so
# the parser captures the entire entity-encoded destination as one
# token, and only the HTML render side decodes them into real LFs.
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
  echo "::error::actions.json contains protocol-relative Markdown link destinations to disallowed hosts (redacted):"
  printf '%s\n' "$bad_proto_rel" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::Protocol-relative destinations (//host/...) resolve to the page's protocol; on a github.com page, //evil.example/x becomes https://evil.example/x. Only //github.com/... (and *.github.com / *.githubusercontent.com) is allowed."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 3: HTML attribute destinations (href= and src=).
#
# GitHub renders a sanitized subset of raw HTML inside markdown comments,
# notably <a href="..."> and <img src="...">. That gives a model output
# three more shapes that bypass Layers 1 and 2 if not also extracted:
#   (a) <a href="//evil.example/x">click</a>
#       -- protocol-relative href; browser resolves against page scheme.
#   (b) <a href="https://evil.example/x">click</a>
#       -- bare http(s) href; the bare-URL scan in Layer 1 already catches
#       the substring https://evil.example/x in the body, but only after
#       the entity-decode pre-pass. Layer 3 makes the rejection
#       attribute-aware so the error message points at the real shape.
#   (c) <a href="mailto:test@example.com">click</a> and any other non-
#       http(s) scheme (ftp:, javascript:, data:, file:, vbscript:).
#       Even when GitHub's HTML sanitizer strips javascript:/data:, the
#       mailto:/ftp: forms remain clickable, and the pipeline has no
#       legitimate use for any of them.
#   (d) <img src="//evil.example/track.png">
#       -- tracking pixel / data exfil channel via auto-fetched image.
#   (e) <a href="//evil&#10;example.com/x">click</a> (newline-split
#       attribute values, literal or entity-encoded). HTML5 attribute
#       syntax permits LF/CR/TAB inside quoted attribute values, and
#       per WHATWG URL §4.4 the URL parser strips those bytes before
#       host resolution -- so the browser resolves the value as if the
#       LF/CR/TAB weren't there. A grep that operates on physical lines
#       sees `<a href="//evil` on the first line and would extract the
#       leading-quote-prefixed token `"//evil` via the unquoted-attr
#       alternative (`[^[:space:]>]+`, the on-same-line `^"(.*)"$`
#       unquote sed cannot match across lines), defeating Layer 3a/3b's
#       leading anchors. The fix is to read $strings_decoded_oneline_tmp
#       which has had ASCII tab / LF / CR stripped per-string, exactly
#       the bytes WHATWG strips -- aligning the sanitizer's parse with
#       the browser's.
# All extraction happens against $strings_decoded_oneline_tmp so:
#   - entity-encoded attribute values (e.g.
#     href="&#x2F;&#x2F;evil.example/x") are caught after entity decode;
#   - newline-split attribute values (e.g. href="//evil\nhost/x" or its
#     entity-encoded twin href="//evil&#10;host/x") are caught after the
#     intra-string LF/CR/TAB strip.
#
# Extraction handles three attribute-quoting forms:
#   - double-quoted: href="..."
#   - single-quoted: href='...'
#   - unquoted:      href=value-up-to-whitespace-or->
# We only look at href and src; other URL-bearing attributes (action,
# formaction, srcset, xlink:href) are not in GitHub's HTML allow-list
# for comments so they are stripped before render.
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
  echo "::error::Only http(s)://github.com (and *.github.com / *.githubusercontent.com) URLs are valid href/src destinations. mailto:, ftp:, javascript:, data:, file:, vbscript: etc. are rejected."
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
  echo "::error::Protocol-relative href/src (//host/...) resolves to the page's protocol; on a github.com page, //evil.example/x becomes https://evil.example/x. Only //github.com/... (and *.github.com / *.githubusercontent.com) is allowed."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 4: bracketed-IP-literal hosts (categorical rejection).
#
# Per RFC 3986 §3.2.2 the host component of a URL authority may be an
# IP-literal:
#
#     IP-literal = "[" ( IPv6address / IPvFuture ) "]"
#
# A bracketed authority bypasses every preceding URL check because:
#
#   - Layer 1's bare-URL grep pattern is `https?://[A-Za-z0-9._~:@/-]+`,
#     and `[` is NOT in that character class. Against `https://[::1]/x`
#     the `+` quantifier requires at least one match after `://`, the
#     next byte is `[`, and grep simply does not match. Layer 1 never
#     even sees the URL.
#   - Layer 2a / 3a (non-http(s) scheme) does not fire: the scheme IS
#     `https:` or absent.
#   - Layer 2b / 3b (protocol-relative) does not fire on `https://[::1]
#     /x` because the URL has an explicit scheme. The proto-relative
#     form `//[::1]/x` is the one shape these layers DO catch (their
#     `^//[^/?#]+` extraction yields `[::1]`, which fails the host
#     allow-list), but an attacker would simply use the explicit-scheme
#     form to bypass.
#
# Bracketed authorities are categorically rejected here. There is no
# legitimate review-body use case: github.com / gist.github.com / etc.
# are never reached via a raw IP literal, and the host allow-list logic
# ("`github.com` is OK; everything else isn't") cannot be applied to an
# IP literal in the first place -- you cannot tell from
# `[2606:50c0:8000::153]` whether it points at GitHub Pages or at an
# attacker's host that happens to have pinned the same address. Any
# legitimate cross-link can use the hostname instead.
#
# The single regex `(https?:)?//\[[^]]+\]` covers all bypass shapes:
#   - bare URL                   : https://[::1]/x
#   - Markdown destination       : [click](https://[::1]/x)
#   - protocol-relative Markdown : [click](//[::1]/x)
#   - HTML href / src            : <a href="https://[::1]/x">,
#                                  <a href="//[::1]/x">,
#                                  <img src="//[::1]/track.png">
#   - IPvFuture                  : https://[v1.fe80::a+en1]/x
#   - IPv4-mapped IPv6           : https://[::ffff:1.2.3.4]/x
#   - entity-encoded brackets    : https://&#x5B;::1&#x5D;/x
#                                  (caught after the entity-decode
#                                  pre-pass; brackets become real `[`
#                                  / `]` before this regex runs).
#   - newline-split brackets     : <a href="https://[::1\n]/x">
#                                  (caught after the LF/CR/TAB strip
#                                  in $strings_decoded_oneline_tmp;
#                                  see Layer 3 commentary above).
#
# The regex requires `//` to be IMMEDIATELY followed by `[`, so a
# bracketed segment in a URL PATH (e.g. `https://github.com/[::1]/x`)
# is correctly NOT matched -- only the AUTHORITY position is.
bracketed_hosts=$(grep -oiE '(https?:)?//\[[^]]+\]' "$strings_decoded_oneline_tmp" \
  | sort -u \
  || true)
if [[ -n "$bracketed_hosts" ]]; then
  echo "::error::actions.json contains URLs with bracketed-IP-literal hosts (redacted):"
  printf '%s\n' "$bracketed_hosts" | awk 'NR<=10' | sed 's/^/  - /' | sed -E 's/[A-Za-z0-9_-]/x/g'
  echo "::error::IPv6 / IPvFuture URL authorities (https://[2606:...]/x, //[2606:...]/x, etc.) are categorically rejected. github.com is never reached via a raw IP literal; reference resources by hostname so the host allow-list can apply."
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 5: percent-encoded authorities (categorical rejection).
#
# Per the WHATWG URL spec (https://url.spec.whatwg.org/#host-parsing),
# the host component of a URL is percent-decoded before IDNA processing
# and address resolution. Every modern browser (Chromium, WebKit, Gecko)
# implements this. Concretely:
#
#   https://%65vil.example/x          renders as evil.example/x
#                                     (`%65` decodes to `e`)
#   https://github.com%2eevil.com/x   renders as github.com.evil.com/x
#                                     (`%2e` decodes to `.`, turning
#                                     "github.com" into a subdomain)
#   https://%67ithub.com/x            renders as github.com/x
#                                     (`%67` decodes to `g`; benign by
#                                     itself, but a payload that LOOKS
#                                     like it points elsewhere when
#                                     scanned as raw bytes yet renders
#                                     as github.com is still a clear
#                                     prompt-injection-shaped output
#                                     and we want a hard NO on it)
#
# This bypasses every preceding URL layer:
#
#   - Layer 1's bare-URL regex `https?://[A-Za-z0-9._~:@/-]+` does NOT
#     include `%`. Against `https://%65vil.example/x` the `+` quantifier
#     requires at least one match after `://`, the next byte is `%`, no
#     match. Against `https://github.com%2eevil.example/x` the regex
#     greedily matches `https://github.com` (stopping at `%`), the host
#     extracts as `github.com`, and the allow-list passes -- silently
#     letting through a URL that the browser resolves to evil.example.
#   - Layer 2a / 3a (non-http(s) scheme) does not fire: the scheme IS
#     `https:` (or absent and protocol-relative).
#   - Layer 2b / 3b (protocol-relative) only run on URLs that LACK an
#     explicit scheme; an attacker simply uses `https://%65vil/x` to
#     bypass.
#   - Layer 4 (bracketed-IP-literal) requires a literal `[`. The
#     percent-encoded variant `https://%5B::1%5D/x` has no literal `[`,
#     so Layer 4 doesn't fire either; this layer catches it instead.
#
# Same reasoning as Layer 4: github.com / *.github.com /
# *.githubusercontent.com hostnames are pure ASCII alphanumeric + `.-`
# and never legitimately require percent-encoding in the authority.
# Anything with `%XX` in the authority position is categorically
# rejected, so we don't have to reason about percent-decode
# normalization, overlong / double-percent-encoded sequences, or IDNA
# round-trips. Percent-encoding in the PATH, QUERY, or FRAGMENT is
# unaffected (e.g. `https://github.com/owner/repo/blob/main/file%20with%20spaces.txt`
# and `https://github.com/foo?q=hello%20world` and
# `https://github.com/foo#section%20one` all pass) -- only the
# AUTHORITY position is checked.
#
# Detection: extract URL DESTINATIONS from three real URL contexts
# (bare URL, Markdown destination, HTML href / src), isolate the
# AUTHORITY component of each (between `//` and the first `/`, `?`,
# `#`, or end-of-destination), then categorically reject any
# authority containing `%`.
#
# An earlier shape of this layer used a single byte-level regex
# `(https?:)?//[^/?#[:space:]]*%[^/?#[:space:]]*` scanning the entire
# document. That regex matched `//foo%bar` substrings ANYWHERE,
# including inside the URL PATH (e.g.
# `https://github.com/a//foo%2fbar` -- a valid github.com URL where
# `%2f` is in the path), inside the URL QUERY (e.g.
# `https://github.com/foo?next=//evil%2eexample/x` -- a query
# parameter, not a destination the browser will resolve), and inside
# the URL FRAGMENT (e.g. `https://github.com/foo#section%20//more%2e`).
# Those are valid github.com URLs that the prior version incorrectly
# rejected; the destination-then-authority extraction below is the
# fix.
#
# The three contexts cover every URL shape the model can output:
#   - Bare URLs (this file's text):
#       https://%65vil.example/x          (Layer 1's char class
#                                          excludes `%`, so Layer 1
#                                          can't extract this; the
#                                          extraction here uses a
#                                          wider class that includes
#                                          `%` so the URL is seen)
#       https://github.com%2eevil.example/x (subdomain trick)
#       https://%67ithub.com/x            (prefix-substitution trick)
#       https://%5B::1%5D/x               (percent-encoded brackets;
#                                          Layer 4 misses this shape)
#   - Markdown destinations ($md_dests, computed in Layer 2):
#       [click](https://%65vil.example/x)
#       [1]: https://%65vil.example/x
#       [click](//%65vil.example/x)       (also Layer 2b)
#   - HTML attribute destinations ($attr_dests, computed in Layer 3):
#       <a href="https://%65vil/x">click</a>
#       <a href="//%65vil/x">click</a>    (also Layer 3b)
#       <img src="//%65vil/track.png">    (also Layer 3b)
#
# Reads `$strings_decoded_oneline_tmp` (the entity-decoded + per-string
# LF/CR/TAB-stripped view) for the bare-URL extraction, so:
#   - entity-encoded percent signs (`&#37;65vil.example`) are caught
#     after the entity-decode pre-pass;
#   - literal percent-encoded authorities survive the decode unchanged
#     and are caught directly;
#   - newline-split percent-encoded authorities inside HTML attributes
#     (e.g. `<a href="//e\nvil%2eexample/x">`) are caught because
#     $attr_dests itself is computed from the LF-stripped oneline view
#     in Layer 3, so the destination string is already whole by the
#     time we look for `%` here.
#
# Bare-URL extraction. The class includes `%` (unlike Layer 1's class)
# so percent-encoded-authority URLs are seen at all. After extracting,
# the same `^https?://([^/?#]*).*` sed pipeline Layer 1 uses isolates
# the authority. URLs with no `%` in their authority (the common case)
# yield an authority that doesn't contain `%` and are therefore
# correctly NOT flagged here.
bare_pct_auths=$(grep -oiE 'https?://[A-Za-z0-9._~:@/%-]+' "$strings_decoded_oneline_tmp" \
  | sed -E 's|^https?://([^/?#]*).*|\1|' \
  || true)
# Markdown-destination authorities. $md_dests is one destination per
# line, already lowercased and entity-decoded by Layer 2's pipeline.
# `grep -oE '^(https?:)?//[^/?#]*'` matches the scheme + authority
# prefix of any destination starting with `(https?:)?//` (i.e. bare
# http(s) destinations and protocol-relative ones); destinations
# without `//` (in-repo paths, fragment anchors) yield no match and
# are dropped.
md_pct_auths=$(printf '%s\n' "$md_dests" \
  | grep -oE '^(https?:)?//[^/?#]*' \
  | sed -E 's|^(https?:)?//||' \
  || true)
# HTML attribute authorities. Same shape as md_pct_auths but reading
# from $attr_dests (computed in Layer 3, also one destination per
# line, lowercased, with the LF/CR/TAB-stripped view applied).
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
  echo "::error::Per the WHATWG URL spec the host component is percent-decoded before resolution, so https://%65vil.example/x renders as https://evil.example/x and https://github.com%2eevil.example/x renders as a subdomain of evil.example. github.com / *.github.com / *.githubusercontent.com hostnames are pure ASCII and never legitimately need percent-encoding in the authority; all percent-encoded authorities are categorically rejected. Reference resources by literal hostname instead. (Percent-encoding in the URL path / query / fragment is unaffected -- only the authority is checked.)"
  exit 2
fi

# ---------------------------------------------------------------------
# Layer 6: explicit http(s):// destination host check (post-Layer-5).
#
# Layer 1 catches bare http(s)://disallowed.example/x in body prose,
# but on its $strings_decoded_tmp per-line view, an LF/CR/TAB-split
# URL inside a renderable destination context (a Markdown link
# destination or an HTML href / src attribute value) truncates at the
# embedded ASCII whitespace -- and if the truncated form is a
# github.com PREFIX of a longer disallowed host, Layer 1 happily
# allow-lists the prefix:
#
#   <a href="https://github.com\n.evil.com/x">click</a>
#       Layer 1's per-line view: the first line ends after
#       `https://github.com`. Layer 1's grep stops at LF and the host
#       extracts to `github.com`. ALLOWED_HOST_RE accepts. PASS.
#       Browser (after WHATWG URL §4.4 ASCII tab/LF/CR strip):
#       `https://github.com.evil.com/x` -- a disallowed host that
#       was never host-checked.
#
#   [c](https://github.com\n.evil.com/x)
#       Same situation in Markdown link form. GitHub's renderer is
#       more permissive than CommonMark in some shapes; even when it
#       refuses to render the link, defense in depth says we still
#       reject the destination.
#
# Layer 6 closes that gap by host-checking absolute http(s)://
# destinations in BOTH md_dests AND attr_dests, both of which include
# the LF/CR/TAB-stripped oneline view in their extraction (see Layer
# 2 inline_dests_oneline construction and Layer 3's read of
# $strings_decoded_oneline_tmp). After this layer, an LF-split
# disallowed host inside a renderable context is rejected even when
# its per-line truncation would have allow-listed.
#
# Why post-Layer-5? Layer 4 (bracketed-IP-literal) and Layer 5
# (percent-encoded authority) catch their respective bypass classes
# with more-specific diagnostics that point at the actual shape
# (`https://[::1]/x` -> "bracketed-IP-literal", `https://%65vil/x`
# -> "percent-encoded authorities"). If Layer 6 ran earlier, those
# URLs would be rejected here as plain "disallowed hosts" and the
# more-helpful error messages (and the operator's mental model of
# what failed) would be lost. After Layer 5, every percent-encoded
# authority and every bracketed authority has already been rejected,
# so Layer 6 only sees vanilla "host doesn't match the allow-list"
# cases.
#
# Authority-extraction sed chain matches Layer 1's exactly: strip
# scheme, strip userinfo per RFC 3986 §3.2.1 (everything before the
# LAST @), strip trailing port. The `^https?://` filter ensures we
# only host-check absolute http(s)://; Markdown destinations that are
# in-repo paths, fragment anchors, or non-http(s) schemes are caught
# by Layer 2a (or pass through as legitimate paths/anchors), and the
# protocol-relative shape is caught by Layer 2b / 3b.
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
  echo "::error::Only http(s)://github.com (and *.github.com / *.githubusercontent.com) URLs are allowed as Markdown link destinations or HTML href / src attribute values. Layer 1 catches the same shape in bare-URL prose; this layer makes the rejection destination-aware and closes the LF/CR/TAB-split bypass that Layer 1's per-line view cannot detect when the truncated host is a github.com prefix of a longer disallowed host (see WHATWG URL §4.4)."
  exit 2
fi

echo "Sanitizer OK: ${inline_count} inline comments, ${thread_count} thread updates, ${actual_bytes} bytes."
