#!/usr/bin/env bash
# Regression + bypass-class fixture suite for sanitize_claude_actions.sh.
#
# Every fixture builds a minimal actions.json (one thread_update body)
# from a literal payload, runs the sanitizer against it, and asserts an
# expected accept/reject outcome (and, for rejects, that an expected
# error-string fragment is present).
#
# Add a new fixture for every bypass class the sanitizer learns to
# close. The accompanying CI job re-runs this whole corpus on every PR
# that touches the sanitizer or the patterns it depends on, so a future
# change cannot silently regress past hardening.
#
# Usage:
#   bash .github/scripts/tests/test_sanitize.sh
#
# Exits 0 on full pass, 1 on any failure. Sets a couple of dummy env
# vars before running the sanitizer so the env-var-VALUE scan layer
# has predictable values to compare against -- the sanitizer otherwise
# silently skips that scan when the var is empty (which is the
# fork-context dry-run case).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANITIZER="$SCRIPT_DIR/../sanitize_claude_actions.sh"

if [[ ! -f "$SANITIZER" ]]; then
    echo "ERROR: sanitizer not found at $SANITIZER" >&2
    exit 1
fi

# Dummy values for the env-var-value scan. Chosen to NOT match any
# allowed github.com URL and to be unique enough that an accidental
# collision with fixture body content is essentially impossible.
export ANTHROPIC_BASE_URL="https://gw.example.internal"
export LLM_GATEWAY_KEY="sk-secret-xyzzy-12345"
export USER_NTID="alice.bob"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAIL_NAMES=()

# Print the captured sanitizer output for a failed fixture, truncated
# to a readable head. Implemented with bash parameter expansion (NOT
# `... | head -c 400`) deliberately: under the script's `set -euo
# pipefail`, a `head -c N` consumer that closes the pipe after N
# bytes will SIGPIPE the upstream `echo "$out"`, the pipeline exits
# 141, set -e fires, and the harness aborts in the middle of a
# multi-fixture failure -- losing every later case's output and the
# final summary. ${out:0:N} performs the truncation in-process before
# anything is written to a pipe, so the printf below has nothing to
# truncate downstream and the `| sed` cannot ever signal upstream.
print_fail_blob() {
    local blob="$1"
    printf '%s\n' "${blob:0:400}" | sed 's/^/        | /'
}

# Build a minimal actions.json with the given body in a single
# thread_update. Reading body from a file (rather than passing as a
# jq argument) so any literal byte (including LF, CR, TAB) round-trips
# exactly through jq's --rawfile.
#
# The thread_update is a schema-valid `clarify` entry (`type` +
# `claude_comment_id` + `body`), matching the actions.json shape
# that claude-code-action emits in production. An earlier fixture
# generator used `{path, line, side, body}` (the inline_comments
# shape), which the sanitizer's bad_thread check skipped silently
# because `.type` was null -- hiding any future regression in the
# type-specific validation rules and making the fixture suite less
# representative of production payloads. The string-scanning layers
# (URL allow-list, secret/env scans, marker spoof, etc.) all walk
# `[.. | strings]` so they see the body identically regardless of
# which thread_update shape carries it; using the schema-valid shape
# is strictly more accurate without changing what the URL/secret
# fixtures exercise.
make_blob() {
    local body_file="$1" out="$2"
    jq -nR --rawfile body "$body_file" \
        '{verdict:"COMMENT",
          summary:"x",
          inline_comments:[],
          thread_updates:[{type:"clarify",
                           claude_comment_id:1,
                           body:$body}]}' \
        > "$out"
}

# run_reject NAME BODY EXPECTED_ERROR_REGEX
# Asserts the sanitizer rejects the input (rc != 0) AND the stderr
# contains a match for the expected error fragment. Fragment may be
# an alternation: "(foo|bar)".
run_reject() {
    local name="$1" body="$2" want="$3"
    local body_file="$TMP_DIR/body" json="$TMP_DIR/in.json"
    printf '%s' "$body" > "$body_file"
    make_blob "$body_file" "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && echo "$out" | grep -qE "$want"; then
        PASS=$((PASS + 1))
        printf '  PASS  reject  %-44s rc=%d\n' "$name" "$rc"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("reject:$name")
        printf '  FAIL  reject  %-44s rc=%d want=/%s/\n' "$name" "$rc" "$want"
        print_fail_blob "$out"
    fi
}

# run_accept NAME BODY
# Asserts the sanitizer accepts the input (rc == 0).
run_accept() {
    local name="$1" body="$2"
    local body_file="$TMP_DIR/body" json="$TMP_DIR/in.json"
    printf '%s' "$body" > "$body_file"
    make_blob "$body_file" "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -eq 0 ]]; then
        PASS=$((PASS + 1))
        printf '  PASS  accept  %s\n' "$name"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("accept:$name")
        printf '  FAIL  accept  %s rc=%d\n' "$name" "$rc"
        print_fail_blob "$out"
    fi
}

# run_redact_check NAME BODY SECRET_SUBSTRING
# Asserts: sanitizer rejects, AND the secret substring does NOT appear
# anywhere in stderr (the structural redaction policy must mask all
# alphanumeric / `_-` characters with `x`).
run_redact_check() {
    local name="$1" body="$2" needle="$3"
    local body_file="$TMP_DIR/body" json="$TMP_DIR/in.json"
    printf '%s' "$body" > "$body_file"
    make_blob "$body_file" "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && ! echo "$out" | grep -qF -- "$needle"; then
        PASS=$((PASS + 1))
        printf '  PASS  redact  %s (substr "%s" not in stderr)\n' "$name" "$needle"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("redact:$name")
        printf '  FAIL  redact  %s rc=%d (substr "%s" leaked)\n' "$name" "$rc" "$needle"
        print_fail_blob "$out"
    fi
}

echo "=== sanitize_claude_actions.sh fixture suite ==="
echo
echo "--- Layer 1: bare URL host allow-list ---"
run_reject "bare evil URL"                  "See http://evil.com/x"                                "URLs to disallowed hosts"
run_reject "userinfo strip"                 "https://github.com@evil.com/x"                        "URLs to disallowed hosts"
run_reject "uppercase scheme evil"          "See HTTPS://EVIL.COM/x"                               "URLs to disallowed hosts"
run_reject "port + userinfo + evil"         "See https://gh.com:80@evil.example/x"                 "URLs to disallowed hosts"

echo
echo "--- Layer 2: Markdown destinations ---"
run_reject "md inline evil host"            '[click](http://evil.com/x)'                           "URLs to disallowed hosts|disallowed hosts"
run_reject "md ref-style evil"              $'[r][1]\n\n[1]: http://evil.com/x'                    "URLs to disallowed hosts|reference|disallowed"
run_reject "md proto-rel evil"              '[click](//evil.com/x)'                                "protocol-relative Markdown link"
run_reject "md mailto"                      '[ping](mailto:a@b.c)'                                 "non-http"
run_reject "md javascript:"                 '[c](javascript:alert(1))'                             "non-http"
run_reject "md data:"                       '[x](data:text/html,xx)'                              "non-http"
run_reject "md ftp:"                        '[x](ftp://evil/x)'                                    "non-http"
run_reject "md file:"                       '[x](file:///etc/passwd)'                              "non-http"
run_reject "md vbscript:"                   '[x](vbscript:msgbox(1))'                              "non-http"

echo
echo "--- Layer 3: HTML href / src attributes ---"
run_reject "html href dq evil"              '<a href="http://evil.com/x">x</a>'                    "URLs to disallowed hosts|href"
run_reject "html href sq evil"              "<a href='http://evil.com/x'>x</a>"                    "URLs to disallowed hosts|href"
run_reject "html href unq evil"             '<a href=http://evil.com/x>x</a>'                      "URLs to disallowed hosts|href"
run_reject "html img src evil"              '<img src="http://evil.com/p.png">'                    "URLs to disallowed hosts|src"
run_reject "html href mailto"               '<a href="mailto:a@b">x</a>'                           "href= or src= attributes with non-http"
run_reject "html src protorel evil"         '<img src="//evil.com/p.png">'                         "protocol-relative href"

echo
echo "--- Layer 3: newline / tab / CR-split attribute values (WHATWG §4.4) ---"
# Literal LF / CR / TAB inside attribute values. Browsers strip these
# before URL parsing per WHATWG URL §4.4; we strip them too in the
# oneline view that Layer 3 reads.
run_reject "lf inside dq href"              $'<a href="//evil\nexample.com/x">x</a>'              "(href|src|disallowed)"
run_reject "lf inside sq href"              $'<a href=\'//evil\nexample.com/x\'>x</a>'            "(href|src|disallowed)"
run_reject "lf inside img src"              $'<img src="//evil\ntracker.example/p.gif">'         "(href|src|disallowed)"
run_reject "lf inside javascript: scheme"   $'<a href="java\nscript:alert(1)">x</a>'             "non-http"
run_reject "lf inside mailto: scheme"       $'<a href="mai\nlto:secret@target">x</a>'            "non-http"
run_reject "cr inside dq href"              $'<a href="//evil\rexample.com/x">x</a>'             "(href|src|disallowed|protocol-relative)"
run_reject "tab inside dq href"             $'<a href="//evil\texample.com/x">x</a>'             "(href|src|disallowed|protocol-relative)"
# Entity-encoded LF / CR / TAB. These decode to the real bytes in the
# entity-decode pre-pass and are then stripped in the oneline view.
run_reject "ent &#10; inside href"          '<a href="//evil&#10;example.com/x">x</a>'           "(href|src|disallowed)"
run_reject "ent &NewLine; inside href"      '<a href="//evil&NewLine;example.com/x">x</a>'       "(href|src|disallowed)"
run_reject "ent &#13; inside href"          '<a href="//evil&#13;example.com/x">x</a>'           "(href|src|disallowed|protocol-relative)"
run_reject "ent &#9; inside href"           '<a href="//evil&#9;example.com/x">x</a>'            "(href|src|disallowed|protocol-relative)"

echo
echo "--- Layer 2c / 3c: LF-split absolute http(s) host (github.com prefix bypass) ---"
# A model-controlled href / Markdown destination of the shape
#   https://github.com\n.evil.com/x
# is the residual case Layer 1's per-line truncation cannot catch
# on its own: the truncated host `github.com` IS in the allow-list,
# so Layer 1 happily passes the URL while the renderer/browser, after
# the WHATWG URL §4.4 ASCII tab/LF/CR strip, resolves the destination
# to the disallowed host `github.com.evil.com`. Layer 2c (Markdown)
# and Layer 3c (HTML) close this by host-checking absolute http(s)://
# destinations after the oneline-view extraction. Variants below
# exercise LF, CR, TAB, and the entity-encoded versions of each in
# both the Markdown link destination and HTML attribute contexts.
run_reject "lf-split https host in href"    $'<a href="https://github.com\n.evil.com/x">x</a>'    "disallowed hosts"
run_reject "cr-split https host in href"    $'<a href="https://github.com\r.evil.com/x">x</a>'    "disallowed hosts"
run_reject "tab-split https host in href"   $'<a href="https://github.com\t.evil.com/x">x</a>'    "disallowed hosts"
run_reject "ent &#10; https host in href"   '<a href="https://github.com&#10;.evil.com/x">x</a>'  "disallowed hosts"
run_reject "ent &#13; https host in href"   '<a href="https://github.com&#13;.evil.com/x">x</a>'  "disallowed hosts"
run_reject "md lf-split https host"         $'[c](https://github.com\n.evil.com/x)'               "disallowed hosts"
run_reject "md cr-split https host"         $'[c](https://github.com\r.evil.com/x)'               "disallowed hosts"
run_reject "md tab-split https host"        $'[c](https://github.com\t.evil.com/x)'               "disallowed hosts"
run_reject "md ent &#10; https host"        '[c](https://github.com&#10;.evil.com/x)'             "disallowed hosts"
# Reference-style destinations carrying entity-encoded LF/CR/TAB.
# CommonMark forbids literal LF/CR/TAB in destinations, so the
# entity-encoded form is the only renderable bypass shape: entities
# are TEXT in the markdown source, so the parser captures the entire
# destination as one token, then HTML attribute parsing on the render
# side decodes them to real bytes that the browser strips per WHATWG
# URL §4.4 -- resolving the href to the longer disallowed host. The
# sanitizer matches the browser by extracting from the raw view and
# post-decoding + LF/CR/TAB-stripping each captured destination.
run_reject "ref-style ent &#10; host"       $'[c][1]\n\n[1]: https://github.com&#10;.evil.com/x'   "disallowed hosts"
run_reject "ref-style ent &#13; host"       $'[c][1]\n\n[1]: https://github.com&#13;.evil.com/x'   "disallowed hosts"
run_reject "ref-style ent &#9; host"        $'[c][1]\n\n[1]: https://github.com&#9;.evil.com/x'    "disallowed hosts"
run_reject "ref-style ent &NewLine; host"   $'[c][1]\n\n[1]: https://github.com&NewLine;.evil.com/x' "disallowed hosts"
run_reject "ref-style ent &#x0A; host"      $'[c][1]\n\n[1]: https://github.com&#x0A;.evil.com/x'  "disallowed hosts"
run_reject "ref-style angle-bracket ent LF" $'[c][1]\n\n[1]: <https://github.com&#10;.evil.com/x>' "disallowed hosts"
run_reject "ref-style indented ent LF"      $'[c][1]\n\n   [1]: https://github.com&#10;.evil.com/x' "disallowed hosts"

echo
echo "--- Layer 4: bracketed-IP-literal hosts (categorical reject) ---"
run_reject "bracketed IPv6"                 'See https://[::1]/admin'                              "bracketed-IP-literal"
run_reject "bracketed IPvFuture"            'See https://[v1.fe80::]/admin'                        "bracketed-IP-literal"
run_reject "bracketed IPv4-mapped"          'See https://[::ffff:127.0.0.1]/admin'                 "bracketed-IP-literal"
run_reject "bracketed proto-rel md"         '[click](//[::1]/admin)'                               "(bracketed-IP-literal|protocol-relative Markdown)"
run_reject "bracketed in href"              '<a href="https://[::1]/x">x</a>'                      "bracketed-IP-literal"
run_reject "lf-split bracketed in href"     $'<a href="https://[::1\n]/x">x</a>'                  "bracketed-IP-literal"

echo
echo "--- Layer 5: percent-encoded authorities (categorical reject) ---"
run_reject "pct host bare"                  "See https://%65vil.example/x"                         "percent-encoded authorities"
run_reject "pct subdomain trick"            "See https://github.com%2eevil.example/x"              "percent-encoded authorities"
run_reject "pct prefix-sub trick"           "See https://%67ithub.com/foo"                         "percent-encoded authorities"
run_reject "pct in md inline"               '[c](https://%65vil.example/x)'                        "percent-encoded authorities"
run_reject "pct in md ref"                  $'[c][1]\n\n[1]: https://%65vil.example/x'             "percent-encoded authorities"
run_reject "pct in md proto-rel"            '[c](//%65vil.example/x)'                              "(percent-encoded authorities|protocol-relative)"
run_reject "pct in href dq"                 '<a href="https://%65vil.example/x">x</a>'             "percent-encoded authorities"
run_reject "pct in href sq"                 "<a href='https://%65vil.example/x'>x</a>"             "percent-encoded authorities"
run_reject "pct in href unq"                '<a href=https://%65vil.example/x>x</a>'               "percent-encoded authorities"
run_reject "pct in img src"                 '<img src="https://%65vil.example/p.png">'             "percent-encoded authorities"
run_reject "pct in href proto-rel"          '<a href="//%65vil.example/x">x</a>'                   "(percent-encoded authorities|protocol-relative)"
run_reject "pct-encoded brackets"           'See https://%5B::1%5D/admin'                          "percent-encoded authorities"
run_reject "entity-then-pct"                'See &#104;ttps://%65vil.example/x'                    "percent-encoded authorities"
run_reject "pct-encoded percent"            'See https://%2565vil.example/x'                       "percent-encoded authorities"
run_reject "lf-split pct authority"         $'<a href="//e\nvil%2eexample.com/x">x</a>'           "(percent-encoded authorities|disallowed)"

echo
echo "--- Entity-encoded URL variants (entity-decode pre-pass) ---"
run_reject "entity-encoded URL bare"        '&#104;ttps://evil.com/x'                              "URLs to disallowed hosts"
run_reject "entity-encoded md dest"         '[c](&#x68;ttp://evil.com/x)'                          "URLs to disallowed hosts"
run_reject "entity-encoded href"            '<a href="&#104;ttp://evil.com/x">x</a>'               "URLs to disallowed hosts"

echo
echo "--- Secret + env-var name + env-var value scans (raw + decoded views) ---"
run_reject "raw secret pattern"             'token=sk-secret-xyzzy-12345 hi'                       "secret-like|LLM_GATEWAY_KEY"
run_reject "USER_NTID literal"              'logged as alice.bob'                                  "the USER_NTID value"
run_reject "LLM_GATEWAY_KEY literal"        'key=sk-secret-xyzzy-12345'                            "the LLM_GATEWAY_KEY value|secret-like"
run_reject "ANTHROPIC_BASE_URL literal"     'POST https://gw.example.internal/foo'                 "the ANTHROPIC_BASE_URL value"
run_reject "entity-encoded NTID value"      'user a&#108;ice.bob did this'                         "the USER_NTID value"
run_reject "entity-encoded ANTH URL"        'see &#104;ttps://gw.example.internal/foo'             "the ANTHROPIC_BASE_URL value"
run_reject "entity-encoded LLM key"         'k=&#115;k-secret-xyzzy-12345'                         "the LLM_GATEWAY_KEY value|secret-like"
# LF-split secret / NTID / URL: the oneline view now also catches
# secrets / env-var values whose bytes are split across a literal LF.
run_reject "lf-split NTID"                  $'logged as al\nice.bob'                              "the USER_NTID value"
run_reject "lf-split LLM key"               $'key=sk-sec\nret-xyzzy-12345'                        "the LLM_GATEWAY_KEY value|secret-like"
run_reject "lf-split ANTHROPIC URL"         $'POST https://gw.exa\nmple.internal/foo'             "the ANTHROPIC_BASE_URL value"

echo
echo "--- Marker-spoof guard (PR-author cannot inject Claude markers) ---"
run_reject "spoofed master marker"          '<!-- claude-pr-review-marker:v1 -->'                  "marker"
run_reject "spoofed action marker"          '<!-- claude-pr-review-action:resolve -->'             "marker"
run_reject "spoofed bare prefix"            'prefix <!-- claude-pr-review-x --> suffix'            "marker"

echo
echo "--- Diagnostic-redaction (rejected hostname must not leak in stderr) ---"
run_redact_check "URL diag redacted host"   'http://supersecret-evil-host-12345.com/x'             "supersecret-evil-host-12345"
run_redact_check "pct-host diag redacted"   'http://supersecret-evil-host-12345%65xyz.com/x'       "supersecret-evil-host-12345"

echo
echo "--- Verdict field validation (formal review state) ---"
# Sanitizer re-checks the enum even though --json-schema does too,
# as defense-in-depth against a schema regression.
run_verdict_reject() {
    local name="$1" verdict_jq="$2" want="$3"
    local json="$TMP_DIR/in.json"
    # Build a payload with the given verdict expression. `--argjson verdict ...`
    # lets us pass either a string ("APPROVE") or `null` / a wrong-type value.
    if [[ "$verdict_jq" == "OMIT" ]]; then
        jq -n '{summary:"x", inline_comments:[], thread_updates:[]}' > "$json"
    else
        jq -n --argjson verdict "$verdict_jq" \
            '{verdict:$verdict, summary:"x", inline_comments:[], thread_updates:[]}' \
            > "$json"
    fi
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && echo "$out" | grep -qE "$want"; then
        PASS=$((PASS + 1))
        printf '  PASS  reject  %-44s rc=%d\n' "$name" "$rc"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("verdict-reject:$name")
        printf '  FAIL  reject  %-44s rc=%d want=/%s/\n' "$name" "$rc" "$want"
        print_fail_blob "$out"
    fi
}
run_verdict_accept() {
    local name="$1" verdict="$2"
    local json="$TMP_DIR/in.json"
    jq -n --arg verdict "$verdict" \
        '{verdict:$verdict, summary:"x", inline_comments:[], thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -eq 0 ]]; then
        PASS=$((PASS + 1))
        printf '  PASS  accept  verdict=%s\n' "$verdict"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("verdict-accept:$name")
        printf '  FAIL  accept  verdict=%s rc=%d\n' "$verdict" "$rc"
        print_fail_blob "$out"
    fi
}
run_verdict_reject "verdict missing"        'OMIT'                  "missing or invalid .verdict"
run_verdict_reject "verdict null"           'null'                  "missing or invalid .verdict"
run_verdict_reject "verdict empty string"   '""'                    "missing or invalid .verdict"
run_verdict_reject "verdict wrong enum"     '"approve"'             "missing or invalid .verdict"
run_verdict_reject "verdict freeform"       '"LGTM"'                "missing or invalid .verdict"
run_verdict_reject "verdict integer"        '0'                     "missing or invalid .verdict"
run_verdict_accept "verdict APPROVE"        "APPROVE"
run_verdict_accept "verdict REQUEST_CHANGES" "REQUEST_CHANGES"
run_verdict_accept "verdict COMMENT"        "COMMENT"

# Verdict is model-controlled and the sanitizer runs in the
# secret-bearing review job; assert the raw value never leaks to
# stderr on the rejection path.
run_verdict_redact_check() {
    local name="$1" verdict_jq="$2" needle="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson verdict "$verdict_jq" \
        '{verdict:$verdict, summary:"x", inline_comments:[], thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && ! echo "$out" | grep -qF -- "$needle"; then
        PASS=$((PASS + 1))
        printf '  PASS  redact  verdict %-30s (substr "%s" not in stderr)\n' "$name" "$needle"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("verdict-redact:$name")
        printf '  FAIL  redact  verdict %s rc=%d (substr "%s" leaked)\n' "$name" "$rc" "$needle"
        print_fail_blob "$out"
    fi
}
run_verdict_redact_check "secret-shaped"    '"sk-secret-xyzzy-12345"' "sk-secret-xyzzy-12345"
run_verdict_redact_check "host-shaped"      '"supersecret-evil-host"' "supersecret-evil-host"

echo
echo "--- Summary field validation (non-empty string contract) ---"
# Sanitizer re-checks .summary is a non-empty string even though the
# action's --json-schema enforces type:string + minLength:1, as
# defense-in-depth against a schema regression. The predicate rejects
# missing key, null, non-string types, and empty string -- matching
# the schema constraint exactly.
run_summary_reject() {
    local name="$1" summary_jq="$2" want="$3"
    local json="$TMP_DIR/in.json"
    if [[ "$summary_jq" == "OMIT" ]]; then
        jq -n '{verdict:"COMMENT", inline_comments:[], thread_updates:[]}' > "$json"
    else
        jq -n --argjson summary "$summary_jq" \
            '{verdict:"COMMENT", summary:$summary, inline_comments:[], thread_updates:[]}' \
            > "$json"
    fi
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && echo "$out" | grep -qE "$want"; then
        PASS=$((PASS + 1))
        printf '  PASS  reject  %-44s rc=%d\n' "$name" "$rc"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("summary-reject:$name")
        printf '  FAIL  reject  %-44s rc=%d want=/%s/\n' "$name" "$rc" "$want"
        print_fail_blob "$out"
    fi
}
run_summary_reject "summary missing"        'OMIT'                  "must be a non-empty string"
run_summary_reject "summary null"           'null'                  "must be a non-empty string"
run_summary_reject "summary empty string"   '""'                    "must be a non-empty string"
run_summary_reject "summary integer"        '42'                    "must be a non-empty string"
run_summary_reject "summary boolean"        'true'                  "must be a non-empty string"
run_summary_reject "summary array"          '["a","b"]'             "must be a non-empty string"
run_summary_reject "summary object"         '{"k":"v"}'             "must be a non-empty string"
# Accept case: any non-empty string passes.
run_summary_accept() {
    local name="$1" summary_jq="$2"
    local json="$TMP_DIR/in.json"
    jq -n --argjson summary "$summary_jq" \
        '{verdict:"COMMENT", summary:$summary, inline_comments:[], thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -eq 0 ]]; then
        PASS=$((PASS + 1))
        printf '  PASS  accept  %s\n' "$name"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("summary-accept:$name")
        printf '  FAIL  accept  %s rc=%d\n' "$name" "$rc"
        print_fail_blob "$out"
    fi
}
run_summary_accept "summary single char"    '"x"'
run_summary_accept "summary multi-line md"  '"## Scope\nfoo\n\n## Findings\nNone."'

# Summary is model-controlled and the sanitizer runs in the secret-
# bearing review job; assert that secret-shaped content nested inside
# a non-string summary never leaks to stderr on the rejection path.
# Today this holds trivially because the rejection error is fixed
# text -- the pin is here so a future maintainer who "helpfully" adds
# the summary value to the diagnostic catches it via a CI failure.
run_summary_redact_check() {
    local name="$1" summary_jq="$2" needle="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson summary "$summary_jq" \
        '{verdict:"COMMENT", summary:$summary, inline_comments:[], thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && ! echo "$out" | grep -qF -- "$needle"; then
        PASS=$((PASS + 1))
        printf '  PASS  redact  summary %-30s (substr "%s" not in stderr)\n' "$name" "$needle"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("summary-redact:$name")
        printf '  FAIL  redact  summary %s rc=%d (substr "%s" leaked)\n' "$name" "$rc" "$needle"
        print_fail_blob "$out"
    fi
}
run_summary_redact_check "secret in object value" '{"hidden":"sk-secret-leak-12345"}' "sk-secret-leak-12345"
run_summary_redact_check "secret in array"        '["sk-secret-leak-67890"]'         "sk-secret-leak-67890"
run_summary_redact_check "host-shaped in key"     '{"evil-host-domain":"x"}'         "evil-host-domain"

echo
echo "--- Field-type defense-in-depth (non-string string fields) ---"
# The sanitizer re-checks that inline_comments[].body, inline_comments[].
# suggestion, and thread_updates[].body are strings (or null, for optional
# fields) even though the action's --json-schema enforces type:string on
# each. The re-check closes the same redaction-bypass class as the
# verdict/summary type checks: a schema regression that slipped a non-
# string through would otherwise reach `utf8bytelength` / `test` /
# `contains` / `split` in later scans, and jq's error message includes
# the value (truncated at ~10 chars) -- partial-secret-leaking on
# stderr before the secret-pattern layer ever runs.

# Build a payload with a single inline_comments entry whose .body is the
# raw JSON expression supplied by the caller (--argjson, so the value
# round-trips with its declared type intact).
run_ic_body_reject() {
    local name="$1" body_jq="$2" want="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson v "$body_jq" \
        '{verdict:"COMMENT", summary:"x",
          inline_comments:[{path:"f.cpp",line:1,side:"RIGHT",severity:"Major",body:$v}],
          thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && echo "$out" | grep -qE "$want"; then
        PASS=$((PASS + 1))
        printf '  PASS  reject  %-44s rc=%d\n' "$name" "$rc"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("ic-body-reject:$name")
        printf '  FAIL  reject  %-44s rc=%d want=/%s/\n' "$name" "$rc" "$want"
        print_fail_blob "$out"
    fi
}
run_ic_body_reject "ic.body integer"        '42'                                "non-string value"
run_ic_body_reject "ic.body boolean"        'true'                              "non-string value"
run_ic_body_reject "ic.body array"          '["a","b"]'                         "non-string value"
run_ic_body_reject "ic.body object"         '{"k":"v"}'                         "non-string value"

# Same shape, but exercises inline_comments[].suggestion. Body is fixed
# to a valid string so the new check is the only gate that can fire.
run_ic_sugg_reject() {
    local name="$1" sugg_jq="$2" want="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson v "$sugg_jq" \
        '{verdict:"COMMENT", summary:"x",
          inline_comments:[{path:"f.cpp",line:1,side:"RIGHT",severity:"Major",
                            body:"ok",suggestion:$v}],
          thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && echo "$out" | grep -qE "$want"; then
        PASS=$((PASS + 1))
        printf '  PASS  reject  %-44s rc=%d\n' "$name" "$rc"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("ic-sugg-reject:$name")
        printf '  FAIL  reject  %-44s rc=%d want=/%s/\n' "$name" "$rc" "$want"
        print_fail_blob "$out"
    fi
}
run_ic_sugg_reject "ic.suggestion integer"  '42'                                "non-string value"
run_ic_sugg_reject "ic.suggestion array"    '["return foo();"]'                 "non-string value"
run_ic_sugg_reject "ic.suggestion object"   '{"code":"return foo();"}'          "non-string value"

# Same shape on a clarify thread_update body. (For resolve / resolve_
# with_reaction the schema does not carry a body field, so a non-string
# body there is the same regression shape but only this fixture exercises
# the user-facing clarify path.)
run_tu_body_reject() {
    local name="$1" body_jq="$2" want="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson v "$body_jq" \
        '{verdict:"COMMENT", summary:"x", inline_comments:[],
          thread_updates:[{type:"clarify",claude_comment_id:1,body:$v}]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && echo "$out" | grep -qE "$want"; then
        PASS=$((PASS + 1))
        printf '  PASS  reject  %-44s rc=%d\n' "$name" "$rc"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("tu-body-reject:$name")
        printf '  FAIL  reject  %-44s rc=%d want=/%s/\n' "$name" "$rc" "$want"
        print_fail_blob "$out"
    fi
}
run_tu_body_reject "tu.body integer"        '42'                                "non-string value"
run_tu_body_reject "tu.body array"          '["clarification"]'                 "non-string value"
run_tu_body_reject "tu.body object"         '{"k":"clarification"}'             "non-string value"

# Redaction: secret-shaped content nested inside a non-string string-
# field must never leak to stderr on the rejection path. This pins the
# fixed-text contract on the new defense-in-depth check the same way
# the summary-redact tests above pin it on the summary check; a future
# maintainer who "helpfully" adds the offending value to the
# diagnostic catches it via a CI failure.
run_ic_body_redact_check() {
    local name="$1" body_jq="$2" needle="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson v "$body_jq" \
        '{verdict:"COMMENT", summary:"x",
          inline_comments:[{path:"f.cpp",line:1,side:"RIGHT",severity:"Major",body:$v}],
          thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && ! echo "$out" | grep -qF -- "$needle"; then
        PASS=$((PASS + 1))
        printf '  PASS  redact  ic.body %-30s (substr "%s" not in stderr)\n' "$name" "$needle"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("ic-body-redact:$name")
        printf '  FAIL  redact  ic.body %s rc=%d (substr "%s" leaked)\n' "$name" "$rc" "$needle"
        print_fail_blob "$out"
    fi
}
run_ic_body_redact_check "secret in object"  '{"hidden":"sk-secret-bodyleak-11111"}' "sk-secret-bodyleak-11111"
run_ic_body_redact_check "secret in array"   '["sk-secret-bodyleak-22222"]'          "sk-secret-bodyleak-22222"

run_ic_sugg_redact_check() {
    local name="$1" sugg_jq="$2" needle="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson v "$sugg_jq" \
        '{verdict:"COMMENT", summary:"x",
          inline_comments:[{path:"f.cpp",line:1,side:"RIGHT",severity:"Major",
                            body:"ok",suggestion:$v}],
          thread_updates:[]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && ! echo "$out" | grep -qF -- "$needle"; then
        PASS=$((PASS + 1))
        printf '  PASS  redact  ic.suggestion %-24s (substr "%s" not in stderr)\n' "$name" "$needle"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("ic-sugg-redact:$name")
        printf '  FAIL  redact  ic.suggestion %s rc=%d (substr "%s" leaked)\n' "$name" "$rc" "$needle"
        print_fail_blob "$out"
    fi
}
run_ic_sugg_redact_check "secret in object" '{"k":"sk-secret-suggleak-33333"}' "sk-secret-suggleak-33333"

run_tu_body_redact_check() {
    local name="$1" body_jq="$2" needle="$3"
    local json="$TMP_DIR/in.json"
    jq -n --argjson v "$body_jq" \
        '{verdict:"COMMENT", summary:"x", inline_comments:[],
          thread_updates:[{type:"clarify",claude_comment_id:1,body:$v}]}' \
        > "$json"
    local out rc
    out=$(bash "$SANITIZER" "$json" 2>&1) || rc=$? && rc=${rc:-0}
    if [[ $rc -ne 0 ]] && ! echo "$out" | grep -qF -- "$needle"; then
        PASS=$((PASS + 1))
        printf '  PASS  redact  tu.body %-30s (substr "%s" not in stderr)\n' "$name" "$needle"
    else
        FAIL=$((FAIL + 1))
        FAIL_NAMES+=("tu-body-redact:$name")
        printf '  FAIL  redact  tu.body %s rc=%d (substr "%s" leaked)\n' "$name" "$rc" "$needle"
        print_fail_blob "$out"
    fi
}
run_tu_body_redact_check "secret in array"  '["sk-secret-threadleak-44444"]'   "sk-secret-threadleak-44444"

echo
echo "--- Negative cases: legitimate content must be accepted ---"
run_accept "plain prose"                    "Refactor foo() to return early"
run_accept "github bare URL"                "See https://github.com/foo/bar/issues/1"
run_accept "github md inline link"          '[issue](https://github.com/foo/bar/issues/1)'
run_accept "github md ref-style link"       $'See [issue][1].\n\n[1]: https://github.com/foo/bar/issues/1'
run_accept "github HTML href"               '<a href="https://github.com/foo">go</a>'
run_accept "raw.githubusercontent.com"      'See https://raw.githubusercontent.com/foo/bar/main/x'
run_accept "gist.github.com"                'See https://gist.github.com/foo/abcd'
run_accept "subdomain.github.com"           'See https://docs.github.com/en/rest'
run_accept "code fence cpp"                 $'```cpp\n#include <iostream>\nint main(){return 0;}\n```'
run_accept "code fence bash"                $'```bash\necho "hello"\n```'
run_accept "no URLs"                        "I think this should use a different lookup table for clarity"
# Legitimate %XX in github.com PATH / QUERY / FRAGMENT must still pass.
# Layer 5 extracts the AUTHORITY component of each destination and only
# checks that for `%`, so percent-encoding outside the authority is
# unaffected. The regression cases in this group cover an earlier shape
# of Layer 5 that scanned for `(https?:)?//[^/?#]*%` ANYWHERE in the
# document and would falsely reject any of these.
run_accept "pct in github path"             'See https://github.com/foo%20bar/baz'
run_accept "pct in github query"            'See https://github.com/foo?q=hello%20world'
run_accept "pct in github fragment"         'See https://github.com/foo#section%20one'
run_accept "pct in raw.gh path"             'See https://raw.githubusercontent.com/foo/bar/main/file%20with%20spaces.txt'
# Reviewer's regression repros (PR #2375): valid github.com URLs whose
# path / query / fragment contains both `//` and `%XX`. The earlier
# Layer 5 byte-level scan would falsely flag these as "percent-encoded
# authorities" because a `//foo%bar` substring inside the path looked
# like a protocol-relative URL with a percent-encoded host.
run_accept "double-slash in github path"    'See https://github.com/a//foo%2fbar'
run_accept "double-slash %2e in query"      'See https://github.com/foo?next=//evil%2eexample/x'
run_accept "double-slash + % in fragment"   'See https://github.com/foo#section%20//something%2eelse'
run_accept "%65 in query value (path %)"    'See https://github.com/foo?key=//evil%65xample.com/x'
run_accept "%5B brackets in github path"    'See https://github.com/foo/%5Bbar%5D'
# Multi-line legitimate content (LF inside a string but URL is on one line)
run_accept "multiline body w/ github URL"   $'See:\n  - https://github.com/foo/bar\n  - referenced in the docs'
run_accept "multiline prose no URLs"        $'This is paragraph one.\n\nThis is paragraph two.'

# Bare-prose / angle-bracket autolink LF-split URLs whose per-line
# truncation lands on a github.com prefix. These intentionally pass
# the sanitizer:
#
#   - Layer 1 stays on the per-line decoded view to avoid false-
#     positives on legitimate cross-LF prose like
#     `Visit https://github.com\nfor more info.` (joining LF-stripped
#     would give `https://github.comfor` -- a host that fails the
#     allow-list and rejects perfectly benign prose).
#   - Layer 6 (which DOES catch the LF-split-host bypass) is
#     intentionally scoped to renderable Markdown link destinations
#     and HTML href / src attribute values, because those are the
#     contexts where the renderer / browser actually reassembles the
#     LF-stripped href and resolves the disallowed host.
#   - For bare-prose and `<...>` autolink shapes, GitHub's markdown
#     renderer stops autolinking at LF in BOTH the bare-URL and
#     `<URL>` syntaxes. The rendered comment shows a clickable
#     `https://github.com` followed by a separate visual line of
#     plain text (`.evil.com/x`) -- never a single clickable link
#     to `github.com.evil.com/x`. The disallowed continuation is
#     therefore not a phishing vector under the bot identity, only
#     a confusable visual artifact a maintainer would see (and could
#     copy verbatim) at the cost of false-positives that would block
#     ordinary multi-line review prose.
#
# These accept-fixtures pin that contract so a future change cannot
# silently flip Layer 1 to the oneline view (which would create the
# false-positive problem above) without having to flip these
# fixtures explicitly.
run_accept "lf-split bare prose URL"        $'See https://github.com\n.evil.com/x for ...'
run_accept "lf-split autolink syntax"       $'<https://github.com\n.evil.com/x>'
run_accept "lf-split bare prose w/ space"   $'Visit https://github.com\nfor more info.'
run_accept "lf-split bare prose w/ path"    $'See https://github.com/foo/bar\nThis is the next paragraph.'

echo
echo "============================================================"
printf "  PASS: %3d   FAIL: %3d\n" "$PASS" "$FAIL"
echo "============================================================"
if [[ $FAIL -ne 0 ]]; then
    echo "Failed cases:"
    for n in "${FAIL_NAMES[@]}"; do
        echo "  - $n"
    done
    exit 1
fi
