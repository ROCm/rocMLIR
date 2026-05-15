#!/usr/bin/env bash
# Secret/credential pattern definitions for the Claude PR review pipeline.
#
# Sourced by sanitize_claude_actions.sh, which scans extracted JSON strings
# from actions.json before the file is uploaded as an artifact and consumed by
# the post job. Kept as a separate file so we can also exercise the patterns
# from local fixture tests without re-parsing the sanitizer.
#
# Patterns are kept as a single-line ERE alternation (not PCRE extended mode) to
# maximise grep portability across platforms. They are deliberately generous:
# a few false positives that block a build are far better than leaking a key.
#
# Pattern catalogue:
#   sk-ant-api##-...          real Anthropic API key
#   sk-[30+ chars]            generic OpenAI-style sk- key (excludes the dummy
#                             literal "sk-ant-dummy-gateway-key" which is too
#                             short to match)
#   Bearer <token>            HTTP Bearer auth tokens
#   Ocp-Apim-Subscription-Key: <value>   the gateway header itself
#   ghp_/gho_/ghu_/ghs_/ghr_/github_pat_  GitHub PATs and the GHA installation
#                             token (ghs_) that actions/checkout writes to
#                             .git/config when persist-credentials is on. We
#                             disable persist-credentials in the workflow, but
#                             this is the belt-and-braces backstop.
#   eC1hY2Nlc3MtdG9rZW46Z2hz  base64 of "x-access-token:ghs" -- the prefix of
#                             actions/checkout's basic-auth header. Catches an
#                             attacker that leaks the full AUTHORIZATION value.
#   xox[baprs]-...            Slack tokens
#   AKIA<16>                  AWS access key id
#   -----BEGIN ... PRIVATE KEY-----
export SUSPICIOUS_PATTERNS='sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{30,}|[Bb]earer[[:space:]]+[A-Za-z0-9_.-]{20,}|[Oo]cp-[Aa]pim-[Ss]ubscription-[Kk]ey[[:space:]]*:[[:space:]]*[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|eC1hY2Nlc3MtdG9rZW46Z2hz|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN[[:space:]]+([A-Z]+[[:space:]]+)?PRIVATE[[:space:]]+KEY-----'

# Names of the LLM-Gateway env vars. Mentioning them in either output is treated
# as a probable exfil attempt even if the value itself does not appear (an
# attacker may try to print the var name as a way of confirming their injection
# landed before the actual extraction).
export ENV_VAR_NAMES='ANTHROPIC_BASE_URL|LLM_GATEWAY_KEY|USER_NTID|ANTHROPIC_CUSTOM_HEADERS|ANTHROPIC_API_KEY'
