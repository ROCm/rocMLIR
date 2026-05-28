---
name: review-rocmlir-pr
description: Review a rocMLIR pull request with deep expertise in MLIR/LLVM coding standards, the Rock dialect, MIGraphX integration, kernel codegen for AMD GPUs, lit/E2E testing, and the rocMLIR CMake build. Use when asked to review a rocMLIR PR or check a rocMLIR change. Read-only; never posts to GitHub.
argument-hint: [PR-number]
agent: general-purpose
allowed-tools: Read, Grep, Glob
---

# rocMLIR PR Review

## IMPORTANT: Do NOT post to GitHub

This skill is **read-only**. Do NOT post any comments, reviews, or reactions. Do NOT use
`gh pr comment`, `gh pr review`, `gh api ... -X POST/PUT/PATCH/DELETE`. Posting is the
job of the workflow's post step, which runs in a separate job that does not have access
to the LLM Gateway secrets.

---

## Tool budget — READ THIS BEFORE STEP 1

This skill runs in one of two modes; the available tools are different in each.
Picking the wrong tool wastes turns on permission denials and can starve the
review of budget before it reaches the final JSON output.

**CI mode (default — this is your mode if you are reading this from
`.github/workflows/claude_auto_review.yml`).** The workflow passes
`--allowedTools "Skill,Read,Grep,Glob"` and `--json-schema '...'` to
claude-code-action; the final JSON answer is captured as the action's
`structured_output`. The pre-fetched context is already on disk under
`/tmp/pr/` (the workflow has already done all the `gh`/`jq` work for you):

| File | How to access |
|---|---|
| `/tmp/pr/meta.json` | `Read('/tmp/pr/meta.json')` -- a few KB, read it whole. |
| `/tmp/pr/diff.patch` | `Read('/tmp/pr/diff.patch')` -- can be tens of KB; use `Read` offset/limit to page through it, or `Grep` it for a specific path. |
| `/tmp/pr/checks.json` | `Read('/tmp/pr/checks.json')` then scan the array yourself for entries with `bucket == "fail"` or `bucket == "cancel"`. |
| `/tmp/pr/prev_comments.json` | `Read('/tmp/pr/prev_comments.json')` then scan for entries authored by `rocmlir-pr-reviewer[bot]` with `in_reply_to_id == null` and the marker `<!-- claude-pr-review-marker:v1 -->` in the body. |
| PR-head source files (any path in `meta.files`) | `Read('<path>')` directly from the working directory -- the PR head is checked out there. Use `Grep`/`Glob` to navigate. |

In CI mode, **do not attempt** `Bash`, `jq <something>`, `head -200 file`,
`gh api ...`, `cat`, `find`, `curl`, `wget`, `Write`, or any other shell-style
or write-side tool. None of these are in the allowed list and every attempt
returns a permission denial that counts against `--max-turns`. If you find
yourself reasoning "I should run X to extract Y", stop and reformulate as
"I should `Read` (or `Grep`, or `Glob`) Z". Examples in this file shown
inside code fences are documentation for the *interactive* mode (see the
appendix at the bottom); they are never to be executed in CI.

**Interactive Stage-B mode (local dry-run only).** A maintainer runs the
standalone Claude Code CLI with a broader tool set, e.g.

    claude --allowedTools "Skill,Read,Grep,Glob,Bash(gh *),Bash(jq *)" \
           --skill review-rocmlir-pr <PR-number>

In this mode `/tmp/pr/` may not be populated; pre-fetch it yourself with
the commands in the [appendix](#appendix-interactive-stage-b-only-do-not-execute-in-ci).
Everything in the appendix is **off-limits in CI mode**.

---

## Step 1 -- Load PR context

The workflow has pre-fetched the PR data into `/tmp/pr/`. Note that several
fields are deliberately derived from the LOCALLY checked-out PR HEAD instead
of the live PR API, to defend against a force-push that lands during the
review run -- everything in the table below describes the SAME pinned SHA
the workspace is on (`headRefOid` in `meta.json`):

| File | Contents |
|------|----------|
| `/tmp/pr/meta.json` | PR metadata: `title`, `body`, `author`, `baseRefName`, `headRefName`, plus two locally-injected fields. `headRefOid` is the SHA of the pinned checkout in the workspace (force-push defense), and `files` is an array of `{path}` objects describing the same set of changed paths that `diff.patch` covers. |
| `/tmp/pr/diff.patch` | Unified diff between the merge-base with `baseRefName` and the pinned PR HEAD. Equivalent to GitHub's "Files changed" view for this SHA, but generated locally so it can never disagree with the workspace or with `meta.files` if a force-push lands mid-run. |
| `/tmp/pr/checks.json` | CI status: an array of `{name, state, bucket}` covering both the modern Checks API (e.g. GitHub Actions) and the legacy Commit Statuses API (e.g. some Jenkins integrations), so neither category of red CI is silently missed. `bucket` is one of `pass`, `fail`, `pending`, `skipping`, `cancel`. |
| `/tmp/pr/prev_comments.json` | All existing inline review comments on this PR, in the order the GitHub API returns them. |

The PR head is checked out in the working directory, so you can `Read` source files
directly to see them at their PR-state line numbers.

In CI mode the four files in the table above are *already populated*; the only
thing you need to do is `Read` them. Concretely:

- Start by `Read('/tmp/pr/meta.json')`. Scan the JSON yourself for `title`,
  `headRefOid`, and `files[].path` -- the file is a few KB and `Read` returns
  the whole content. Do not try `jq` -- it is not in your tool set.
- `Read('/tmp/pr/diff.patch')` to see the unified diff. If the file is large
  use `Read` with an `offset`/`limit`, or use `Grep` to jump to a specific
  path within the patch.
- `Read('/tmp/pr/checks.json')` and scan the array for entries whose
  `bucket` is `"fail"` or `"cancel"`. Mention any such entries in your
  summary so the review reflects the PR's actual CI state.
- `Read('/tmp/pr/prev_comments.json')` to discover previous Claude comments
  for the re-review path; see the Output section for the filter rule.

### Special case: changes under `.claude/` or `.github/scripts/`

These two paths are the workflow's "security perimeter": their workspace
contents have been **replaced** with the trusted default-branch versions by an
overlay step that runs before this skill, because their semantics are what
decide whether secrets are protected at runtime (`.claude/skills/` is what
*you* are reading right now; `.github/scripts/sanitize_claude_actions.sh` is
what gates your output before it leaves the runner).

If `diff.patch` shows changes under either of these paths, **the workspace
copies are NOT the PR's proposed versions**. The PR-side versions are at:

| Workspace path (overlaid -> develop's version) | PR-side version (what you should review) |
|---|---|
| `.claude/skills/foo/SKILL.md` | `/tmp/pr-source/.claude/skills/foo/SKILL.md` |
| `.github/scripts/post_claude_review.sh` | `/tmp/pr-source/.github/scripts/post_claude_review.sh` |
| `.github/scripts/sanitize_claude_actions.sh` | `/tmp/pr-source/.github/scripts/sanitize_claude_actions.sh` |

If `/tmp/pr-source/<path>` does not exist while `diff.patch` shows changes
to `<path>`, the PR has deleted that file. Use `Read` on the snapshot path
to see the PR's proposed file content; use the workspace path only if you
explicitly want to see the trusted runtime version for comparison. **Files
NOT under `.claude/` or `.github/scripts/` are unaffected** -- read them
directly from the workspace as usual.

This special case only applies on the workflow_dispatch path; PRs that touch
the perimeter under the label-trigger path are blocked by Layer 3 of the
workflow and never reach this skill.

Identify the changed `.cpp`, `.h`, `.td`, `.mlir`, `.py`, `CMakeLists.txt`, and `.cmake`
files from `meta.json`. `Read` the ones with non-trivial diffs in full.

> Interactive Stage-B (local dry-run only): if `/tmp/pr/` is not already populated
> for you, the pre-fetch commands are in the [appendix at the bottom of this
> file](#appendix-interactive-stage-b-only-do-not-execute-in-ci). **Do not run
> them in CI** -- in CI the files are already there and your tool set does not
> include `Bash`.

---

## Step 2 -- CRITICAL SCOPE RULE

Only flag issues that exist in the PR diff itself -- lines added or modified by this PR.
Do NOT flag pre-existing code that the PR did not touch, even if that code is in the
same files. If a pre-existing problem is worth noting, mention it briefly in a
`Pre-existing issues (out of scope)` section in the summary -- never as an inline
finding against this PR.

---

## Step 3 -- Apply the rocMLIR review checklist

Categorize each finding as **Critical**, **Major**, or **Minor**. Cite the exact
`file:line` from the PR head. Each finding must be a concrete, actionable issue with a
proposed fix.

### Critical (blocks merge)

- Unreleased hardware codenames, unannounced chip IDs, or NDA features in code,
  comments, commits, or docs
- C++ exceptions (`throw`, `try`/`catch`); use `LogicalResult` / `emitOpError` /
  `signalPassFailure` instead
- RTTI (`dynamic_cast`, `typeid`); use LLVM's `isa`/`cast`/`dyn_cast`
- Magic sentinel values (`-1`, `nullptr`) to signal failure; use `FailureOr<>` instead
- `#include <iostream>`; use LLVM's `raw_ostream`
- `using namespace std` at file scope or in headers
- Static constructors/destructors (global objects with non-trivial ctors/dtors)
- Committed temp/generated files: build artifacts, `*.pyc`, editor swap files, secrets,
  profiler output, tuning DBs that don't belong in the repo
- Breaking IR or C-API changes without documentation or a coordinated MIGraphX update

### Major

- DRY/YAGNI/KISS violations: redundant code, dead code, unnecessarily complex algorithms,
  opportunities to use existing upstream LLVM/MLIR utilities instead of custom code
- Raw `new`/`delete`; use MLIR allocation utilities, `std::unique_ptr`, or arena
  ownership
- Inheritance where composition would do; CRTP only where MLIR/LLVM requires it
- `std::string`/`std::vector` for non-owning parameters where `StringRef`/`ArrayRef`/
  `MutableArrayRef` would suffice
- `std::vector` for small local collections where `SmallVector` is preferred
- `std::map`/`std::unordered_map` where `llvm::DenseMap` is preferred
- Missing `assert` with descriptive message on non-trivial preconditions; use
  `llvm_unreachable` for impossible paths (not `assert(false)`)
- C-style casts; use `static_cast`/`const_cast`
- Visibility leaks: file-local helpers without `static` or anonymous namespace
- `default:` label in a switch over an enum that already covers every case (defeats
  `-Wswitch`)
- `std::sort` instead of `llvm::sort` -- LLVM coding standard. `llvm::sort`
  wraps `std::sort` and, under `EXPENSIVE_CHECKS` builds, deterministically
  shuffles the input first to surface order-dependent bugs that would
  otherwise hide behind a libc++/libstdc++ implementation that happens to
  preserve input order. (Note: neither call is *stable*; if equal elements
  must keep their relative order, the fix is `llvm::stable_sort`, not
  `llvm::sort`. Don't suggest `llvm::sort` as a "stability" fix.)
- Naming: classes not `CamelCase`, functions/vars not `camelBack`
- New op without `hasVerifier = 1` and a `verify()` implementation
- New pass or op without positive E2E coverage and both positive and negative Lit tests
  with FileCheck
- New optimization without a FileCheck test asserting the expected IR is produced
- `LogicalResult` returned but ignored (not checked with `failed(...)`)
- `librockcompiler_deps.cmake` not updated when dependencies change
- License header missing or wrong year on a new `.cpp`/`.h`/`.py` file (SPDX
  `Apache-2.0 WITH LLVM-exception`)
- `external/` changes mixed into the same commit as rocMLIR changes (must be separate,
  prefixed `[EXTERNAL]`)
- `TODO` without an issue reference (`TODO(#issue-number)`)
- Architecture coverage: a new op/pass that should work on multiple GPU archs
  (gfx90a, gfx942, gfx950) is implemented for only one
- Data type coverage: an op that should support multiple dtypes
  (f16/bf16/f32/f8/i8/i4) silently falls through for unhandled dtypes instead of
  returning `emitOpError`
- Fusion-related changes that lack tests in `mlir/test/fusion/` or
  `mlir/test/fusion/pr-e2e/`
- Custom CMake targets that bypass `add_rocmlir_dialect_library` /
  `add_rocmlir_conversion_library` / `add_rocmlir_tool` / `add_rocmlir_unittest`
- Downstream MIGraphX impact: changes to public IR, C API, or `librockcompiler` that
  need coordinated updates and aren't called out in the PR description

### Minor

- Include order wrong: should be main module header, then local/private, then MLIR/LLVM,
  then stdlib (each group sorted lexicographically)
- Header lacks self-contained guards
- Comments not English prose with proper capitalization; missing `///` Doxygen on public
  APIs
- Missing early returns; `else` after `return`
- Postincrement (`i++`) where preincrement (`++i`) would do
- `for (auto it = c.begin(); it != c.end(); ++it)` re-evaluating `end()`; prefer
  range-based for
- Braces around single-statement bodies (omit them); missing braces around
  multi-statement bodies
- `auto` where the type isn't obvious; missing `auto &` / `auto *` causing copies
- `inline` on a function defined inside the class body (already implicit)
- Spaces before parentheses in function calls (allowed only in control flow)
- File missing trailing newline; trailing whitespace
- `LLVM_DEBUG` block missing `#define DEBUG_TYPE "rock-..."` at the top of the file
- Lit test missing `// RUN:` line, `-verify-diagnostics`, or `FileCheck` prefix coverage
- New `.toml` E2E config not registered in `mlir/test/e2e/CMakeLists.txt`

### License-header reference (verify on every new file)

C++/header files (`.cpp`, `.h`):

```
//===- FileName.cpp - Brief description ----------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
```

Python files (`.py`):

```
# Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

---

## Step 4 -- Output

Return a single JSON object with this exact shape AS YOUR FINAL RESPONSE. Do not write
it to a file -- the workflow uses claude-code-action's `--json-schema` flag to validate
your final message and capture it as the action's `structured_output`. Findings without
a concrete `path` and `line` from the diff MUST be dropped (do not coerce them into the
summary).

```json
{
  "summary": "Reviewed N files. Posted M inline comments (C critical, J major, K minor). Verdict: APPROVE | REQUEST_CHANGES | COMMENT.",
  "inline_comments": [
    {
      "path": "mlir/lib/Dialect/Rock/Transforms/Foo.cpp",
      "line": 142,
      "side": "RIGHT",
      "severity": "Major",
      "body": "`std::vector<int64_t>` here is preferred as `SmallVector<int64_t, 4>` per LLVM coding standards. (Will also need `#include \"llvm/ADT/SmallVector.h\"`.)",
      "suggestion": "  SmallVector<int64_t, 4> indices;"
    }
  ],
  "thread_updates": []
}
```

Field rules:

- `summary` -- 3-5 lines: scope of change, total counts by severity, overall verdict.
  Do NOT restate individual findings; they go in `inline_comments`.
- `inline_comments[].path` -- repo-relative file path, must appear in the PR diff.
- `inline_comments[].line` -- exact line number in the PR head (`headRefOid` in
  `/tmp/pr/meta.json`); do NOT use diff-relative line numbers.
- `inline_comments[].side` -- MUST be `"RIGHT"`. We do not support `"LEFT"`
  comments (which would target deleted lines on the *base* file). GitHub's
  PR-review-comment API uses *base-file* line numbers for `LEFT` comments
  but *head-file* line numbers for `RIGHT` comments; this skill's "use the
  PR head line number" rule (and the diff context loaded into the prompt)
  is head-relative only, so a `LEFT` comment would either anchor to the
  wrong place or get a 422 from GitHub. If a deleted line is the right
  place to discuss something, comment instead on the nearest surviving
  RIGHT-side context line and reference the deletion in the body.
- `inline_comments[].severity` -- one of `Critical`, `Major`, `Minor`.
- `inline_comments[].body` -- a single short paragraph: what the issue is, why it
  matters, and a concrete proposed fix (in prose -- the verbatim replacement, if
  any, goes in `suggestion`).
- `inline_comments[].suggestion` -- **OPTIONAL.** Verbatim replacement text for
  the single line at `line`. The post script wraps this in a fenced
  ` ```suggestion ` block at the bottom of the comment, which renders as a
  "Commit suggestion" button in the GitHub UI -- one click and the developer
  has applied the fix. Strict rules:
  1. **Single line only.** This pipeline does not support multi-line ranges
     today (`start_line`/`start_side` are not in the schema). The suggestion
     replaces exactly one line: the line at `line` on the side `side`. If the
     fix needs more than one line, omit the `suggestion` and describe the fix
     in `body` instead. The schema's `pattern` AND the workflow's sanitizer
     both reject any `suggestion` containing a line feed, carriage return,
     or the literal sequence of three backticks (which would close the
     wrapping fence early and break the rendered comment) -- a violating
     payload fails the workflow rather than producing a broken comment.
  2. **Verbatim, with correct indentation.** GitHub commits the suggestion
     bytes-for-bytes into the file. Match the file's existing indentation
     (tabs vs spaces, depth) and trailing-whitespace conventions exactly. Do
     NOT add a trailing newline -- the surrounding file lines already provide
     it.
  3. **Self-contained.** The suggestion must fully address the finding without
     requiring matching edits elsewhere (e.g. don't suggest a `SmallVector`
     replacement if the user also has to add a `#include` somewhere -- mention
     the include in `body` and let the developer write it; or skip the
     suggestion entirely).
  4. **High confidence only.** A wrong suggestion is worse than no suggestion:
     the developer might click-commit it and ship a bug. If you have any doubt
     about the surrounding context, indentation, or whether the replacement
     compiles, omit the field.

  Good `suggestion` cases:
    - `std::vector<X>` -> `SmallVector<X, 4>` on a single declaration line.
    - `std::sort(...)` -> `llvm::sort(...)`.
    - `i++` -> `++i` in a `for` header.
    - missing `auto &` / `auto *` on a single line.
    - `(int)x` C-style cast -> `static_cast<int>(x)`.
    - missing `static` / `inline` modifier on a single declaration.

  Skip `suggestion` for:
    - findings that need a new `#include`, a new helper function, or any edit
      on a different line.
    - findings that need the developer to choose between options.

  **Never embed a ` ```suggestion ` fence in `summary`, `inline_comments[].body`,
  or `thread_updates[].body`.** GitHub renders that fence as a one-click
  "Commit suggestion" UI, and a fence in a free-form prose field bypasses the
  single-line / verbatim / high-confidence contract above (and would also bypass
  the workflow sanitizer's checks on the structured `suggestion` field). The
  sanitizer rejects any payload whose body fields contain a ` ```suggestion `
  fence and fails the workflow. If you want to suggest a code change, put the
  replacement bytes in the structured `inline_comments[].suggestion` field;
  to *show* code in prose without offering it as a commit, use a different
  language tag (e.g. ` ```cpp `).
    - any case where you have not read enough context to be sure of the exact
      replacement bytes.
- `thread_updates` -- empty `[]` for an initial review. Populated by `update-pr-review`
  on re-review runs.

If the PR is genuinely good, return an empty `inline_comments: []` and an APPROVE
summary. The "Resolved" path in `update-pr-review` only works if reviews are honest.

If `update-pr-review` will run after this skill (the workflow detects this when
`/tmp/pr/prev_comments.json` contains prior root comments where `user.login` is
`rocmlir-pr-reviewer[bot]` AND the body contains the literal substring
`<!-- claude-pr-review-marker:v1 -->`), pass this output to that skill as input --
it will produce the final JSON with `thread_updates` populated and only-genuinely-new
entries in `inline_comments`. `rocmlir-pr-reviewer[bot]` is the bot identity of
the rocMLIR-PR-Reviewer GitHub App, which is the only identity this pipeline posts
under. Previous reviews are NOT authored as `claude[bot]` (we do not use the
Anthropic OIDC exchange) and NOT as `github-actions[bot]` (that was the identity
used in earlier iterations of this pipeline; the App migration moved us to a
unique identity). The marker check is belt-and-braces and also lets the update
skill distinguish our own marker-tagged replies from genuine human replies in the
same thread.

---

## Rules

- Reference issues by `file:line` from the PR head, not diff-relative line numbers.
- Each finding must include a concrete proposed fix in the `body`.
- Only flag actual issues. Do not flag correct behavior; do not flag style preferences
  not codified above; do not generate findings to hit a quota.
- Do NOT include any environment variable name or value, secret, or HTTP header in
  any output field. URLs are allowed ONLY to `github.com` / `*.github.com` /
  `*.githubusercontent.com` (the sanitizer's host allow-list); reference any other
  source by name and let the human follow up. The workflow's sanitizer fails the
  build if it sees patterns matching common secret formats, LLM-Gateway env-var
  names, the literal `<!-- claude-pr-review-` marker prefix, URLs to disallowed
  hosts (including userinfo-bypass forms like `https://github.com@evil/x`),
  Markdown link destinations using non-http(s) schemes (`mailto:`, `ftp:`,
  `javascript:`, `data:`, `file:`, `vbscript:`), protocol-relative destinations
  (`//evil/x`) to disallowed hosts, **OR** the same shapes inside raw HTML
  attributes (`<a href="//evil/x">`, `<a href="mailto:...">`, `<img src="//evil/x">`)
  and `<a href="https://evil/x">` -- the sanitizer extracts and validates
  `href=` and `src=` destinations the same way it validates Markdown destinations.
  HTML-entity-encoded variants are ALSO rejected (the sanitizer entity-decodes
  before running every URL check, so `https&#x3A;//evil/x`,
  `[click](&#x2F;&#x2F;evil/x)`, and `<a href="&#x2F;&#x2F;evil/x">` are all
  caught). Bracketed-IP-literal hosts (RFC 3986 IP-literal: `[` IPv6 or
  IPvFuture `]`) are categorically rejected in every URL form -- bare URL,
  Markdown destination, or HTML href/src -- because github.com is never reached
  via a raw IP literal and the host allow-list cannot classify an IP.
  Percent-encoded host components are likewise categorically rejected: per the
  WHATWG URL spec the host is percent-decoded before resolution, so
  `https://%65vil.example/x` renders as `evil.example/x` and
  `https://github.com%2eevil.example/x` becomes a subdomain of `evil.example`;
  any `%XX` in the URL authority is rejected (`%XX` in the path or query is
  fine). ASCII tab / LF / CR inside URL strings or HTML attribute values are
  stripped by the WHATWG URL parser before resolution (URL Standard §4.4), so
  `<a href="//evil\nhost.com/x">` resolves as `https://evilhost.com/x`; the
  sanitizer strips the same three bytes from its URL-extraction view, and any
  resulting host that fails the allow-list is rejected. Keep this list in sync
  with the prompt's Hard constraints block in
  `.github/workflows/claude_auto_review.yml`.

---

## Appendix: Interactive Stage B only (DO NOT execute in CI)

> **Stop and reread the "Tool budget" section at the top of this file if
> you are about to use anything below from inside the
> `claude_auto_review.yml` workflow.** In CI mode the allowed tool set is
> `Skill,Read,Grep,Glob` -- `Bash`, `jq`, `gh`, `head`, `cat`, `Write`
> are all denied and every attempt counts against `--max-turns`. The
> commands below are for a maintainer running the Claude Code CLI
> locally to populate `/tmp/pr/` before invoking this skill.

The workflow uses local-`git` derivations for `diff.patch` and `meta.files`
(force-push race defense); for an interactive run those races don't matter,
so plain `gh pr diff` / `gh pr view --json …,files` is fine and produces a
shape compatible with the table at the top of this file:

```bash
mkdir -p /tmp/pr
gh pr view "$ARGUMENTS" --json title,body,author,baseRefName,headRefName,headRefOid,files \
  > /tmp/pr/meta.json
gh pr diff "$ARGUMENTS" > /tmp/pr/diff.patch
# Mirror the workflow's REST-API path so local dry-runs surface the same
# {name, state, bucket} shape and survive on any gh version. `gh pr checks
# --json` was only added in gh v2.36 and has rotated its field set since
# (the `conclusion` field went away in favour of `bucket`); the workflow's
# self-hosted runner pool serves heterogeneous images, and at least one
# pod ships a gh without `--json` at all, where `gh pr checks --json ... ||
# echo '[]'` silently lost all CI signal. The two endpoints below are
# disjoint -- /check-runs is the modern Checks API (GitHub Actions etc.),
# /status is the legacy Commit Statuses API (some Jenkins integrations);
# either one alone misses the other half of red CI.
# Both endpoints are paginated -- /check-runs paginates .check_runs at
# 30/page by default, /status paginates .statuses the same way. A PR
# with >30 entries on either side would otherwise silently drop the
# overflow. `state` mirrors gh's old `pr checks --json state` semantics
# (conclusion-when-completed, status-while-pending; legacy .state for
# /status), so the field still distinguishes SUCCESS vs FAILURE vs
# TIMED_OUT etc. rather than always reading COMPLETED.
SHA=$(jq -r .headRefOid /tmp/pr/meta.json)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
gh api --paginate "repos/$REPO/commits/$SHA/check-runs" \
  | jq -s '[.[] | .check_runs[]?] | map({
        name: .name,
        state: (
          (if (.status != "completed") then .status
           else (.conclusion // "unknown")
           end) | ascii_upcase
        ),
        bucket: (
          if (.status != "completed") then "pending"
          elif (.conclusion == "success" or .conclusion == "neutral") then "pass"
          elif (.conclusion == "skipped") then "skipping"
          elif (.conclusion == "cancelled") then "cancel"
          else "fail"
          end)
      })' > /tmp/pr/check_runs.json
gh api --paginate "repos/$REPO/commits/$SHA/status" \
  | jq -s '[.[] | .statuses[]?] | map({
        name: .context,
        state: ((.state // "unknown") | ascii_upcase),
        bucket: (
          if (.state == "pending") then "pending"
          elif (.state == "success") then "pass"
          else "fail"
          end)
      })' > /tmp/pr/statuses.json
jq -s 'add' /tmp/pr/check_runs.json /tmp/pr/statuses.json > /tmp/pr/checks.json
rm /tmp/pr/check_runs.json /tmp/pr/statuses.json
gh api --paginate "repos/$REPO/pulls/$ARGUMENTS/comments" | jq -s 'add // []' \
  > /tmp/pr/prev_comments.json
```
