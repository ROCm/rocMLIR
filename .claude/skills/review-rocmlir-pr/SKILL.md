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

### Special case: changes under `.claude/`, `.github/scripts/`, or `docs/CODING_STANDARDS.md`

These three paths are the workflow's [trust perimeter](../../../.github/workflows/CLAUDE_AUTO_REVIEW.md#17-glossary).
Their workspace contents have been **replaced** with trusted default-branch
versions by an overlay step that runs before this skill. `.claude/skills/`
is what *you* are reading right now, `.github/scripts/sanitize_claude_actions.sh`
is what gates your output before it leaves the runner, and
`docs/CODING_STANDARDS.md` is the **single source of truth** for the
Critical / Major / Minor tier categorization you apply in Step 3.

If `diff.patch` shows changes under any of these paths, **the workspace
copies are NOT the PR's proposed versions**. The PR-side versions are at:

| Workspace path (overlaid -> develop's version) | PR-side version (what you should review) |
|---|---|
| `.claude/skills/foo/SKILL.md` | `/tmp/pr-source/.claude/skills/foo/SKILL.md` |
| `.github/scripts/post_claude_review.sh` | `/tmp/pr-source/.github/scripts/post_claude_review.sh` |
| `.github/scripts/sanitize_claude_actions.sh` | `/tmp/pr-source/.github/scripts/sanitize_claude_actions.sh` |
| `docs/CODING_STANDARDS.md` | `/tmp/pr-source/docs/CODING_STANDARDS.md` |

If `/tmp/pr-source/<path>` does not exist while `diff.patch` shows changes
to `<path>`, the PR has deleted that file. Use `Read` on the snapshot path
to see the PR's proposed file content; use the workspace path only if you
explicitly want to see the trusted runtime version for comparison. **Files
NOT under those three paths are unaffected** -- read them directly from
the workspace as usual.

This special case only applies on the workflow_dispatch path; PRs that touch
`.claude/` or `.github/scripts/` under the label-trigger path are blocked by
Layer 3 of the workflow and never reach this skill. `docs/CODING_STANDARDS.md`
is NOT in Layer 3's perimeter regex (it's a docs file, not security-sensitive),
so a label-trigger PR may legitimately diff it -- still review the PR-side
version at `/tmp/pr-source/docs/CODING_STANDARDS.md`; the workspace copy is
the trusted version your tier categorization actually used.

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

## Step 3 -- Apply the coding-standards checklist

The coding standards reach you through `docs/CODING_STANDARDS.md` --
the **single source of truth**. The workflow loads it for you in two
ways, both sourced from the same default-branch ref:

1. The `snapshot_standards` workflow step reads the (overlaid, trusted)
   file at runtime and substitutes its content into the prompt heredoc
   between the `<BEGIN/END docs/CODING_STANDARDS.md>` markers (the
   "## Coding standards (canonical reference)" section above this skill
   in your conversation). You already have it in context -- no Read
   needed.
2. The same file is overlaid into the workspace (see the Special case
   section above), so `Read('docs/CODING_STANDARDS.md')` returns the
   same content if you want to confirm.

Both channels read from the same trusted file at the same workflow run,
so the **Critical / Major / Minor** tiers and the license-header
template you see in either view are authoritative. The two views are
content-equivalent but not literally byte-identical: the injected
BEGIN/END block intentionally omits the H1 + blank-line prelude (the
prompt's "## Coding standards" heading replaces it) and is
LF-normalized, while `Read('docs/CODING_STANDARDS.md')` returns the
full file as-is. Use either. Categorize each finding against those
tiers.

Each finding must:

- Cite the exact `file:line` from the PR head (not diff-relative line
  numbers).
- Include a concrete, actionable proposed fix in the `body`.
- Reference the specific bullet in `docs/CODING_STANDARDS.md` that
  applies, so the author can look up the rationale (for example:
  "Critical: `using namespace std` at file scope" or "Major:
  `std::vector` for small local collections").

---

## Step 4 -- Output

Return a single JSON object with this exact shape AS YOUR FINAL RESPONSE. Do not write
it to a file -- the workflow uses claude-code-action's `--json-schema` flag to validate
your final message and capture it as the action's `structured_output`. Findings without
a concrete `path` and `line` from the diff MUST be dropped (do not coerce them into the
summary).

```json
{
  "verdict": "APPROVE",
  "summary": "## Scope\n[1-2 sentences on what the PR does]\n\n## Findings\nNo blocking issues found.\n\n## Notes\n[optional observations]",
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

- `verdict` -- one of `APPROVE`, `REQUEST_CHANGES`, `COMMENT`. Pick what
  a human reviewer would say about the PR's current state: any Critical
  finding -> `REQUEST_CHANGES`; zero findings -> `APPROVE`; otherwise
  `COMMENT` unless the findings materially affect correctness/security
  (then `REQUEST_CHANGES`). Justify the choice in `summary`. On a
  re-review the verdict reflects the PR's CURRENT state after the
  author's fixes -- a previously-REQUEST_CHANGES PR with everything
  resolved gets an `APPROVE`.

  The post job submits **every verdict** as `gh pr review --comment`
  (the rendered body header shows your verdict with a "submitted as
  COMMENT" annotation). **Do NOT change your verdict because of this** --
  emit the honest verdict; full rationale in `CLAUDE_AUTO_REVIEW.md` §13.

- `summary` -- Markdown body of the formal review. The post job prepends a
  one-line header (verdict + finding counts); EVERYTHING else is this field.
  Use `##` section headings; typical layout:

      ## Scope        -- 1-2 sentences: what the PR does / files touched
      ## Findings     -- one-sentence each, citing `path:line` (full bodies
                         go in `inline_comments[]`). Zero findings: a single
                         confirming line, e.g. "No blocking issues found."
      ## Notes        -- (optional) spot-checks, out-of-scope observations,
                         follow-up suggestions for a separate PR
      ## CI status    -- (optional) non-self checks in `/tmp/pr/checks.json`
                         with `bucket == "fail"` or `"cancel"`; the
                         auto-review pipeline's own check is expected to be
                         in-progress / failed and is NOT a CI failure

  Target ~2000 chars (sanitizer cap is 8 KiB). Do NOT restate individual
  findings (those go in `inline_comments[]`). Do NOT print "Verdict: ..."
  in the body -- the post job adds that header line.

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
  one-click "Commit suggestion" button in the GitHub UI. Strict rules:
  1. **Single line only.** Multi-line ranges are not supported. The schema and
     sanitizer both reject any `suggestion` containing `\n`, `\r`, or
     ` ``` ` (which would close the wrapping fence early).
  2. **Verbatim, with correct indentation.** GitHub commits the bytes as-is.
     Match the file's existing indentation (tabs vs spaces, depth) exactly.
     No trailing newline -- surrounding file lines already provide it.
  3. **Self-contained.** Fully address the finding without requiring matching
     edits elsewhere (e.g. omit if a `#include` is also needed).
  4. **High confidence only.** A wrong suggestion is worse than no suggestion;
     omit if you have any doubt about context, indentation, or whether the
     replacement compiles.

  Good cases: `std::vector<X>` → `SmallVector<X, 4>`; `std::sort` → `llvm::sort`;
  `i++` → `++i` in a `for` header; missing `auto &` / `auto *`; C-style cast →
  `static_cast<T>(...)`.

  Skip `suggestion` for: anything needing a new `#include` or a different-line
  edit; anything where the developer must choose between options; anything
  where you have not read enough context to be sure of the exact bytes.

  **Never embed a ` ```suggestion ` fence in `summary`, `inline_comments[].body`,
  or `thread_updates[].body`.** GitHub renders the fence as a one-click commit
  UI regardless of which field it appears in, bypassing the single-line /
  verbatim / high-confidence contract above. The sanitizer rejects any
  prose-body fence and fails the workflow. To *show* code in prose without
  offering a commit, use a different language tag (e.g. ` ```cpp `).
- `thread_updates` -- empty `[]` for an initial review. Populated by `update-pr-review`
  on re-review runs.

If the PR is genuinely good, return an empty `inline_comments: []`, an
`APPROVE` verdict, and a short summary confirming you found no blocking
issues. The "Resolved" path in `update-pr-review` only works if reviews are
honest.

If `update-pr-review` will run after this skill (the workflow detects this when
`/tmp/pr/prev_comments.json` contains prior root comments where `user.login` is
`rocmlir-pr-reviewer[bot]` AND the body contains the literal substring
`<!-- claude-pr-review-marker:v1 -->`), pass this output to that skill as input --
it will produce the final JSON with `thread_updates` populated and only-genuinely-new
entries in `inline_comments`. `rocmlir-pr-reviewer[bot]` is the bot identity of
the rocMLIR-PR-Reviewer GitHub App, which is the only identity this pipeline posts
under. Previous reviews are NOT authored as `claude[bot]` (we do not use the
Anthropic OIDC exchange) and NOT as `github-actions[bot]` (the default Actions
identity is not unique to this pipeline). The marker check is belt-and-braces and
also lets the update skill distinguish our own marker-tagged replies from genuine
human replies in the same thread.

---

## Rules

- Reference issues by `file:line` from the PR head, not diff-relative line numbers.
- Each finding must include a concrete proposed fix in the `body`.
- Only flag actual issues. Do not flag correct behavior; do not flag style preferences
  not codified above; do not generate findings to hit a quota.
- Do NOT include any environment variable name or value, secret, or HTTP header
  in any output field. URLs are allowed ONLY to `github.com` / `*.github.com` /
  `*.githubusercontent.com`; reference any other source by name and let the
  human follow up. The complete enumeration of rejected URL forms (userinfo
  bypass, protocol-relative, non-http(s) schemes, raw HTML attributes,
  entity-encoded variants, bracketed-IP-literal hosts, percent-encoded
  authorities, TAB/LF/CR-split hosts) lives in the workflow prompt's "Hard
  constraints" block. The sanitizer enforces all of them; full design in
  `CLAUDE_AUTO_REVIEW.md` §10.

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
