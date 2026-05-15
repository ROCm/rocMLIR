---
name: review-rocmlir-pr
description: Review a rocMLIR pull request with deep expertise in MLIR/LLVM coding standards, the Rock dialect, MIGraphX integration, kernel codegen for AMD GPUs, lit/E2E testing, and the rocMLIR CMake build. Use when asked to review a rocMLIR PR or check a rocMLIR change. Read-only; never posts comments.
argument-hint: [PR-number]
agent: general-purpose
allowed-tools: Bash(gh pr view *), Bash(gh pr diff *), Bash(gh pr checks *), Bash(gh api repos/*), Bash(git log *), Bash(git show *), Bash(git diff *), Bash(git blame *), Bash(git fetch *), Bash(head *), Bash(grep *), Bash(jq *), Read, Grep, Glob
---

# rocMLIR PR Review

## IMPORTANT: Do NOT post comments

This skill is **read-only**. Do NOT post any comments, reviews, or reactions to GitHub.
Do NOT use `gh pr comment`, `gh pr review`, `gh api` with `POST`/`PUT`/`PATCH`/`DELETE`,
or any other write operation. Only return your findings as text output -- the caller
(workflow) is responsible for posting them as inline comments via the
`mcp__github_inline_comment__create_inline_comment` MCP tool.

---

## Step 1 -- Fetch PR context

`$ARGUMENTS` is the PR number. Run these commands to gather context:

```bash
gh pr view $ARGUMENTS --json title,body,author,baseRefName,headRefName,headRefOid,files,statusCheckRollup
gh pr diff $ARGUMENTS --name-only
gh pr diff $ARGUMENTS
gh pr checks $ARGUMENTS 2>/dev/null || echo "(no checks yet)"
gh pr view $ARGUMENTS --comments 2>/dev/null || echo "(no comments yet)"
```

The PR branch is already checked out by the workflow at `refs/pull/$ARGUMENTS/head`, so you
can `Read` files at their PR-state line numbers directly. Use `git show HEAD:<file>` if you
need to disambiguate.

Identify the changed `.cpp`, `.h`, `.td`, `.mlir`, `.py`, `CMakeLists.txt`, and `.cmake`
files. Read the ones with non-trivial diffs in full.

---

## Step 2 -- CRITICAL SCOPE RULE

Only flag issues that exist in the PR diff itself -- lines added or modified by this PR.
Do NOT flag pre-existing code that the PR did not touch, even if that code is in the same
files. If a pre-existing problem is worth noting, mention it briefly in a separate
"Pre-existing issues (out of scope)" section at the end of your output -- never as an
inline finding against this PR.

---

## Step 3 -- Apply the rocMLIR review checklist

Categorize each finding as **Critical**, **Major**, or **Minor**. Cite the exact
`file:line` from the PR head. Each finding must be a concrete, actionable issue with a
proposed fix.

### Critical (blocks merge)

- Unreleased hardware codenames, unannounced chip IDs, or NDA features in code, comments,
  commits, or docs
- C++ exceptions (`throw`, `try`/`catch`); use `LogicalResult` / `emitOpError` /
  `signalPassFailure` instead
- RTTI (`dynamic_cast`, `typeid`); use LLVM's `isa`/`cast`/`dyn_cast`
- Magic sentinel values (`-1`, `nullptr`) to signal failure; use `FailureOr<>` instead
- `#include <iostream>`; use LLVM's `raw_ostream`
- `using namespace std` at file scope or in headers; always use explicit `std::`
- Static constructors/destructors (global objects with non-trivial ctors/dtors)
- Committed temp/generated files: build artifacts, `*.pyc`, editor swap files, secrets,
  profiler output, tuning DBs that don't belong in the repo
- Breaking IR or C-API changes without documentation or a coordinated MIGraphX update

### Major

- DRY/YAGNI/KISS violations: redundant code, dead code, unnecessarily complex algorithms,
  opportunities to use existing upstream LLVM/MLIR utilities instead of custom code
- Raw `new`/`delete`; use MLIR allocation utilities, `std::unique_ptr`, or arena ownership
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
- `std::sort` instead of `llvm::sort` (non-deterministic for equal elements)
- Naming: classes not `CamelCase`, functions/vars not `camelBack`
- New op without `hasVerifier = 1` and a `verify()` implementation
- New pass or op without positive E2E coverage and both positive and negative Lit tests
  with FileCheck
- New optimization without a FileCheck test asserting the expected IR is produced
- `LogicalResult` returned but ignored (not checked with `failed(...)` or
  `LLVM_ATTRIBUTE_USED`)
- `librockcompiler_deps.cmake` not updated when dependencies change
- License header missing or wrong year on a new `.cpp`/`.h`/`.py` file (SPDX
  `Apache-2.0 WITH LLVM-exception`)
- `external/` changes mixed into the same commit as rocMLIR changes (must be separate,
  prefixed `[EXTERNAL]`)
- `TODO` without an issue reference (`TODO(#issue-number)`)
- Architecture coverage: a new op/pass that should work on multiple GPU archs (gfx90a,
  gfx942, gfx950) is implemented for only one
- Data type coverage: an op that should support multiple dtypes (f16/bf16/f32/f8/i8/i4)
  silently falls through for unhandled dtypes instead of returning `emitOpError`
- Fusion-related changes that lack tests in `mlir/test/fusion/` or
  `mlir/test/fusion/pr-e2e/`
- Custom CMake targets that bypass `add_rocmlir_dialect_library` /
  `add_rocmlir_conversion_library` / `add_rocmlir_tool` / `add_rocmlir_unittest`
- Downstream MIGraphX impact: changes to public IR, C API, or `librockcompiler` that need
  coordinated updates and aren't called out in the PR description

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
- Braces around single-statement bodies (omit them); missing braces around multi-statement
  bodies
- `auto` where the type isn't obvious; missing `auto &` / `auto *` causing copies
- `inline` on a function defined inside the class body (already implicit)
- Spaces before parentheses in function calls (allowed only in control flow)
- File missing trailing newline; trailing whitespace
- `LLVM_DEBUG` block missing `#define DEBUG_TYPE "rock-..."` at the top of the file
- Lit test missing `// RUN:` line, `-verify-diagnostics`, or `FileCheck` prefix coverage
  (`FILECHECK_OPTS="-enable-var-scope --allow-unused-prefixes=false"` enforces all
  prefixes are used)
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

Year must match the current year on net-new files (do not copy an old year).

---

## Step 4 -- Inline-comment output contract

Every finding you return MUST be a structured record with these fields. If a finding does
not have a concrete `path` and `line` from the diff, **drop it** -- do not coerce it into
the summary text.

```
- path: <repo-relative file path from the diff>
  line: <exact line number in the PR head>
  side: RIGHT      # RIGHT for added/modified lines, LEFT for deleted lines
  severity: Critical | Major | Minor
  body: |
    <one short paragraph describing the issue>

    Suggested fix: <concrete patch or replacement code>
```

For multi-line concerns (e.g. "this whole if-block could be simplified"), anchor to the
most representative line and describe the broader range in the body.

The workflow's prompt iterates these records and calls
`mcp__github_inline_comment__create_inline_comment(path, line, side, body)` once per
record, producing one threaded inline comment per finding -- exactly like a human reviewer
leaving line-by-line comments in the Files Changed tab.

---

## Step 5 -- Output format

Return your findings in this exact shape:

```markdown
## Summary

<3-5 line summary: scope of change, overall verdict (APPROVE / REQUEST_CHANGES / COMMENT),
total counts by severity. Do NOT restate individual findings here.>

## Findings

- path: mlir/lib/Dialect/Rock/Transforms/Foo.cpp
  line: 142
  side: RIGHT
  severity: Major
  body: |
    `std::vector<int64_t>` here is preferred as `SmallVector<int64_t, 4>` per LLVM
    coding standards -- the typical size of this collection is well under 16, so the
    inline storage avoids a heap allocation.

    Suggested fix: replace `std::vector<int64_t>` with `SmallVector<int64_t, 4>` and
    add `#include "llvm/ADT/SmallVector.h"` if not already included.

- path: mlir/test/Dialect/Rock/my-pass.mlir
  line: 1
  side: RIGHT
  severity: Major
  body: |
    This new pass is missing a negative test exercising the verifier failure path.

    Suggested fix: add a second `// RUN: rocmlir-opt --my-pass -split-input-file
    -verify-diagnostics %s | FileCheck %s` block with input that triggers
    `emitOpError`, and a `// expected-error @+1 {{...}}` annotation.

## Pre-existing issues (out of scope)

<Optional. List pre-existing problems noticed but NOT caused by this PR. Brief, no
inline-comment fields. Omit this section entirely if none.>
```

---

## Rules

- Reference issues by `file:line` from the PR head, not diff-relative line numbers.
- Each finding must include a concrete proposed fix.
- Only flag actual issues. Do not flag correct behavior, do not flag style preferences
  not codified above, do not generate findings to hit a quota.
- If the PR is genuinely good, return an empty `## Findings` section and an APPROVE
  summary. The "Resolved" path in `update-pr-review` only works if reviews are honest.
