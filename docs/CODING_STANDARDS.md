# rocMLIR Coding Standards

This is the authoritative coding-standards checklist for rocMLIR.
Reviewers -- human and automated -- categorize findings against the
**Critical / Major / Minor** tiers below; a Critical finding here is one
of the bullets in the "Critical" section, a Major finding is one of the
bullets in "Major", etc.

When you contribute, run `git clang-format --diff origin/develop`
(or `upstream/develop` if you've forked and named the `ROCm/rocMLIR`
remote `upstream`) and self-review your diff against this checklist.
When you review, cite the specific bullet so the author can look up
the rationale.

## References

- [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) -- the
  authoritative style guide; everything below derives from or extends it.
- [MLIR Developer Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/)
  -- the one naming-convention deviation (`camelBack` for variables,
  parameters, and class members instead of LLVM's traditional `Capitalized`)
  lives there.
- `.clang-format` (LLVM base style) and `.clang-tidy` at the repo root --
  machine-enforced subset; the premerge `clang-format` job runs
  `git clang-format --diff origin/develop` (CI's checkout always names
  the `ROCm/rocMLIR` remote `origin`) and fails on any non-empty diff.
- Python helpers follow [`yapf`](../.style.yapf) and
  [`flake8`](../.flake8); format with `yapf -i <files>` and lint with
  `flake8 <files>` before committing.

## Critical (blocks merge)

- Unreleased hardware codenames, unannounced chip IDs, or NDA features in
  code, comments, commits, or docs.
- C++ exceptions (`throw`, `try`/`catch`); use `LogicalResult` /
  `emitOpError` / `signalPassFailure` instead.
- RTTI (`dynamic_cast`, `typeid`); use LLVM's `isa`/`cast`/`dyn_cast`.
- Magic sentinel values (`-1`, `nullptr`) to signal failure; use
  `FailureOr<>` instead.
- `#include <iostream>`; use LLVM's `raw_ostream`.
- `using namespace std` at file scope or in headers.
- Static constructors/destructors (global objects with non-trivial
  ctors/dtors).
- Committed temp/generated files: build artifacts, `*.pyc`, editor swap
  files, secrets, profiler output, tuning DBs that don't belong in the
  repo.
- Breaking IR or C-API changes without documentation or a coordinated
  MIGraphX update.

## Major

- DRY/YAGNI/KISS violations: redundant code, dead code, unnecessarily
  complex algorithms, opportunities to use existing upstream LLVM/MLIR
  utilities instead of custom code.
- Raw `new`/`delete`; use MLIR allocation utilities, `std::unique_ptr`, or
  arena ownership.
- Inheritance where composition would do; CRTP only where MLIR/LLVM
  requires it.
- `std::string`/`std::vector` for non-owning parameters where
  `StringRef`/`ArrayRef`/`MutableArrayRef` would suffice.
- `std::vector` for small local collections where `SmallVector` is
  preferred.
- `std::map`/`std::unordered_map` where `llvm::DenseMap` is preferred.
- Missing `assert` with descriptive message on non-trivial preconditions;
  use `llvm_unreachable` for impossible paths (not `assert(false)`).
- C-style casts; use `static_cast`/`const_cast`.
- Visibility leaks: file-local helpers without `static` or anonymous
  namespace.
- `default:` label in a switch over an enum that already covers every case
  (defeats `-Wswitch`).
- `std::sort` instead of `llvm::sort` -- LLVM coding standard. `llvm::sort`
  wraps `std::sort` and, under `EXPENSIVE_CHECKS` builds, deterministically
  shuffles the input first to surface order-dependent bugs that would
  otherwise hide behind a libc++/libstdc++ implementation that happens to
  preserve input order. (Note: neither call is *stable*; if equal elements
  must keep their relative order, the fix is `llvm::stable_sort`, not
  `llvm::sort`. Don't suggest `llvm::sort` as a "stability" fix.)
- Naming: classes not `CamelCase`, functions/vars not `camelBack`.
- New op without `hasVerifier = 1` and a `verify()` implementation.
- New pass or op without positive E2E coverage and both positive and
  negative Lit tests with FileCheck.
- New optimization without a FileCheck test asserting the expected IR is
  produced.
- `LogicalResult` returned but ignored (not checked with `failed(...)`).
- `librockcompiler_deps.cmake` not updated when dependencies change.
- License header missing or wrong year on a new `.cpp`/`.h`/`.py` file
  (SPDX `Apache-2.0 WITH LLVM-exception`).
- `external/` changes mixed into the same commit as rocMLIR changes (must
  be separate, prefixed `[EXTERNAL]`).
- `TODO` without an issue reference (`TODO(#issue-number)`).
- Architecture coverage: a new op/pass that should work on multiple GPU
  archs (gfx90a, gfx942, gfx950) is implemented for only one.
- Data type coverage: an op that should support multiple dtypes
  (f16/bf16/f32/f8/i8/i4) silently falls through for unhandled dtypes
  instead of returning `emitOpError`.
- Fusion-related changes that lack tests in `mlir/test/fusion/` or
  `mlir/test/fusion/pr-e2e/`.
- Custom CMake targets that bypass `add_rocmlir_dialect_library` /
  `add_rocmlir_conversion_library` / `add_rocmlir_tool` /
  `add_rocmlir_unittest`.
- Downstream MIGraphX impact: changes to public IR, C API, or
  `librockcompiler` that need coordinated updates and aren't called out in
  the PR description.

## Minor

- Include order wrong: should be main module header, then local/private,
  then MLIR/LLVM, then stdlib (each group sorted lexicographically).
- Header lacks self-contained guards.
- Comments not English prose with proper capitalization; missing `///`
  Doxygen on public APIs.
- Missing early returns; `else` after `return`.
- Postincrement (`i++`) where preincrement (`++i`) would do.
- `for (auto it = c.begin(); it != c.end(); ++it)` re-evaluating `end()`;
  prefer range-based for.
- Braces around single-statement bodies (omit them); missing braces around
  multi-statement bodies.
- `auto` where the type isn't obvious; missing `auto &` / `auto *` causing
  copies.
- `inline` on a function defined inside the class body (already implicit).
- Spaces before parentheses in function calls (allowed only in control
  flow).
- File missing trailing newline; trailing whitespace.
- `LLVM_DEBUG` block missing `#define DEBUG_TYPE "rock-..."` at the top of
  the file.
- Lit test missing `// RUN:` line, `-verify-diagnostics`, or `FileCheck`
  prefix coverage.
- New `.toml` E2E config not registered in `mlir/test/e2e/CMakeLists.txt`.

## License-header reference (verify on every new file)

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

Checklist on every new file:

- Header is present.
- Copyright year matches the current year (not copied from older files).
- SPDX identifier is exactly `Apache-2.0 WITH LLVM-exception`.
