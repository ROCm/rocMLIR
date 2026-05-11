# Contributing to rocMLIR

Thanks for your interest in contributing.

## Reporting Issues

Use [GitHub Issues](../../issues) to report bugs or request features. Include a clear description, reproduction steps, and your environment: OS, ROCm version, GPU architecture (e.g. `gfx942`, `gfx950`, `gfx1100`).

## Pull Request Workflow

1. Get the source and create a branch from `develop`. External contributors should [fork the repository](../../fork) first; AMD contributors with write access can clone the upstream directly:
   ```bash
   # Replace <owner> with your fork's owner, or `ROCm` if you have write access.
   git clone https://github.com/<owner>/rocMLIR.git
   cd rocMLIR
   # AMD contributors: name the branch `users/<username>/<description>` or
   # `<jira-id>-<description>` (e.g. AIROCMLIR-123-fix-attention-mask).
   # Forks: any branch name works on your side; this convention only applies in
   # the upstream repo.
   git checkout -b users/<username>/<description>
   ```
2. Make your change. Add tests under `mlir/test/` and update docs if behavior changes.
3. Format your C/C++ changes with `clang-format` (uses the repo's `.clang-format`):
   ```bash
   git clang-format origin/develop
   ```
   The CI `clang-format` job runs `git clang-format --diff origin/develop` and fails on any non-empty diff.
4. Build and run the test suite locally:
   ```bash
   mkdir -p build && cd build
   cmake -G Ninja .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
   ninja check-rocmlir
   ```
5. Open a PR against `develop`. Describe *what* changed and *why*; link any related issue.
6. Ensure CI passes and request review from the relevant [CODEOWNERS](.github/CODEOWNERS).

By opening a PR, you agree your contribution is licensed under the terms in [LICENSE](LICENSE).

## Coding Standards

The codebase follows the [MLIR coding conventions](https://mlir.llvm.org/getting_started/DeveloperGuide/), which inherit the [LLVM coding standard](https://llvm.org/docs/CodingStandards.html) with one notable difference: variables, parameters, and class members use `camelBack` (lowerCamelCase) instead of LLVM's traditional `Capitalized` form. Functions remain `camelBack`; classes, enums, and unions are `PascalCase`.

Style is enforced via `.clang-format` (LLVM base style) and `.clang-tidy` at the repo root. Step 3 of the workflow above (`git clang-format origin/develop`) is the minimum expectation before opening a PR.

Python helpers (under `mlir/utils/performance/` and elsewhere) follow [`yapf`](.style.yapf) and [`flake8`](.flake8). Format with `yapf -i <files>` and lint with `flake8 <files>` before committing changes there.

## AI Tool Use

This project follows the [LLVM AI Tool Use Policy](https://llvm.org/docs/AIToolPolicy.html). In short:

- You may use any tool (LLMs, copilots, code-gen agents, etc.) to produce contributions, but there must be a **human in the loop**: read and understand everything you submit, and be ready to answer questions about it during review.
- Submitting unreviewed tool output that shifts the review burden onto maintainers (an "extractive contribution") is not acceptable.
- Be transparent. If a contribution contains a substantial amount of tool-generated content, note it in the PR description or commit message (an `Assisted-by:` trailer is the recommended form).
- Automated agents that open PRs, post review comments, or otherwise act in this repository without human approval are not allowed.

See the linked policy for the full rationale and details.

## External Contributors

This repo is part of the ROCm org. Non-AMD contributors need admin approval before being added as collaborators and must follow AMD's [open-source contribution guidelines](https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md).

## Security

For security issues, do **not** open a public issue -- see [SECURITY.md](SECURITY.md).
