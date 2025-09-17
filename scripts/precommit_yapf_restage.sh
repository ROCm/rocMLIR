#!/usr/bin/env bash
set -eou pipefail

git stash push -q --keep-index --message "Pre-commit auto-stash" || true

mapfile -t files < <(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$' || true)

if (( ${#files[@]} > 0 )); then
    yapf -i "${files[@]}"
    git add "${files[@]}"
fi

git stash pop -q || true

exit 0
