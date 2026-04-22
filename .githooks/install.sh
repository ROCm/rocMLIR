#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

hooks_src=".githooks"
hooks_dst=".git/hooks"

if [[ ! -d "$hooks_dst" ]]; then
  echo "Error: $hooks_dst not found (is this a git checkout?)"
  exit 1
fi

install_hook() {
  local hook="$1"
  local src="$hooks_src/$hook"
  local dst="$hooks_dst/$hook"

  if [[ ! -f "$src" ]]; then
    echo "Error: missing $src"
    exit 1
  fi

  if [[ -f "$dst" ]] && [[ ! -L "$dst" ]]; then
    cp -f "$dst" "$dst.bak.$(date +%s)"
  fi

  cp -f "$src" "$dst"
  chmod +x "$dst"
  echo "Installed $dst"
}

install_hook pre-commit
install_hook pre-push

echo
echo "Done. Hooks installed into .git/hooks/"
echo "To uninstall, remove .git/hooks/pre-commit and .git/hooks/pre-push (and any .bak.* backups)."

