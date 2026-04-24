#!/usr/bin/env bash
# Helper for `dynsym_only_mgpu.test`. Asserts that `libmlir_rocm_runtime.so`
# exports nothing but the `mgpu*` C entry points. Skips when the library is
# not built (`MLIR_ENABLE_ROCM_RUNNER=OFF`) or when `nm` is not available.
#
# Usage: check_dynsym_only_mgpu.sh <shlib_dir>
#
# Version-agnostic: we glob `libmlir_rocm_runtime.so*` (preferring the SONAME-
# versioned file, falling back to the unversioned dev symlink), so a future
# LLVM bump from `.so.23.0git` to `.so.24.0git` does not silently turn the
# test into a no-op.

set -u
shlib_dir="${1:?shlib dir}"

# Pick the SONAME-versioned file first (what runtime consumers actually
# `dlopen`); fall back to the dev symlink. The version suffix glob is
# anchored on a digit so we never accidentally match `.so.<X>.dwo`
# (split-DWARF), `.debug`, or `.dbg` companion files. `nullglob` makes
# the glob expand to nothing on a clean miss instead of leaving the
# literal pattern. (Brackets must stay outside double-quotes for bash
# to treat them as a character class.)
shopt -s nullglob
candidates=("${shlib_dir}"/libmlir_rocm_runtime.so.[0-9]* \
            "${shlib_dir}/libmlir_rocm_runtime.so")
shopt -u nullglob

target=""
for cand in "${candidates[@]}"; do
  case "${cand}" in
    *.dwo|*.debug|*.dbg) continue ;;
  esac
  if [ -f "${cand}" ] || [ -L "${cand}" ]; then
    target="${cand}"
    break
  fi
done

if [ -z "${target}" ]; then
  echo "dynsym_only_mgpu: skipping; libmlir_rocm_runtime.so* not built." >&2
  exit 0
fi
if ! command -v nm >/dev/null 2>&1; then
  echo "dynsym_only_mgpu: skipping; nm is not available." >&2
  exit 0
fi

# Capture `nm` output separately from the awk filter so a `nm` failure
# surfaces immediately rather than being silently swallowed.
if ! nm_out="$(nm -D --defined-only "${target}" 2>/dev/null)"; then
  echo "FAIL: nm -D --defined-only ${target} failed" >&2
  exit 1
fi

# Single `awk` pass: drop linker pseudo-symbols, partition into "mgpu*"
# (allowed) and everything else (forbidden). Tag the two categories with
# `OK ` / `BAD ` line prefixes so the caller can split them apart with a
# single `grep` per category without sentinel lines or empty spacers.
report="$(awk '
  $3 ~ /^(_init|_fini|_edata|_end|__bss_start)$/ { next }
  $3 == "" { next }
  $3 ~ /^mgpu/ { print "OK " $3; next }
  { print "BAD " $3 }' <<<"${nm_out}")"

bad="$(grep '^BAD ' <<<"${report}" | cut -d' ' -f2- || true)"
if [ -n "${bad}" ]; then
  echo "FAIL: ${target} exports forbidden non-mgpu symbols:" >&2
  while IFS= read -r sym; do
    echo "  ${sym}" >&2
  done <<<"${bad}"
  exit 1
fi

ok="$(grep -c '^OK ' <<<"${report}" || true)"
echo "dynsym_only_mgpu: ${target}: clean (${ok} mgpu* symbols)."
