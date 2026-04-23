#!/usr/bin/env bash
# Helper for `dynsym_only_mgpu.test`. Asserts that
# `libmlir_rocm_runtime.so` exports nothing but the `mgpu*` C entry
# points. Skips when the library is not built (`MLIR_ENABLE_ROCM_RUNNER=OFF`).
#
# Usage: check_dynsym_only_mgpu.sh <shlib_dir>
#
# Version-agnostic: we glob `libmlir_rocm_runtime.so*` and pick the
# real file (skipping bare `.so` dev symlinks), so a future LLVM bump
# from `.so.23.0git` to `.so.24.0git` does not silently turn the test
# into a no-op.

set -u
shlib_dir="${1:?shlib dir}"

# Locate the actual shared object: prefer the SONAME-versioned file
# (which is what runtime consumers `dlopen`), falling back to the
# unversioned dev symlink as a last resort.
target=""
shopt -s nullglob
for cand in "${shlib_dir}/libmlir_rocm_runtime.so."*; do
  # Skip directories (defensive) and prefer regular files / symlinks
  # to a regular file.
  if [ -f "${cand}" ]; then
    target="${cand}"
    break
  fi
done
if [ -z "${target}" ] && [ -e "${shlib_dir}/libmlir_rocm_runtime.so" ]; then
  target="${shlib_dir}/libmlir_rocm_runtime.so"
fi
shopt -u nullglob

if [ -z "${target}" ]; then
  echo "dynsym_only_mgpu: skipping; libmlir_rocm_runtime.so* not built." >&2
  exit 0
fi

if ! command -v nm >/dev/null 2>&1; then
  echo "dynsym_only_mgpu: skipping; nm is not available." >&2
  exit 0
fi

# `nm -D --defined-only` lists exported, defined symbols. Strip linker
# pseudo-symbols (`__bss_start`, `_edata`, `_end`, `_init`, `_fini`),
# keep everything else.
exports="$(nm -D --defined-only "${target}" \
            | awk '{print $3}' \
            | grep -Ev '^(_init|_fini|_edata|_end|__bss_start)$' \
            | grep -v '^$' || true)"

bad=""
while IFS= read -r sym; do
  [ -z "${sym}" ] && continue
  case "${sym}" in
    mgpu*) ;;  # accepted
    *) bad="${bad}${bad:+
}${sym}" ;;
  esac
done <<<"${exports}"

if [ -n "${bad}" ]; then
  echo "FAIL: ${target} exports forbidden non-mgpu symbols:" >&2
  echo "${bad}" >&2
  exit 1
fi

count="$(echo "${exports}" | grep -c '^mgpu' || true)"
echo "dynsym_only_mgpu: ${target}: clean (${count} mgpu* symbols)."
exit 0
