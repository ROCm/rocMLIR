#!/usr/bin/env bash
# Helper for `dynsym_only_mgpu.test`. Asserts that
# `libmlir_rocm_runtime.so` exports nothing but the `mgpu*` C entry
# points. Skips when the library is not built (`MLIR_ENABLE_ROCM_RUNNER=OFF`).

set -u
target="${1:?target shared library}"

if [ ! -e "${target}" ]; then
  echo "dynsym_only_mgpu: skipping; ${target} not built." >&2
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

echo "dynsym_only_mgpu: clean ($(echo "${exports}" | wc -l | tr -d ' ') mgpu symbols)."
exit 0
