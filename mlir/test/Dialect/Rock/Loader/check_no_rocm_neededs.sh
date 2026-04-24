#!/usr/bin/env bash
# Helper for `no_rocm_neededs.test`. Scans build artefacts and asserts that
# none of them transitively `NEED` a ROCm runtime library.
#
# Usage: check_no_rocm_neededs.sh <shlib_dir> <tools_dir>
#
# Version-agnostic by design: the artefact list uses globs against library
# *base names*, never hardcoded `.so.<MAJOR>.<MINOR>` suffixes, so a future
# LLVM / rocMLIR version bump does not silently turn this test into a no-op.

set -u
shlib_dir="${1:?shlib dir}"
tools_dir="${2:?tools dir}"

# SONAME prefixes that must never appear in `NEEDED`. We anchor on the prefix
# (without `.so.<ver>`) so the assertion holds across ROCm major versions.
# `libLLVM.so` (with no in-tree-style `.<MAJOR>git` suffix) is ROCm's
# monolithic libLLVM build; the in-tree LLVM uses `libLLVMSupport.so.<MAJOR>git`
# etc., which we deliberately do NOT match.
forbidden_re='^lib(amdhip64|hiprtc|amd_comgr|LLVM\.so)'

# Resolve the artefact set at run time. Tools have no extension; shared
# libraries are matched by base-name + wildcard suffix so any version-decorated
# variant is picked up. A bare `lib*.so` dev symlink is skipped to avoid
# double-reporting the same NEEDED set.
shopt -s nullglob
artefacts=()
for t in rocmlir-driver rocmlir-opt rocmlir-gen rocmlir-tuning-driver \
         xmir-runner mlir-runner; do
  artefacts+=("${tools_dir}/${t}")
done
for g in libMLIRRockOps.so libMLIRRocmRuntimeLoader.so \
         libmlir_rocm_runtime.so libMLIRRocmExecutionEngineUtils.so; do
  for f in "${shlib_dir}/${g}".*; do
    artefacts+=("${f}")
  done
done
shopt -u nullglob

if [ "${#artefacts[@]}" -eq 0 ]; then
  echo "no_rocm_neededs: no artefacts found under ${shlib_dir} / ${tools_dir}" >&2
  exit 1
fi
if ! command -v readelf >/dev/null 2>&1; then
  echo "no_rocm_neededs: skipping; readelf is not available" >&2
  exit 0
fi

# Single pipeline per artefact: extract the NEEDED set, then `grep -E` against
# the forbidden-prefix regex. `grep` exits 0 when it finds any forbidden
# SONAME, which we report as a failure.
failed=0
checked=0
for art in "${artefacts[@]}"; do
  [ -e "${art}" ] || continue
  checked=$((checked + 1))
  bad="$(readelf -d "${art}" 2>/dev/null \
         | awk '/\(NEEDED\)/{ gsub(/[][]/,"",$5); print $5 }' \
         | grep -E "${forbidden_re}" || true)"
  if [ -n "${bad}" ]; then
    while IFS= read -r soname; do
      echo "FAIL: ${art} declares NEEDED ${soname}" >&2
    done <<<"${bad}"
    failed=1
  fi
done

if [ "${failed}" -ne 0 ]; then
  echo "no_rocm_neededs: at least one forbidden NEEDED entry was found." >&2
  exit 1
fi
echo "no_rocm_neededs: ${checked} artefact(s) checked, all clean."
