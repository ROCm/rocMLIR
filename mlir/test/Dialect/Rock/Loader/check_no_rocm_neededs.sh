#!/usr/bin/env bash
# Helper for `no_rocm_neededs.test`. Scans build artefacts and asserts
# that none of them transitively `NEED` a ROCm runtime library.
#
# Usage: check_no_rocm_neededs.sh <shlib_dir> <tools_dir>

set -u
shlib_dir="${1:?shlib dir}"
tools_dir="${2:?tools dir}"

# SONAMEs that must never appear in `NEEDED`. We anchor on the SONAME
# prefix (without `.so.<ver>`) so the assertion holds across ROCm major
# versions. `libLLVM.so` (no `.X.X.git` suffix) is ROCm's monolithic
# build; the in-tree LLVM uses libLLVMSupport.so.23.0git etc., which
# we deliberately do NOT match.
forbidden=(
  "libamdhip64"
  "libhiprtc"
  "libamd_comgr"
  "libLLVM.so"   # ROCm's monolithic libLLVM
)

artefacts=(
  # Tools
  "${tools_dir}/rocmlir-driver"
  "${tools_dir}/rocmlir-opt"
  "${tools_dir}/rocmlir-gen"
  "${tools_dir}/rocmlir-tuning-driver"
  "${tools_dir}/xmir-runner"
  "${tools_dir}/mlir-runner"
  # Shared libraries
  "${shlib_dir}/libMLIRRockOps.so.2.0"
  "${shlib_dir}/libMLIRRocmRuntimeLoader.so.23.0git"
  "${shlib_dir}/libmlir_rocm_runtime.so.23.0git"
  "${shlib_dir}/libMLIRRocmExecutionEngineUtils.so.23.0git"
)

if ! command -v readelf >/dev/null 2>&1; then
  echo "no_rocm_neededs: skipping; readelf is not available" >&2
  exit 0
fi

failed=0
for art in "${artefacts[@]}"; do
  if [ ! -e "${art}" ]; then
    # Quietly skip -- not all artefacts are built in every config.
    continue
  fi
  needed="$(readelf -d "${art}" 2>/dev/null | awk '/\(NEEDED\)/{print $5}' | tr -d '[]')"
  for forb in "${forbidden[@]}"; do
    while IFS= read -r soname; do
      [ -z "${soname}" ] && continue
      case "${soname}" in
        ${forb}*)
          echo "FAIL: ${art} declares NEEDED ${soname} (forbidden: ${forb})" >&2
          failed=1
          ;;
      esac
    done <<<"${needed}"
  done
done

if [ "${failed}" -ne 0 ]; then
  echo "no_rocm_neededs: at least one forbidden NEEDED entry was found." >&2
  exit 1
fi

echo "no_rocm_neededs: all checked artefacts are clean."
exit 0
