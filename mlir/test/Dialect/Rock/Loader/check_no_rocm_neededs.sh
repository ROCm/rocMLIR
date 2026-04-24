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

# SONAMEs that must never appear in `NEEDED`. We anchor on the full library
# basename (`<name>.so` / `<name>.dll`) followed either by end-of-string or by
# a version separator (`.<MAJOR>` / `-<MAJOR>git`). This rejects false
# positives like `libamd_comgr_helper.so.1` (extra suffix on the basename) or
# the in-tree `libLLVMSupport.so.23.0git` (component-decorated `libLLVM*`),
# while still matching every real ROCm SONAME we have ever seen
# (`libamdhip64.so`, `libamdhip64.so.7`, `libhiprtc.so.7`,
# `libamd_comgr.so.3`, `libLLVM.so.23.0git`, `libLLVM-23git.so` -- the latter
# normalised to its `libLLVM.so.<X>` SONAME by the dynamic linker).
forbidden_re='^lib(amdhip64|hiprtc|amd_comgr|LLVM)\.(so|dll)([.-]|$)'

# Resolve the artefact set at run time. Tools have no extension and are added
# only when they actually exist; shared libraries are matched by base name +
# wildcard suffix so any version-decorated variant is picked up. A run with
# zero artefacts is treated as a wrong invocation and fails loudly so the test
# never silently degrades to a no-op (e.g. when the caller passes the wrong
# build directory).
shopt -s nullglob
artefacts=()
for t in rocmlir-driver rocmlir-opt rocmlir-gen rocmlir-tuning-driver \
         xmir-runner mlir-runner; do
  if [ -e "${tools_dir}/${t}" ]; then
    artefacts+=("${tools_dir}/${t}")
  fi
done
# Match `.so.<digit>...` and the bare `.so` symlink, but NOT side files
# like `.so.<...>.dwo` (split-DWARF debug info), `.so.<...>.debug`
# (separate debug info), or `.so.<...>.dbg`. We anchor the version suffix
# on a digit so non-version decorations are rejected; the trailing case
# statement also drops any debug companion that slipped through.
# (Brackets must stay outside double-quotes for bash to treat them as a
# character class.)
for g in libMLIRRockOps libMLIRRocmRuntimeLoader libmlir_rocm_runtime \
         libMLIRRocmExecutionEngineUtils; do
  for f in "${shlib_dir}/${g}".so.[0-9]* "${shlib_dir}/${g}".so; do
    case "${f}" in
      *.dwo|*.debug|*.dbg) continue ;;
    esac
    if [ -f "${f}" ] || [ -L "${f}" ]; then
      artefacts+=("${f}")
    fi
  done
done
shopt -u nullglob

if [ "${#artefacts[@]}" -eq 0 ]; then
  echo "no_rocm_neededs: no artefacts found under" \
       "tools=${tools_dir} shlib=${shlib_dir}" >&2
  echo "(this usually means the caller passed the wrong build directory;" \
       "check the lit substitutions \`%rocmlir_tools_dir\` and" \
       "\`%rocmlir_shlib_dir\`.)" >&2
  exit 1
fi
if ! command -v readelf >/dev/null 2>&1; then
  echo "no_rocm_neededs: skipping; readelf is not available" >&2
  exit 0
fi

# Per artefact: get `readelf -d`, split out the NEEDED entries, then grep for
# any forbidden basename. We capture `readelf` and `awk` outputs separately
# from the final `grep` so a failure in either tool surfaces as a non-zero
# exit code and a diagnostic, rather than being silently swallowed by an
# empty pipeline.
failed=0
for art in "${artefacts[@]}"; do
  if ! readelf_out="$(readelf -d "${art}" 2>/dev/null)"; then
    echo "FAIL: readelf -d ${art} failed" >&2
    failed=1
    continue
  fi
  needed="$(awk '/\(NEEDED\)/{ gsub(/[][]/,"",$5); print $5 }' \
            <<<"${readelf_out}")"
  bad="$(grep -E "${forbidden_re}" <<<"${needed}" || true)"
  if [ -n "${bad}" ]; then
    while IFS= read -r soname; do
      echo "FAIL: ${art} declares NEEDED ${soname}" >&2
    done <<<"${bad}"
    failed=1
  fi
done
checked="${#artefacts[@]}"

if [ "${failed}" -ne 0 ]; then
  echo "no_rocm_neededs: at least one forbidden NEEDED entry was found." >&2
  exit 1
fi
echo "no_rocm_neededs: ${checked} artefact(s) checked, all clean."
