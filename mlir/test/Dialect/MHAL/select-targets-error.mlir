// RUN: rocmlir-opt --mhal-select-targets='archs=amdgcn-amd-amdhsa:gfx90a target-types=GPU' \
// RUN:   --verify-diagnostics --split-input-file %s -o /dev/null

// COM: Exercises the error path of MHALSelectTargetsPass (external/mlir-hal/
// COM: lib/Dialect/MHAL/Transforms/SelectTargets.cpp lines 89-95): when
// COM: --archs is non-empty AND --target-types excludes CPU AND no kernel
// COM: package matches, the pass emits "target object not found" attached to
// COM: the offending func.func (func.emitError on line 91). One section per
// COM: error so the expected-error annotation aligns with a single emitted
// COM: diagnostic.

// COM: ---- 1: CPU-only package can never match when target-types=GPU.

// expected-error @+1 {{target object not found}}
func.func @gpu_required_cpu_pkg() attributes {mhal.targets = [
  #mhal.kernel_pkg<CPU = "x86_64" : x86_only [1, 1]
    -> #mhal.target_obj<ELF = "x86_64" -> "B">>]} {
  return
}

// -----

// COM: ---- 2: GPU package with a non-matching arch fails when no other
// COM: package matches the requested archs.

// expected-error @+1 {{target object not found}}
func.func @gpu_arch_mismatch() attributes {mhal.targets = [
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx942" : mismatch [16, 64]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx942" -> "B">>]} {
  return
}

// -----

// COM: ---- 3: multiple non-matching packages also produce a single error.

// expected-error @+1 {{target object not found}}
func.func @multi_non_matching() attributes {mhal.targets = [
  #mhal.kernel_pkg<CPU = "x86_64" : pa [1, 1]
    -> #mhal.target_obj<ELF = "x86_64" -> "A">>,
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx942" : pb [16, 64]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx942" -> "B">>]} {
  return
}
