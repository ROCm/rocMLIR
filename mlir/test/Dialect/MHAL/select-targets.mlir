// RUN: rocmlir-opt --mhal-select-targets='archs=amdgcn-amd-amdhsa:gfx90a' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=GFX90A
// RUN: rocmlir-opt --mhal-select-targets='archs=amdgcn-amd-amdhsa:gfx942' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=GFX942
// RUN: rocmlir-opt --mhal-select-targets='archs=gfx90a target-types=CPU' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=CPUONLY
// RUN: rocmlir-opt --mhal-select-targets \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=NOOPTS

// COM: Exercises MHALSelectTargetsPass (external/mlir-hal/lib/Dialect/MHAL/
// COM: Transforms/SelectTargets.cpp). For each func with an mhal.targets
// COM: attribute, the pass iterates over kernel packages and uses
// COM: mhal::SystemDevice::isCompatible (Support/SystemDevice.cpp) to match
// COM: the package's arch against the configured --archs option, then filters
// COM: by --target-types. The last matching package wins and replaces
// COM: mhal.targets. If no package matches and at least one non-CPU type was
// COM: requested with archs, the pass emits "target object not found".
// COM: Bare-identifier arches (e.g. gfx90a) are printed unquoted by
// COM: printKeywordOrString, full triples like amdgcn-amd-amdhsa:gfx90a are
// COM: printed with quotes.

// COM: ---- Case 1: arch filters between two GPU packages -------------------

// GFX90A-LABEL: func.func @two_archs
// GFX90A-SAME: mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : two_archs [16, 64]
// GFX90A-NOT: gfx942

// GFX942-LABEL: func.func @two_archs
// GFX942-SAME: mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx942" : two_archs [32, 128]
// GFX942-NOT: gfx90a

// NOOPTS-LABEL: func.func @two_archs
// NOOPTS-NOT: mhal.targets
func.func @two_archs() attributes {mhal.targets = [
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : two_archs [16, 64]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "B0">>,
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx942" : two_archs [32, 128]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx942" -> "B1">>]} {
  return
}

// -----

// COM: ---- Case 2: type filter picks CPU over GPU when --target-types=CPU.

// CPUONLY-LABEL: func.func @cpu_or_gpu
// CPUONLY-SAME: mhal.targets = [#mhal.kernel_pkg<CPU = gfx90a : cpu_or_gpu [1, 1]

// GFX90A-LABEL: func.func @cpu_or_gpu
// GFX90A-SAME: mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : cpu_or_gpu [16, 64]
func.func @cpu_or_gpu() attributes {mhal.targets = [
  #mhal.kernel_pkg<CPU = "gfx90a" : cpu_or_gpu [1, 1]
    -> #mhal.target_obj<ELF = "gfx90a" -> "B_CPU">>,
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : cpu_or_gpu [16, 64]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "B_GPU">>]} {
  return
}

// -----

// COM: ---- Case 3: 'last match wins'. Both packages match arch gfx90a, so
// COM: the second one (last_wins_b) overwrites the first.

// GFX90A-LABEL: func.func @last_match_wins
// GFX90A-SAME: mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : last_wins_b [4, 256]
// GFX90A-NOT: last_wins_a
func.func @last_match_wins() attributes {mhal.targets = [
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : last_wins_a [1, 64]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "A">>,
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : last_wins_b [4, 256]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "B">>]} {
  return
}

// -----

// COM: ---- Case 4: a func with no mhal.targets attribute is untouched.

// GFX90A-LABEL: func.func @no_targets_attr
// GFX90A-NOT: mhal.targets
// GFX942-LABEL: func.func @no_targets_attr
// GFX942-NOT: mhal.targets
func.func @no_targets_attr() {
  return
}

// -----

// COM: ---- Case 5: an empty --target-types list accepts any TargetType, so
// COM: a CPU-only package whose arch matches is selected.

// GFX90A-LABEL: func.func @cpu_only_arch_match
// GFX90A-SAME: mhal.targets = [#mhal.kernel_pkg<CPU = gfx90a : cpu_only_match [1, 1]
func.func @cpu_only_arch_match() attributes {mhal.targets = [
  #mhal.kernel_pkg<CPU = "gfx90a" : cpu_only_match [1, 1]
    -> #mhal.target_obj<ELF = "gfx90a" -> "B_CPU">>]} {
  return
}

// -----

// COM: ---- Case 6: with --archs=gfx90a a package whose arch is gfx942 is
// COM: not selected; since the default --target-types includes CPU
// COM: (testType(CPU) returns true when targetTypes is empty), no error is
// COM: emitted and the attribute is simply removed (the "target object not
// COM: found" emit path is skipped). With --archs=gfx942 this same package
// COM: matches.

// GFX90A-LABEL: func.func @arch_mismatch_no_error
// GFX90A-NOT: mhal.targets

// GFX942-LABEL: func.func @arch_mismatch_no_error
// GFX942-SAME: mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx942" : mismatch [16, 64]
func.func @arch_mismatch_no_error() attributes {mhal.targets = [
  #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx942" : mismatch [16, 64]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx942" -> "B">>]} {
  return
}

// -----

// COM: ---- Case 7: a chip-only arch in the request (e.g. just "gfx90a")
// COM: compatibly matches a package whose arch is the full triple
// COM: "amdgcn-amd-amdhsa:gfx90a" because SystemDevice::isCompatible
// COM: treats an empty triple on either side as a wildcard.

// CPUONLY-LABEL: func.func @triple_vs_chip_only
// CPUONLY-SAME: mhal.targets = [#mhal.kernel_pkg<CPU = "amdgcn-amd-amdhsa:gfx90a" : triple_vs_chip [1, 1]
// GFX90A-LABEL: func.func @triple_vs_chip_only
// GFX90A-SAME: mhal.targets = [#mhal.kernel_pkg<CPU = "amdgcn-amd-amdhsa:gfx90a" : triple_vs_chip [1, 1]
func.func @triple_vs_chip_only() attributes {mhal.targets = [
  #mhal.kernel_pkg<CPU = "amdgcn-amd-amdhsa:gfx90a" : triple_vs_chip [1, 1]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "B">>]} {
  return
}
