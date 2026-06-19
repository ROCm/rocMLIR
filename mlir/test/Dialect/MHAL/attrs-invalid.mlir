// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for the custom attribute parsers in
// COM: external/mlir-hal/lib/Dialect/MHAL/IR/MHAL.cpp: the unknown-type-name
// COM: error branches of TargetObjectAttr::parse and KernelPackageAttr::parse.

// COM: TargetObjectAttr: unknown target object type keyword
func.func @target_obj_unknown_type() attributes {
  // expected-error @+1 {{expected a name of a known target object type}}
  obj = #mhal.target_obj<NotAType = "amdgcn-amd-amdhsa:gfx90a" -> "BINARY">
} {
  return
}

// -----

// COM: KernelPackageAttr: unknown target type keyword
func.func @kernel_pkg_unknown_type() attributes {
  // expected-error @+1 {{expected a name of a known target type}}
  pkg = #mhal.kernel_pkg<NotAType = "amdgcn-amd-amdhsa:gfx90a" : my_kernel [16, 64] -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "B">>
} {
  return
}
