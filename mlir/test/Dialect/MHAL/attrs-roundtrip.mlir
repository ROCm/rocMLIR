// RUN: rocmlir-opt %s --split-input-file | rocmlir-opt --split-input-file | \
// RUN:   FileCheck %s
// RUN: rocmlir-opt %s --split-input-file --mlir-print-op-generic | \
// RUN:   rocmlir-opt --split-input-file | FileCheck %s

// COM: Exercises the custom parse/print logic in
// COM: external/mlir-hal/lib/Dialect/MHAL/IR/MHAL.cpp for the three MHAL
// COM: attributes (#mhal.target_obj, #mhal.kernel_pkg, #mhal.prefill).
// COM: Each section round-trips at least once through the pretty printer and
// COM: once through the generic printer, so any new field or formatting
// COM: change in MHAL.cpp must keep the printed form re-parseable.

// COM: ---- TargetObjectAttr: bare minimal and all three target types ------

// CHECK-LABEL: func.func @target_obj_minimal
// CHECK-SAME: obj = #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "BINARY">
func.func @target_obj_minimal() attributes {
  obj = #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "BINARY">
} {
  return
}

// COM: All three TargetObjectType keywords (ELF, LLVMIR, SPIRV). LLVMIR
// COM: uses the optional attribute dict slot. Note that MLIR sorts
// COM: attributes in a func's attribute dict alphabetically on output, so the
// COM: CHECK-SAME lines below must match that order.

// CHECK-LABEL: func.func @target_obj_all_types
// CHECK-SAME: elf = #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "ELF_BIN">
// CHECK-SAME: llvmir = #mhal.target_obj<LLVMIR = "amdgcn-amd-amdhsa:gfx942" {O = 3 : i32} -> "B2">
// CHECK-SAME: spirv = #mhal.target_obj<SPIRV = "spirv-vulkan" -> "B">
func.func @target_obj_all_types() attributes {
  elf = #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "ELF_BIN">,
  llvmir = #mhal.target_obj<LLVMIR = "amdgcn-amd-amdhsa:gfx942" {O = 3 : i32} -> "B2">,
  spirv = #mhal.target_obj<SPIRV = "spirv-vulkan" -> "B">
} {
  return
}

// COM: Architecture strings that are valid bare identifiers (alphanumeric +
// COM: underscore, starts with letter) are printed unquoted by
// COM: parseKeywordOrString / printKeywordOrString.

// CHECK-LABEL: func.func @target_obj_bare_ident_arch
// CHECK-SAME: obj = #mhal.target_obj<ELF = gfx90a -> "B">
func.func @target_obj_bare_ident_arch() attributes {
  obj = #mhal.target_obj<ELF = "gfx90a" -> "B">
} {
  return
}

// -----

// COM: ---- KernelPackageAttr: every TargetType keyword + every optional slot

// CHECK-LABEL: func.func @kernel_pkg_gpu_full_triple
// CHECK-SAME: pkg = #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : my_kernel [16, 64] -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "B">>
func.func @kernel_pkg_gpu_full_triple() attributes {
  pkg = #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : my_kernel [16, 64]
    -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" -> "B">>
} {
  return
}

// COM: CPU / NPU / ALT TargetType keywords plus arches that are valid bare
// COM: identifiers. The output dict is alphabetically sorted.

// CHECK-LABEL: func.func @kernel_pkg_all_target_types
// CHECK-SAME: alt = #mhal.kernel_pkg<ALT = alt : alt_kernel [8, 8] -> #mhal.target_obj<ELF = alt -> "B">>
// CHECK-SAME: cpu = #mhal.kernel_pkg<CPU = x86_64 : cpu_kernel [1, 1] -> #mhal.target_obj<ELF = x86_64 -> "B">>
// CHECK-SAME: npu = #mhal.kernel_pkg<NPU = npu0 : npu_kernel [2, 4] -> #mhal.target_obj<ELF = npu0 -> "B">>
func.func @kernel_pkg_all_target_types() attributes {
  cpu = #mhal.kernel_pkg<CPU = "x86_64" : cpu_kernel [1, 1]
    -> #mhal.target_obj<ELF = "x86_64" -> "B">>,
  npu = #mhal.kernel_pkg<NPU = "npu0" : npu_kernel [2, 4]
    -> #mhal.target_obj<ELF = "npu0" -> "B">>,
  alt = #mhal.kernel_pkg<ALT = "alt" : alt_kernel [8, 8]
    -> #mhal.target_obj<ELF = "alt" -> "B">>
} {
  return
}

// COM: The optional `{<attrs>}` slot between launch dims and `->` survives
// COM: a round trip and the attribute dict is sorted alphabetically.

// CHECK-LABEL: func.func @kernel_pkg_with_attrs
// CHECK-SAME: pkg = #mhal.kernel_pkg<GPU = gfx90a : k [16, 64] {a_first = 1 : i32, bare_ptr_abi = true} -> #mhal.target_obj<ELF = gfx90a -> "B">>
func.func @kernel_pkg_with_attrs() attributes {
  pkg = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64]
    {bare_ptr_abi = true, a_first = 1 : i32}
    -> #mhal.target_obj<ELF = "gfx90a" -> "B">>
} {
  return
}

// COM: launchDims is `Variadic<int>`, so any positive number of dims works.

// CHECK-LABEL: func.func @kernel_pkg_many_launch_dims
// CHECK-SAME: pkg = #mhal.kernel_pkg<GPU = gfx90a : k [1, 2, 3, 4, 5] -> #mhal.target_obj<ELF = gfx90a -> "B">>
func.func @kernel_pkg_many_launch_dims() attributes {
  pkg = #mhal.kernel_pkg<GPU = "gfx90a" : k [1, 2, 3, 4, 5]
    -> #mhal.target_obj<ELF = "gfx90a" -> "B">>
} {
  return
}

// COM: Object payload may itself be a structured #gpu.object (used after
// COM: --mhal-package-targets attaches a gpu.binary), not just a bare string.

// CHECK-LABEL: func.func @kernel_pkg_with_gpu_object_payload
// CHECK-SAME: pkg = #mhal.kernel_pkg<GPU = gfx90a : k [16, 64] -> #mhal.target_obj<ELF = gfx90a -> #gpu.object<#rocdl.target<chip = "gfx90a">, "BIN">>>
func.func @kernel_pkg_with_gpu_object_payload() attributes {
  pkg = #mhal.kernel_pkg<GPU = "gfx90a" : k [16, 64] ->
    #mhal.target_obj<ELF = "gfx90a" ->
      #gpu.object<#rocdl.target<chip = "gfx90a">, "BIN">>>
} {
  return
}

// -----

// COM: ---- PrefillAttr is generated with the auto `<` params `>` assembly
// COM: format. Its mnemonic is literally "mhal.prefill", so the
// COM: dialect-qualified name printed by MLIR is `#mhal.mhal.prefill<...>`.
// COM: argIndex is uint32_t, initValue is any TypedAttr; exercise integer and
// COM: floating point element types of several widths.

// CHECK-LABEL: func.func @prefill_attrs
// CHECK-SAME: f16_val = #mhal.mhal.prefill<4, 3.140630e+00 : f16>
// CHECK-SAME: f32_one = #mhal.mhal.prefill<1, 1.000000e+00 : f32>
// CHECK-SAME: f32_zero = #mhal.mhal.prefill<0, 0.000000e+00 : f32>
// CHECK-SAME: i32_val = #mhal.mhal.prefill<2, 42 : i32>
// CHECK-SAME: i8_val = #mhal.mhal.prefill<3, -1 : i8>
func.func @prefill_attrs() attributes {
  f32_zero = #mhal.mhal.prefill<0, 0.0 : f32>,
  f32_one  = #mhal.mhal.prefill<1, 1.0 : f32>,
  i32_val  = #mhal.mhal.prefill<2, 42 : i32>,
  i8_val   = #mhal.mhal.prefill<3, -1 : i8>,
  f16_val  = #mhal.mhal.prefill<4, 3.140625 : f16>
} {
  return
}
