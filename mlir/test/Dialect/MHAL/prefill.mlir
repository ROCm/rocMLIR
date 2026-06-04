// RUN: rocmlir-opt --mhal-prefill --split-input-file %s | FileCheck %s

// COM: Exercises MHALPrefillPass (external/mlir-hal/lib/Dialect/MHAL/
// COM: Transforms/Prefill.cpp). For each gpu.launch_func in a func, the pass
// COM: looks up the referenced gpu.binary, reads its single object's
// COM: 'properties' dict, finds an ArrayAttr keyed by the launched kernel
// COM: name, filters that array for #mhal.prefill attributes, and for each
// COM: one inserts an arith.constant + gpu.memset on the matching kernel
// COM: operand right before the launch.

// CHECK-LABEL: func.func @one_prefill_f32
// CHECK-DAG: %[[CST:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: gpu.memset %{{.*}}, %[[CST]] : memref<256xf32>, f32
// CHECK: gpu.launch_func @kernels::@k_one
module attributes {gpu.container_module} {
  gpu.binary @kernels [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                properties = {k_one = [#mhal.mhal.prefill<0, 1.0 : f32>]},
                kernels = #gpu.kernel_table<[
                  #gpu.kernel_metadata<"k_one", (memref<256xf32>) -> ()>]>,
                "BIN">]
  func.func @one_prefill_f32(%arg0: memref<256xf32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@k_one
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<256xf32>)
    return
  }
}

// -----

// COM: Two prefill attrs on the same launch produce two memsets, one per
// COM: operand. The order in IR matches the array order in properties.

// CHECK-LABEL: func.func @two_prefills
// CHECK: %[[Z:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: gpu.memset %{{.*}}, %[[Z]] : memref<64xf32>, f32
// CHECK: %[[Q:.+]] = arith.constant -1 : i32
// CHECK-NEXT: gpu.memset %{{.*}}, %[[Q]] : memref<32xi32>, i32
// CHECK: gpu.launch_func @kernels::@k_two
module attributes {gpu.container_module} {
  gpu.binary @kernels [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                properties = {k_two = [
                  #mhal.mhal.prefill<0, 0.0 : f32>,
                  #mhal.mhal.prefill<1, -1 : i32>]},
                kernels = #gpu.kernel_table<[
                  #gpu.kernel_metadata<"k_two", (memref<64xf32>, memref<32xi32>) -> ()>]>,
                "BIN">]
  func.func @two_prefills(%arg0: memref<64xf32>, %arg1: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@k_two
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<64xf32>, %arg1 : memref<32xi32>)
    return
  }
}

// -----

// COM: The element type of the memset constant must match the memref's
// COM: element type. Covers f16, i8, and a memset on a non-zero argument
// COM: index to verify the argIdx field of #mhal.prefill is honored.

// CHECK-LABEL: func.func @element_types
// CHECK: %[[H:.+]] = arith.constant 5.000000e-01 : f16
// CHECK-NEXT: gpu.memset %{{.*}}, %[[H]] : memref<128xf16>, f16
// CHECK: %[[B:.+]] = arith.constant 7 : i8
// CHECK-NEXT: gpu.memset %{{.*}}, %[[B]] : memref<16xi8>, i8
// CHECK: gpu.launch_func @kernels::@k_types
module attributes {gpu.container_module} {
  gpu.binary @kernels [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                properties = {k_types = [
                  #mhal.mhal.prefill<0, 5.0e-1 : f16>,
                  #mhal.mhal.prefill<2, 7 : i8>]},
                kernels = #gpu.kernel_table<[
                  #gpu.kernel_metadata<"k_types", (memref<128xf16>, memref<8xf32>, memref<16xi8>) -> ()>]>,
                "BIN">]
  func.func @element_types(%arg0: memref<128xf16>, %arg1: memref<8xf32>, %arg2: memref<16xi8>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@k_types
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<128xf16>, %arg1 : memref<8xf32>, %arg2 : memref<16xi8>)
    return
  }
}

// -----

// COM: When the properties dict has no entry whose key matches the launched
// COM: kernel name, no memset is inserted (objectProps.get(name) returns
// COM: nullptr at Prefill.cpp line 61).

// CHECK-LABEL: func.func @kernel_not_in_props
// CHECK-NOT: gpu.memset
// CHECK: gpu.launch_func @kernels::@k_missing
module attributes {gpu.container_module} {
  gpu.binary @kernels [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                properties = {some_other_key = [#mhal.mhal.prefill<0, 0.0 : f32>]},
                kernels = #gpu.kernel_table<[
                  #gpu.kernel_metadata<"k_missing", (memref<8xf32>) -> ()>]>,
                "BIN">]
  func.func @kernel_not_in_props(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@k_missing
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<8xf32>)
    return
  }
}

// -----

// COM: When the object has no 'properties' dict, the pass bails out early
// COM: (Prefill.cpp lines 59-60) and emits no memset.

// CHECK-LABEL: func.func @no_properties
// CHECK-NOT: gpu.memset
// CHECK: gpu.launch_func @kernels::@k_no_props
module attributes {gpu.container_module} {
  gpu.binary @kernels [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                kernels = #gpu.kernel_table<[
                  #gpu.kernel_metadata<"k_no_props", (memref<8xf32>) -> ()>]>,
                "BIN">]
  func.func @no_properties(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@k_no_props
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<8xf32>)
    return
  }
}

// -----

// COM: The ArrayAttr keyed by the kernel name may contain non-#mhal.prefill
// COM: attributes (mixed metadata). The dyn_cast filter at Prefill.cpp line
// COM: 64 silently skips them and only the prefill entries are realized as
// COM: memsets.

// CHECK-LABEL: func.func @mixed_attr_array
// CHECK: %[[V:.+]] = arith.constant 3 : i32
// CHECK: gpu.memset %{{.*}}, %[[V]] : memref<4xi32>, i32
// CHECK: gpu.launch_func @kernels::@k_mixed
module attributes {gpu.container_module} {
  gpu.binary @kernels [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                properties = {k_mixed = ["not_a_prefill", 42 : i64, #mhal.mhal.prefill<0, 3 : i32>]},
                kernels = #gpu.kernel_table<[
                  #gpu.kernel_metadata<"k_mixed", (memref<4xi32>) -> ()>]>,
                "BIN">]
  func.func @mixed_attr_array(%arg0: memref<4xi32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@k_mixed
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<4xi32>)
    return
  }
}

// -----

// COM: A func with no gpu.launch_func at all is a no-op for the pass:
// COM: nothing is walked, nothing is inserted, the IR is unchanged.

// CHECK-LABEL: func.func @no_launches
// CHECK-NEXT: return
func.func @no_launches() {
  return
}

// -----

// COM: When the properties entry for the launched kernel exists but is NOT
// COM: an ArrayAttr (here a bare string), the `dyn_cast<ArrayAttr>(moduleAttr)`
// COM: at Prefill.cpp line 62 fails and the inner loop is skipped. No memset
// COM: is inserted; the IR is unchanged apart from a no-op pass walk.

// CHECK-LABEL: func.func @kernel_props_not_array
// CHECK-NOT: gpu.memset
// CHECK: gpu.launch_func @kernels::@k_not_array
module attributes {gpu.container_module} {
  gpu.binary @kernels [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                properties = {k_not_array = "string_instead_of_array"},
                kernels = #gpu.kernel_table<[
                  #gpu.kernel_metadata<"k_not_array", (memref<8xf32>) -> ()>]>,
                "BIN">]
  func.func @kernel_props_not_array(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@k_not_array
      blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<8xf32>)
    return
  }
}
