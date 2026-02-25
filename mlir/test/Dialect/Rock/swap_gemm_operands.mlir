// RUN: rocmlir-opt -rock-swap-gemm-operands -split-input-file %s | FileCheck %s

// Test 1: Basic swap - no fusions, no C transpose -> swap should apply
// CHECK-LABEL: @swap_basic
// CHECK: rock.gemm tr %{{.*}} = tr %arg1 * tr %arg0
func.func @swap_basic(%a: memref<1x128x256xf32>, %b: memref<1x256x512xf32>, %c: memref<1x128x512xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.gemm %c = %a * %b features = mfma storeMethod = set : memref<1x128x512xf32> = memref<1x128x256xf32> * memref<1x256x512xf32>
  return
}

// -----

// Test 2: Should NOT swap when C is already transposed
// CHECK-LABEL: @no_swap_c_transposed
// CHECK: rock.gemm tr %{{.*}} = %arg0 * %arg1
// CHECK-NOT: tr %arg1 * tr %arg0
func.func @no_swap_c_transposed(%a: memref<1x128x256xf32>, %b: memref<1x256x512xf32>, %c: memref<1x512x128xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.gemm tr %c = %a * %b features = mfma storeMethod = set : memref<1x512x128xf32> = memref<1x128x256xf32> * memref<1x256x512xf32>
  return
}

// -----

// Test 3: Output fusion - gemm writes to alloc, then linalg.generic adds
// the result with another tensor. C traces to memref.alloc, not block arg.
// CHECK-LABEL: @no_swap_output_fusion
// CHECK: rock.gemm %alloc = %{{.*}} * %{{.*}}
// CHECK-NOT: rock.gemm tr
func.func @no_swap_output_fusion(
    %a: memref<1x5x4xf32>, %b: memref<1x4x3xf32>,
    %bias: memref<1x5x3xf32>, %out: memref<1x5x3xf32>
) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5x3xf32>
  rock.gemm %alloc = %a * %b features = mfma storeMethod = set
    : memref<1x5x3xf32> = memref<1x5x4xf32> * memref<1x4x3xf32>
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%alloc, %bias : memref<1x5x3xf32>, memref<1x5x3xf32>)
    outs(%out : memref<1x5x3xf32>) {
  ^bb0(%in0: f32, %in1: f32, %result: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  }
  return
}

// -----

// Test 4: Input fusion - linalg.generic preprocesses A before gemm.
// A traces to memref.alloc (input fusion), not a block arg.
// CHECK-LABEL: @no_swap_input_fusion
// CHECK: rock.gemm %arg2 = %alloc * %arg1
// CHECK-NOT: rock.gemm tr
func.func @no_swap_input_fusion(
    %a_raw: memref<1x128x512xf32>, %b: memref<1x512x256xf32>,
    %c: memref<1x128x256xf32>
) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%a_raw : memref<1x128x512xf32>)
    outs(%alloc : memref<1x128x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    %relu = arith.maximumf %in, %out : f32
    linalg.yield %relu : f32
  }
  rock.gemm %c = %alloc * %b features = mfma storeMethod = set
    : memref<1x128x256xf32> = memref<1x128x512xf32> * memref<1x512x256xf32>
  return
}

// -----

// Test 5: Input fusion on B side - B traces to alloc
// CHECK-LABEL: @no_swap_b_input_fusion
// CHECK: rock.gemm %arg2 = %arg0 * %alloc
// CHECK-NOT: rock.gemm tr
func.func @no_swap_b_input_fusion(
    %a: memref<1x128x512xf32>, %b_raw: memref<1x512x256xf32>,
    %c: memref<1x128x256xf32>
) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x512x256xf32>
  linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%b_raw : memref<1x512x256xf32>)
    outs(%alloc : memref<1x512x256xf32>) {
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    linalg.yield %neg : f32
  }
  rock.gemm %c = %a * %alloc features = mfma storeMethod = set
    : memref<1x128x256xf32> = memref<1x128x512xf32> * memref<1x512x256xf32>
  return
}

// -----

// Test 6: Swap with aTransposed input
// CHECK-LABEL: @swap_a_transposed
// CHECK: rock.gemm tr %{{.*}} = tr %arg1 * %arg0
func.func @swap_a_transposed(%a: memref<1x256x128xf32>, %b: memref<1x256x512xf32>, %c: memref<1x128x512xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.gemm %c = tr %a * %b features = mfma storeMethod = set : memref<1x128x512xf32> = memref<1x256x128xf32> * memref<1x256x512xf32>
  return
}

// -----

// Test 7: Swap with bTransposed input
// CHECK-LABEL: @swap_b_transposed
// CHECK: rock.gemm tr %{{.*}} = %arg1 * tr %arg0
func.func @swap_b_transposed(%a: memref<1x128x256xf32>, %b: memref<1x512x256xf32>, %c: memref<1x128x512xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.gemm %c = %a * tr %b features = mfma storeMethod = set : memref<1x128x512xf32> = memref<1x128x256xf32> * memref<1x512x256xf32>
  return
}

// -----

// Test 8: Swap with both transposed
// CHECK-LABEL: @swap_both_transposed
// CHECK: rock.gemm tr %{{.*}} = %arg1 * %arg0 features
func.func @swap_both_transposed(%a: memref<1x256x128xf32>, %b: memref<1x512x256xf32>, %c: memref<1x128x512xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.gemm %c = tr %a * tr %b features = mfma storeMethod = set : memref<1x128x512xf32> = memref<1x256x128xf32> * memref<1x512x256xf32>
  return
}

// -----

// Test 9: C through rock.transform (view) should still trace to block arg -> swap
#transform_map_c = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{128, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 512] -> [65536]>
// CHECK-LABEL: @swap_c_with_transform
// CHECK: rock.gemm tr %{{.*}} = tr %arg1 * tr %arg0
func.func @swap_c_with_transform(%a: memref<1x128x256xf32>, %b: memref<1x256x512xf32>, %c: memref<65536xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx950"} {
  %c_view = rock.transform %c by #transform_map_c : memref<65536xf32> to memref<1x128x512xf32>
  rock.gemm %c_view = %a * %b features = mfma storeMethod = set : memref<1x128x512xf32> = memref<1x128x256xf32> * memref<1x256x512xf32>
  return
}
