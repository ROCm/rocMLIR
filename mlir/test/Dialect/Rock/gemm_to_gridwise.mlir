// Ensures that the padding application, group application, etc. in gemm-to-gridwise
// function as expected.

// RUN: rocmlir-opt -rock-gemm-to-gridwise -mlir-print-local-scope %s | FileCheck %s

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#general_gemm_params_splitk = #rock.general_gemm_params<blockSize = 64, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 2, scheduleVersion = 1, outputSwizzle = 2>
#general_gemm_params1 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#xdlops_gemm_params0 = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>
#xdlops_gemm_params1 = #rock.mfma_gemm_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>
#xdlops_gemm_params3 = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 64, mnPerXdl = 32, forceUnroll = true, splitKFactor = 3, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>
#xldops_attn_params_g0 = #rock.mfma_gemm_params<kpackPerBlock = 1, mPerBlock = 32, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>
#xldops_attn_params_g1 = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>
#xldops_attn_params_g1_splitk = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>

// CHECK-LABEL: func.func @gemm_easy_case_from_conv
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
// CHECK-SAME: grid_size = 4
func.func @gemm_easy_case_from_conv(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK-NEXT: rock.gridwise_gemm %[[c]] = %[[a]] * %[[b]]
  rock.gemm %c = tr %a * %b features = none storeMethod = set {
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_splitk
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: grid_size = 8 : i32
func.func @gemm_splitk(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %c = tr %a * %b features = atomic_add storeMethod = set {
    gridSize = 4 : i32,
    params = #general_gemm_params_splitk
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_easy_case_from_conv_xdlops
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
// CHECK-SAME: grid_size = 16 : i32
func.func @gemm_easy_case_from_conv_xdlops(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-NEXT: rock.gridwise_gemm_accel(%[[a]], %[[b]], %[[c]])
  rock.gemm %c = tr %a * %b storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_most_general_padding_case
// CHECK-SAME: (%[[a:.*]]: memref<1x1x1xf32>, %[[b:.*]]: memref<1x1x1xf32>, %[[c:.*]]: memref<1x1x1xf32>)
// CHECK-SAME: grid_size = 1
func.func @gemm_most_general_padding_case(%a: memref<1x1x1xf32>, %b: memref<1x1x1xf32>, %c: memref<1x1x1xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK-DAG: %[[padA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x1x1xf32> to memref<1x16x64xf32{{.*}}>
  // CHECK-DAG: %[[padB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x1x1xf32> to memref<1x16x64xf32{{.*}}>
  // CHECK-DAG: %[[padC:.*]] = rock.transform %[[c]] by {{.*}} : memref<1x1x1xf32> to memref<1x64x64xf32{{.*}}>
  // CHECK: rock.gridwise_gemm %[[padC]] = %[[padA]] * %[[padB]]
  rock.gemm %c = tr %a * %b features = none storeMethod = set {
    gridSize = 1 : i32,
    params = #general_gemm_params1
  } : memref<1x1x1xf32> = memref<1x1x1xf32> * memref<1x1x1xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_in_standard_form
// CHECK-SAME: (%[[a:.*]]: memref<128x72xf32>, %[[b:.*]]: memref<72x512xf32>, %[[c:.*]]: memref<128x512xf32>)
// CHECK-SAME: grid_size = 4
func.func @gemm_in_standard_form(%a: memref<128x72xf32>, %b: memref<72x512xf32>, %c: memref<128x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] by {{.*}} : memref<128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<72x512xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] by {{.*}} : memref<128x512xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm %[[normalizeC]] = %[[normalizeA]] * %[[normalizeB]]
  rock.gemm %c = %a * %b features = none storeMethod = set {
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : memref<128x512xf32> = memref<128x72xf32> * memref<72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_accel_layout_both
// CHECK-SAME: (%[[a:.*]]: memref<128x72xf32>, %[[b:.*]]: memref<72x512xf32>, %[[c:.*]]: memref<128x512xf32>)
func.func @gemm_accel_layout_both(%a: memref<128x72xf32>, %b: memref<72x512xf32>, %c: memref<128x512xf32>) {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] by {{.*}} : memref<128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<72x512xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] by {{.*}} : memref<128x512xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm_accel(%[[normalizeA]], %[[normalizeB]], %[[normalizeC]])
  // CHECK-SAME: aAccelLayout
  // CHECK-SAME: bAccelLayout
  rock.gemm %c = %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx908",
    derivedBlockSize = 256 : i32,
    aAccelLayout,
    bAccelLayout,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params0
  } : memref<128x512xf32> = memref<128x72xf32> * memref<72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_accel_layout_A
// CHECK-SAME: (%[[a:.*]]: memref<128x72xf32>, %[[b:.*]]: memref<72x512xf32>, %[[c:.*]]: memref<128x512xf32>)
func.func @gemm_accel_layout_A(%a: memref<128x72xf32>, %b: memref<72x512xf32>, %c: memref<128x512xf32>) {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] by {{.*}} : memref<128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<72x512xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] by {{.*}} : memref<128x512xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm_accel(%[[normalizeA]], %[[normalizeB]], %[[normalizeC]])
  // CHECK-SAME: aAccelLayout
  // CHECK-NOT: bAccelLayout
  rock.gemm %c = %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx908",
    derivedBlockSize = 256 : i32,
    aAccelLayout,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params0
  } : memref<128x512xf32> = memref<128x72xf32> * memref<72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_accel_layout_B
// CHECK-SAME: (%[[a:.*]]: memref<128x72xf32>, %[[b:.*]]: memref<72x512xf32>, %[[c:.*]]: memref<128x512xf32>)
func.func @gemm_accel_layout_B(%a: memref<128x72xf32>, %b: memref<72x512xf32>, %c: memref<128x512xf32>) {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] by {{.*}} : memref<128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<72x512xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] by {{.*}} : memref<128x512xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm_accel(%[[normalizeA]], %[[normalizeB]], %[[normalizeC]])
  // CHECK-SAME: bAccelLayout
  // CHECK-NOT: aAccelLayout
  rock.gemm %c = %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx908",
    derivedBlockSize = 256 : i32,
    bAccelLayout,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params0
  } : memref<128x512xf32> = memref<128x72xf32> * memref<72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_transposed_from_gridwise
// CHECK-SAME: (%[[a:.*]]: memref<1x128x72xf32>, %[[b:.*]]: memref<1x512x72xf32>, %[[c:.*]]: memref<1x512x128xf32>)
// CHECK-SAME: grid_size = 4
func.func @gemm_transposed_from_gridwise(%a: memref<1x128x72xf32>, %b: memref<1x512x72xf32>, %c: memref<1x512x128xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] {{.*}} : memref<1x128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] {{.*}} : memref<1x512x72xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] {{.*}} : memref<1x512x128xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm %[[normalizeC]] = %[[normalizeA]] * %[[normalizeB]]
  rock.gemm tr %c = %a * tr %b features = none storeMethod = set {
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : memref<1x512x128xf32> = memref<1x128x72xf32> * memref<1x512x72xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_pad_for_split_k
// CHECK-SAME: (%[[a:.*]]: memref<1x128x238xf32>, %[[b:.*]]: memref<1x238x512xf32>, %[[c:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: grid_size = 48
func.func @gemm_pad_for_split_k(%a: memref<1x128x238xf32>, %b: memref<1x238x512xf32>, %c: memref<1x128x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x128x238xf32> to memref<1x238x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : memref<1x238x128xf32> to memref<1x240x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x238x512xf32> to memref<1x240x512xf32{{.*}}>
  // CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : memref<1x240x128xf32> to memref<1x3x80x128xf32{{.*}}>
  // CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : memref<1x240x512xf32> to memref<1x3x80x512xf32{{.*}}>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %a * %b storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x128x512xf32> = memref<1x128x238xf32> * memref<1x238x512xf32>
  memref.copy %alloc, %c : memref<1x128x512xf32> to memref<1x128x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_reduce_and_split_k
// CHECK-SAME: (%[[a:.*]]: memref<1x128x238xf32>, %[[b:.*]]: memref<1x238x512xf32>, %[[c:.*]]: memref<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: grid_size = 48
func.func @gemm_reduce_and_split_k(%a: memref<1x128x238xf32>, %b: memref<1x238x512xf32>, %c: memref<1x128x1xf32>, %d: memref<1x128x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x128x238xf32> to memref<1x238x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : memref<1x238x128xf32> to memref<1x240x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x238x512xf32> to memref<1x240x512xf32{{.*}}>
  // CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : memref<1x240x128xf32> to memref<1x3x80x128xf32{{.*}}>
  // CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : memref<1x240x512xf32> to memref<1x3x80x512xf32{{.*}}>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
  %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %a * %b storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x128x512xf32> = memref<1x128x238xf32> * memref<1x238x512xf32>
  rock.reduce sum %alloc into %alloc2 {axis = 2 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<1x128x512xf32> into memref<1x128x1xf32>
  memref.copy %alloc, %d : memref<1x128x512xf32> to memref<1x128x512xf32>
  memref.copy %alloc2, %c : memref<1x128x1xf32> to memref<1x128x1xf32>

  func.return
}

// CHECK-LABEL: func.func @gemm_reduce_and_split_k_return_reduce_directly
// CHECK-SAME: (%[[a:.*]]: memref<1x128x238xf32>, %[[b:.*]]: memref<1x238x512xf32>, %[[c:.*]]: memref<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: grid_size = 48
func.func @gemm_reduce_and_split_k_return_reduce_directly(%a: memref<1x128x238xf32>, %b: memref<1x238x512xf32>, %c: memref<1x128x1xf32>, %d: memref<1x128x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x128x238xf32> to memref<1x238x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : memref<1x238x128xf32> to memref<1x240x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x238x512xf32> to memref<1x240x512xf32{{.*}}>
  // CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : memref<1x240x128xf32> to memref<1x3x80x128xf32{{.*}}>
  // CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : memref<1x240x512xf32> to memref<1x3x80x512xf32{{.*}}>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %a * %b storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x128x512xf32> = memref<1x128x238xf32> * memref<1x238x512xf32>
  rock.reduce sum %alloc into %c {axis = 2 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<1x128x512xf32> into memref<1x128x1xf32>
  memref.copy %alloc, %d : memref<1x128x512xf32> to memref<1x128x512xf32>

  func.return
}

// CHECK-LABEL: func.func @gemm_fusion_to_f32_split_k
// CHECK-SAME: (%[[a:.*]]: memref<1x5x4xf16>, %[[b:.*]]: memref<1x4x3xf16>, %[[c:.*]]: memref<1x5x3xf16>, %[[d:.*]]: memref<1x5x3xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-SAME: grid_size = 3
func.func @gemm_fusion_to_f32_split_k(%arg0: memref<1x5x4xf16>, %arg1: memref<1x4x3xf16>, %arg2: memref<1x5x3xf16>, %arg3: memref<1x5x3xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx908"} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5x3xf16>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %arg0 * %arg1 storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x5x3xf16> = memref<1x5x4xf16> * memref<1x4x3xf16>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x5x3xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %arg2 : memref<1x5x3xf16>, memref<1x5x3xf16>) outs(%alloc_0 : memref<1x5x3xf32>) {
  ^bb0(%in: f16, %in_1: f16, %out: f32):
    %7 = arith.addf %in, %in_1 : f16
    %8 = arith.extf %7 : f16 to f32
    linalg.yield %8 : f32
  }
  memref.copy %alloc_0, %arg3 : memref<1x5x3xf32> to memref<1x5x3xf32>
  return
}

// CHECK-LABEL: func.func @gemm_fusion_to_f16_split_k
// CHECK-SAME: (%[[a:.*]]: memref<1x5x4xf32>, %[[b:.*]]: memref<1x4x3xf32>, %[[c:.*]]: memref<1x5x3xf32>, %[[d:.*]]: memref<1x5x3xf16> {rock.prefill = 0.000000e+00 : f16})
// CHECK-SAME: grid_size = 3 : i32
func.func @gemm_fusion_to_f16_split_k(%arg0: memref<1x5x4xf32>, %arg1: memref<1x4x3xf32>, %arg2: memref<1x5x3xf32>, %arg3: memref<1x5x3xf16>) attributes {arch = "amdgcn-amd-amdhsa:gfx908"} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5x3xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %arg0 * %arg1 storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x5x3xf32> = memref<1x5x4xf32> * memref<1x4x3xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x5x3xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %arg2 : memref<1x5x3xf32>, memref<1x5x3xf32>) outs(%alloc_0 : memref<1x5x3xf16>) {
  ^bb0(%in: f32, %in_1: f32, %out: f16):
    %7 = arith.addf %in, %in_1 : f32
    %8 = arith.truncf %7 : f32 to f16
    linalg.yield %8 : f16
  }
  memref.copy %alloc_0, %arg3 : memref<1x5x3xf16> to memref<1x5x3xf16>
  return
}

// CHECK-LABEL: func.func @rock_attention_simple
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x1024x64xf32>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 32 : i32
func.func @rock_attention_simple(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]])
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     %arg3 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } { 
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_tr_padded
// CHECK-SAME: (%[[q:.*]]: memref<1x49x7xf32>, %[[k:.*]]: memref<1x7x49xf32>, %[[v:.*]]: memref<1x49x7xf32>, %[[o:.*]]: memref<1x49x7xf32>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 2 : i32
func.func @rock_attention_tr_padded(%arg0: memref<1x49x7xf32>, %arg1: memref<1x7x49xf32>, %arg2: memref<1x49x7xf32>, %arg3: memref<1x49x7xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK-DAG: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : memref<1x49x7xf32> to memref<1x7x49xf32>
  // CHECK-DAG: %[[paddedTrQ:.*]] = rock.transform %[[trQ]] by {{.*}} : memref<1x7x49xf32> to memref<1x8x64xf32>
  // CHECK-DAG: %[[paddedK:.*]] = rock.transform %[[k]] by {{.*}} : memref<1x7x49xf32> to memref<1x8x64xf32>
  // CHECK-DAG: %[[paddedV:.*]] = rock.transform %[[v]] by {{.*}} : memref<1x49x7xf32> to memref<1x64x32xf32>
  // CHECK-DAG: %[[paddedO:.*]] = rock.transform %[[o]] by {{.*}} : memref<1x49x7xf32> to memref<1x64x32xf32>
  // CHECK: rock.gridwise_attention_accel(%[[paddedTrQ]], %[[paddedK]], %[[paddedV]], %[[paddedO]])
  // CHECK-NEXT: prePadG0M = 49 : index, prePadG0N = 49 : index
  rock.attention{
    qk = %arg0 * %arg1 : memref<1x49x7xf32>, memref<1x7x49xf32>
    %arg3 = softmax(qk) * %arg2 : memref<1x49x7xf32> -> memref<1x49x7xf32>
  } { 
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_kvcache
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x1024x64xf32>, %[[currentSeqLen:.*]]: memref<1xi32>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 32 : i32
func.func @rock_attention_kvcache(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>, %arg4: memref<1xi32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[currentSeqLen]], %[[o]])
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     currentSeqLen = (%arg4 : memref<1xi32>)
     %arg3 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_causal
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x1024x64xf32>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 32 : i32
func.func @rock_attention_causal(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]])
  // CHECK-NEXT: , causal,
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     causal
     %arg3 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_lse
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[lse:.*]]: memref<1x1024xf32>, %[[o:.*]]: memref<1x1024x64xf32>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 32 : i32
func.func @rock_attention_lse(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024xf32>, %arg4: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]], %[[lse]])
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     lse = %arg3 : memref<1x1024xf32>
     %arg4 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32,
    storeMethod = #rock<StoreMethod set>,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_splitkv
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[lse:.*]]: memref<4x1024xf32>, %[[o:.*]]: memref<4x1024x64xf32>)
// CHECK-SAME: grid_size = 128
func.func @rock_attention_splitkv(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<4x1024xf32>, %arg4: memref<4x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]], %[[lse]])
  // CHECK-NEXT: splitKV = 4
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     lse = %arg3 : memref<4x1024xf32>
     %arg4 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<4x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 4 : i32,
    storeMethod = #rock<StoreMethod set>,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_splitkv_padding
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x384xf32>, %[[v:.*]]: memref<1x384x64xf32>, %[[lse:.*]]: memref<8x1024xf32>, %[[o:.*]]: memref<8x1024x64xf32>)
// CHECK-SAME: grid_size = 256
func.func @rock_attention_splitkv_padding(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<8x1024xf32>, %arg4: memref<8x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK-DAG: %[[kPadding:.*]] = rock.transform %[[k]] by {{.*}} : memref<1x64x384xf32> to memref<1x64x512xf32>
  // CHECK-DAG: %[[vPadding:.*]] = rock.transform %[[v]] by {{.*}} : memref<1x384x64xf32> to memref<1x512x64xf32>
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[kPadding]], %[[vPadding]], %[[o]], %[[lse]])
  // CHECK-NEXT: splitKV = 8
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x384xf32>
     lse = %arg3 : memref<8x1024xf32>
     %arg4 = softmax(qk) * %arg2 : memref<1x384x64xf32> -> memref<8x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 8 : i32,
    storeMethod = #rock<StoreMethod set>,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_softmaxtype
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf16>, %[[k:.*]]: memref<1x64x1024xf16>, %[[v:.*]]: memref<1x1024x64xf16>, %[[lse:.*]]: memref<1x1024xf16>, %[[o:.*]]: memref<1x1024x64xf16>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 32 : i32
func.func @rock_attention_softmaxtype(%arg0: memref<1x64x1024xf16>, %arg1: memref<1x64x1024xf16>, %arg2: memref<1x1024x64xf16>, %arg3: memref<1x1024xf16>, %arg4: memref<1x1024x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]], %[[lse]])
  // CHECK: softmaxType = f32
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf16>, memref<1x64x1024xf16>
     lse = %arg3 : memref<1x1024xf16>
     %arg4 = softmax(qk) * %arg2 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    storeMethod = #rock<StoreMethod set>,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32,
    softmaxType = f32
  }
  return
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_simple
// CHECK-SAME: (%[[a:.*]]: memref<1x64x1024xf32>, %[[b:.*]]: memref<1x64x1024xf32>, %[[c:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x1024x64xf32>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 32 : i32
func.func @rock_gemmelementwisegemm_simple(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[a]], %[[b]], %[[c]], %[[o]])
  // CHECK-NEXT: enableSoftmax = false
  rock.gemm_elementwise_gemm{
     ab = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     %arg3 = ab * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } { 
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    storeMethod = #rock<StoreMethod set>
  }
  return
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_tr_padded
// CHECK-SAME: (%[[a:.*]]: memref<1x49x7xf32>, %[[b:.*]]: memref<1x7x49xf32>, %[[c:.*]]: memref<1x49x7xf32>, %[[o:.*]]: memref<1x49x7xf32>)
// CHECK-SAME: block_size = 64 : i32, grid_size = 2 : i32
func.func @rock_gemmelementwisegemm_tr_padded(%arg0: memref<1x49x7xf32>, %arg1: memref<1x7x49xf32>, %arg2: memref<1x49x7xf32>, %arg3: memref<1x49x7xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK-DAG: %[[trA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x49x7xf32> to memref<1x7x49xf32>
  // CHECK-DAG: %[[paddedTrA:.*]] = rock.transform %[[trA]] by {{.*}} : memref<1x7x49xf32> to memref<1x8x64xf32>
  // CHECK-DAG: %[[paddedB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x7x49xf32> to memref<1x8x64xf32>
  // CHECK-DAG: %[[paddedC:.*]] = rock.transform %[[c]] by {{.*}} : memref<1x49x7xf32> to memref<1x64x32xf32>
  // CHECK-DAG: %[[paddedO:.*]] = rock.transform %[[o]] by {{.*}} : memref<1x49x7xf32> to memref<1x64x32xf32>
  // CHECK: rock.gridwise_attention_accel(%[[paddedTrA]], %[[paddedB]], %[[paddedC]], %[[paddedO]])
  // CHECK-NEXT: enableSoftmax = false
  // CHECK-SAME: prePadG0M = 49 : index, prePadG0N = 49 : index
  rock.gemm_elementwise_gemm{
    ab = %arg0 * %arg1 : memref<1x49x7xf32>, memref<1x7x49xf32>
    %arg3 = ab * %arg2 : memref<1x49x7xf32> -> memref<1x49x7xf32>
  } { 
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    storeMethod = #rock<StoreMethod set>
  }
  return
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_splitk
// CHECK-SAME: (%[[aRaw:.*]]: memref<1x64x1024xf32>, %[[bRaw:.*]]: memref<1x64x1024xf32>, %[[cRaw:.*]]: memref<1x1024x64xf32>, %[[oRaw:.*]]: memref<1x1024x64xf32>
// CHECK-SAME: {rock.prefill = 0.000000e+00 : f32})
// CHECK-SAME: block_size = 64 : i32, grid_size = 128 : i32
func.func @rock_gemmelementwisegemm_splitk(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  // CHECK-DAG: %[[bSplit:.*]] = rock.transform %[[bRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d3, d1 * 256 + d2)> by [<PassThrough ["gemmG", "gemmK"] at [0, 3] -> ["gemmG", "gemmK"] at [0, 1]>, <Unmerge{4, 256} ["gemmNSplit", "gemmN"] at [1, 2] -> ["gemmN"] at [2]>] bounds = [1, 4, 256, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x4x256x64xf32>
  // CHECK-DAG: %[[b:.*]] = rock.transform %[[bSplit]] by <affine_map<(d0, d1, d2) -> (0, d0, d2, d1)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>, <PassThrough ["gemmN", "gemmK"] at [2, 1] -> ["gemmN", "gemmK"] at [2, 3]>] bounds = [4, 64, 256] -> [1, 4, 256, 64]> : memref<1x4x256x64xf32> to memref<4x64x256xf32>
  // CHECK-DAG: %[[cSplit:.*]] = rock.transform %[[cRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["gemmG", "gemmO"] at [0, 3] -> ["gemmG", "gemmO"] at [0, 2]>, <Unmerge{4, 256} ["gemmNSplit", "gemmN"] at [1, 2] -> ["gemmN"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  // CHECK-DAG: %[[c:.*]] = rock.transform %[[cSplit]] by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>, <PassThrough ["gemmN", "gemmO"] at [1, 2] -> ["gemmN", "gemmO"] at [2, 3]>] bounds = [4, 256, 64] -> [1, 4, 256, 64]> : memref<1x4x256x64xf32> to memref<4x256x64xf32>
  // CHECK-DAG: %[[aSplit:.*]] = rock.transform %[[aRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)> by [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 1, 2] -> ["gemmG", "gemmK", "gemmM"] at [0, 1, 2]>, <AddDim{4} ["gemmNSplit"] at [3] -> [] at []>] bounds = [1, 64, 1024, 4] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x64x1024x4xf32>
  // CHECK-DAG: %[[a:.*]] = rock.transform %[[aSplit]] by <affine_map<(d0, d1, d2) -> (0, d1, d2, d0)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 3]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [1, 2]>] bounds = [4, 64, 1024] -> [1, 64, 1024, 4]> : memref<1x64x1024x4xf32> to memref<4x64x1024xf32>
  // CHECK-DAG: %[[oSplit:.*]] = rock.transform %[[oRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)> by [<AddDim{4} ["gemmNSplit"] at [1] -> [] at []>, <PassThrough ["gemmG", "gemmM", "gemmO"] at [0, 2, 3] -> ["gemmG", "gemmM", "gemmO"] at [0, 1, 2]>] bounds = [1, 4, 1024, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x4x1024x64xf32>
  // CHECK-DAG: %[[o:.*]] = rock.transform %[[oSplit]] by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>, <PassThrough ["gemmM", "gemmO"] at [1, 2] -> ["gemmM", "gemmO"] at [2, 3]>] bounds = [4, 1024, 64] -> [1, 4, 1024, 64]> : memref<1x4x1024x64xf32> to memref<4x1024x64xf32>
  // CHECK: rock.gridwise_attention_accel(%[[a]], %[[b]], %[[c]], %[[o]])
  // CHECK-NEXT: enableSoftmax = false
  // CHECK-SAME: gridSize = 128 : i32
  // CHECK-SAME: storeMethod = #rock<StoreMethod atomic_add>
  rock.gemm_elementwise_gemm{
     ab = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     %arg3 = ab * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } { 
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1_splitk,
    firstGemmIndices = array<i64: 0>,
    storeMethod = #rock<StoreMethod set>
  }
  return
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_splitk_two_outputs
// CHECK-SAME: (%[[aRaw:.*]]: memref<4096xf32>, %[[bRaw:.*]]: memref<4096xf32>, %[[cRaw:.*]]: memref<4096xf32>, %[[oRaw:.*]]: memref<4096xf32> {rock.prefill = 0.000000e+00 : f32},
// CHECK-SAME: %[[reduceOut:.*]]: memref<64xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-SAME: block_size = 64 : i32, grid_size = 8 : i32
func.func @rock_gemmelementwisegemm_splitk_two_outputs(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>, %arg3: memref<4096xf32>, %arg4: memref<64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32} {
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf32> to memref<1x64x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf32> to memref<1x64x64xf32>
  %2 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf32> to memref<1x64x64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x64x64xf32>

  // CHECK-DAG: %[[aReshaped:.*]] = rock.transform %[[aRaw]] {{.*}} memref<4096xf32> to memref<1x64x64xf32>
  // CHECK-DAG: %[[bReshaped:.*]] = rock.transform %[[bRaw]] {{.*}} memref<4096xf32> to memref<1x64x64xf32>
  // CHECK-DAG: %[[cReshaped:.*]] = rock.transform %[[cRaw]] {{.*}} memref<4096xf32> to memref<1x64x64xf32>

  // CHECK-DAG: %[[gemmOut:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x64x64xf32>

  // CHECK-DAG: %[[bSplit:.*]] = rock.transform %[[bReshaped]] {{.*}} memref<1x64x64xf32> to memref<1x4x16x64xf32>
  // CHECK-DAG: %[[b:.*]] = rock.transform %[[bSplit]] {{.*}} memref<1x4x16x64xf32> to memref<4x64x16xf32>
  // CHECK-DAG: %[[bPad:.*]] = rock.transform %[[b]] {{.*}} memref<4x64x16xf32> to memref<4x64x64xf32>

  // CHECK-DAG: %[[cSplit:.*]] = rock.transform %[[cReshaped]] {{.*}} memref<1x64x64xf32> to memref<1x4x16x64xf32>
  // CHECK-DAG: %[[c:.*]] = rock.transform %[[cSplit]] {{.*}} memref<1x4x16x64xf32> to memref<4x16x64xf32>
  // CHECK-DAG: %[[cPad:.*]] = rock.transform %[[c]] {{.*}} memref<4x16x64xf32> to memref<4x64x128xf32>

  // CHECK-DAG: %[[aSplit:.*]] = rock.transform %[[aReshaped]] {{.*}} memref<1x64x64xf32> to memref<1x64x64xf32>
  // CHECK-DAG: %[[a:.*]] = rock.transform %[[aSplit]] {{.*}} memref<1x64x64xf32> to memref<1x64x64x4xf32>
  // CHECK-DAG: %[[aPad:.*]] = rock.transform %[[a]] {{.*}} memref<1x64x64x4xf32> to memref<4x64x64xf32>

  // CHECK-DAG: %[[oSplit:.*]] = rock.transform %[[gemmOut]] {{.*}} memref<1x64x64xf32> to memref<1x4x64x64xf32>
  // CHECK-DAG: %[[o:.*]] = rock.transform %[[oSplit]] {{.*}} memref<1x4x64x64xf32> to memref<4x64x64xf32>
  // CHECK-DAG: %[[oPad:.*]] = rock.transform %[[o]] {{.*}} memref<4x64x64xf32> to memref<4x64x128xf32>

  // CHECK: rock.gridwise_attention_accel(%[[aPad]], %[[bPad]], %[[cPad]], %[[oPad]])
  // CHECK-NEXT: enableSoftmax = false
  // CHECK-SAME: gridSize = 8 : i32
  // CHECK-SAME: storeMethod = #rock<StoreMethod atomic_add>
  rock.gemm_elementwise_gemm{
   ab = %2 * %1 : memref<1x64x64xf32>, memref<1x64x64xf32>
   ab = elementwise {
  ^bb0(%arg5: memref<1x64x64xf32>, %arg6: memref<1x64x64xf32>):
    memref.copy %arg5, %arg6 : memref<1x64x64xf32> to memref<1x64x64xf32>
    rock.yield
  }
   %alloc = ab * %0 : memref<1x64x64xf32> -> memref<1x64x64xf32>
  } {firstGemmIndices = array<i64: 0>, params0 = #rock.mfma_gemm_params<kpackPerBlock = 16, mPerBlock = 64, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>, params1 = #rock.mfma_gemm_params<kpackPerBlock = 16, mPerBlock = 128, nPerBlock = 32, kpack = 4, mPerWave = 64, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>, perf_config = "attn:v2:64,128,32,16,32,16,4,4,1,2,1", storeMethod = #rock<StoreMethod set>}
  %3 = rock.transform %alloc by <affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)> by [<Merge{1, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [4096] -> [1, 64, 64]> : memref<1x64x64xf32> to memref<4096xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x64x1xf32>

  // CHECK-DAG: %[[outCopy:.*]] = rock.transform %[[gemmOut]] {{.*}} memref<1x64x64xf32> to memref<4096xf32>
  // CHECK-DAG: %[[allocReduce:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x64x1xf32>
  // CHECK-DAG: rock.reduce  sum %[[gemmOut]] into %[[allocReduce]] {axis = 2 : index, blockSize = 256 : i32, gridSize = 16 : i32} : memref<1x64x64xf32> into memref<1x64x1xf32>

  // CHECK-DAG: %[[reduceCopy:.*]] = rock.transform %[[allocReduce]] by <affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 64, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [1, 64, 1]> : memref<1x64x1xf32> to memref<64xf32>
  // CHECK-DAG: memref.copy %[[outCopy]], %[[oRaw]] : memref<4096xf32> to memref<4096xf32>
  // CHECK-DAG: memref.copy %[[reduceCopy]], %[[reduceOut]] : memref<64xf32> to memref<64xf32>
  
  rock.reduce  sum %alloc into %alloc_0 {axis = 2 : index, blockSize = 256 : i32, gridSize = 16 : i32} : memref<1x64x64xf32> into memref<1x64x1xf32>
  %4 = rock.transform %alloc_0 by <affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 64, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [1, 64, 1]> : memref<1x64x1xf32> to memref<64xf32>
  memref.copy %3, %arg3 : memref<4096xf32> to memref<4096xf32>
  memref.copy %4, %arg4 : memref<64xf32> to memref<64xf32>
  return
}

// CHECK-LABEL: func.func @rock_attention_gqa
// CHECK-SAME: (%[[q:.*]]: memref<64x1x128xf16>, %[[k:.*]]: memref<8x128x8192xf16>, %[[v:.*]]: memref<8x8192x128xf16>, %[[lse:.*]]: memref<256x1xf16>, %[[o:.*]]: memref<256x1x128xf16>)
// CHECK-SAME: grid_size = 32
func.func @rock_attention_gqa(%arg0: memref<64x1x128xf16>, %arg1: memref<8x128x8192xf16>, %arg2: memref<8x8192x128xf16>, %arg3: memref<256x1xf16>, %arg4: memref<256x1x128xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK-DAG: %[[qNormalized:.*]] = rock.transform %[[q]] by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by {{.*}} memref<64x1x128xf16> to memref<64x128x1xf16>
  // CHECK-DAG: %[[qExtractNumRepeats:.+]] = rock.transform %[[qNormalized]] by <affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d3, d1, d2)> by {{.*}} memref<64x128x1xf16> to memref<8x128x1x8xf16>
  // CHECK-DAG: %[[qMoveToSeqLen:.*]] = rock.transform %[[qExtractNumRepeats]] by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by {{.*}} memref<8x128x1x8xf16> to memref<8x128x8xf16>
  // CHECK-DAG: %[[qPad:.+]] = rock.transform %[[qMoveToSeqLen]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by {{.*}} memref<8x128x8xf16> to memref<8x128x32xf16>
  
  // CHECK-DAG: %[[outUnmerge:.*]] = rock.transform %[[o]] by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 8 + d3) * 4 + d1, d2, d4)> by {{.*}} memref<256x1x128xf16> to memref<8x4x1x8x128xf16>
  // CHECK-DAG: %[[outMerge:.*]] = rock.transform %[[outUnmerge]] by <affine_map<(d0, d1, d2) -> (d0 floordiv 4, d0 mod 4, 0, d1, d2)> by {{.*}} memref<8x4x1x8x128xf16> to memref<32x8x128xf16>
  // CHECK-DAG: %[[outPad:.*]] = rock.transform %[[outMerge]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by {{.*}} memref<32x8x128xf16> to memref<32x32x128xf16>
  
  // CHECK-DAG: %[[lseUnmerge:.*]] = rock.transform %[[lse]] by <affine_map<(d0, d1, d2, d3) -> ((d0 * 8 + d3) * 4 + d1, d2)> by {{.*}} memref<256x1xf16> to memref<8x4x1x8xf16>
  // CHECK-DAG: %[[lseMerge:.*]] = rock.transform %[[lseUnmerge]] by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, 0, d1)> by {{.*}} memref<8x4x1x8xf16> to memref<32x8xf16>
  // CHECK-DAG: %[[lsePad:.*]] = rock.transform %[[lseMerge]] by <affine_map<(d0, d1) -> (d0, d1)> by {{.*}} memref<32x8xf16> to memref<32x32xf16>

  // CHECK: rock.gridwise_attention_accel(%[[qPad]], %[[k]], %[[v]], %[[outPad]], %[[lsePad]])
  // CHECK-NEXT: splitKV = 4
  rock.attention{
     qk = %arg0 * %arg1 : memref<64x1x128xf16>, memref<8x128x8192xf16>
     lse = %arg3 : memref<256x1xf16>
     qk = elementwise {
    ^bb0(%arg5: memref<64x1x8192xf16>, %arg6: memref<64x1x8192xf16>):
      memref.copy %arg5, %arg6 : memref<64x1x8192xf16> to memref<64x1x8192xf16>
      rock.yield
    }
     %arg4 = softmax(qk) * %arg2 : memref<8x8192x128xf16> -> memref<256x1x128xf16>
  } {features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_fmax_f32>, firstGemmIndices = array<i64: 0>, numHeadsKV = 8 : i32, numHeadsQ = 64 : i32, params0 = #rock.wmma_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll  = true>, params1 = #rock.wmma_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll  = true>, softmaxType = f32, splitKV = 4 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

// -----

// Tests for scaled GEMM 

// CHECK-LABEL: func.func @gemm_scaled_fp4_already_f8e8m0
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf4E2M1FN>, %[[b:.*]]: memref<1x72x512xf4E2M1FN>, %[[c:.*]]: memref<1x128x512xf32>, %[[scaleA:.*]]: memref<1x128x72xf8E8M0FNU>, %[[scaleB:.*]]: memref<1x72x512xf8E8M0FNU>)
// CHECK-SAME: grid_size = 16 : i32
func.func @gemm_scaled_fp4_already_f8e8m0(%a: memref<1x72x128xf4E2M1FN>, %b: memref<1x72x512xf4E2M1FN>, %c: memref<1x128x512xf32>, %scaleA: memref<1x128x72xf8E8M0FNU>, %scaleB: memref<1x72x512xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %[[normalizeScaleA:.*]] = rock.transform %[[scaleA]] by {{.*}} : memref<1x128x72xf8E8M0FNU> to memref<1x72x128xf8E8M0FNU{{.*}}>
  // CHECK: rock.gridwise_gemm_accel(%[[a]], %[[b]], %[[c]], %[[normalizeScaleA]], %[[scaleB]])
  rock.gemm %c = tr %a scaled by %scaleA * %b scaled by %scaleB features = mfma storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 16 : i32,
    params = #xdlops_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf4E2M1FN> scaled by memref<1x128x72xf8E8M0FNU> * memref<1x72x512xf4E2M1FN> scaled by memref<1x72x512xf8E8M0FNU>
  func.return
}

// CHECK-LABEL: func.func @gemm_scaled_fp4_with_padding
// CHECK-SAME: (%[[a:.*]]: memref<1x1x1xf4E2M1FN>, %[[b:.*]]: memref<1x1x1xf4E2M1FN>, %[[c:.*]]: memref<1x1x1xf32>, %[[scaleA:.*]]: memref<1x1x1xf8E8M0FNU>, %[[scaleB:.*]]: memref<1x1x1xf8E8M0FNU>)
// CHECK-SAME: grid_size = 1
func.func @gemm_scaled_fp4_with_padding(%a: memref<1x1x1xf4E2M1FN>, %b: memref<1x1x1xf4E2M1FN>, %c: memref<1x1x1xf32>, %scaleA: memref<1x1x1xf8E8M0FNU>, %scaleB: memref<1x1x1xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-DAG: %[[padA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x1x1xf4E2M1FN> to memref<1x8x64xf4E2M1FN{{.*}}>
  // CHECK-DAG: %[[padB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x1x1xf4E2M1FN> to memref<1x8x64xf4E2M1FN{{.*}}>
  // CHECK-DAG: %[[padC:.*]] = rock.transform %[[c]] by {{.*}} : memref<1x1x1xf32> to memref<1x64x64xf32{{.*}}>
  // CHECK-DAG: %[[padScaleA:.*]] = rock.transform %[[scaleA]] by {{.*}} : memref<1x1x1xf8E8M0FNU> to memref<1x8x64xf8E8M0FNU{{.*}}>
  // CHECK-DAG: %[[padScaleB:.*]] = rock.transform %[[scaleB]] by {{.*}} : memref<1x1x1xf8E8M0FNU> to memref<1x8x64xf8E8M0FNU{{.*}}>
  // CHECK: rock.gridwise_gemm_accel(%[[padA]], %[[padB]], %[[padC]], %[[padScaleA]], %[[padScaleB]])
  rock.gemm %c = tr %a scaled by %scaleA * %b scaled by %scaleB features = mfma storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #xdlops_gemm_params0
  } : memref<1x1x1xf32> = memref<1x1x1xf4E2M1FN> scaled by memref<1x1x1xf8E8M0FNU> * memref<1x1x1xf4E2M1FN> scaled by memref<1x1x1xf8E8M0FNU>
  func.return
}

// CHECK-LABEL: func.func @gemm_scaled_fp4_transposed
// CHECK-SAME: (%[[a:.*]]: memref<1x128x72xf4E2M1FN>, %[[b:.*]]: memref<1x512x72xf4E2M1FN>, %[[c:.*]]: memref<1x512x128xf32>, %[[scaleA:.*]]: memref<1x72x128xf8E8M0FNU>, %[[scaleB:.*]]: memref<1x72x512xf8E8M0FNU>)
// CHECK-SAME: grid_size = 16 : i32
func.func @gemm_scaled_fp4_transposed(%a: memref<1x128x72xf4E2M1FN>, %b: memref<1x512x72xf4E2M1FN>, %c: memref<1x512x128xf32>, %scaleA: memref<1x72x128xf8E8M0FNU>, %scaleB: memref<1x72x512xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] {{.*}} : memref<1x128x72xf4E2M1FN> to memref<1x72x128xf4E2M1FN{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] {{.*}} : memref<1x512x72xf4E2M1FN> to memref<1x72x512xf4E2M1FN{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] {{.*}} : memref<1x512x128xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm_accel(%[[normalizeA]], %[[normalizeB]], %[[normalizeC]], %[[scaleA]], %[[scaleB]])
  rock.gemm tr %c = %a scaled by tr %scaleA * tr %b scaled by %scaleB features = mfma storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 16 : i32,
    params = #xdlops_gemm_params0
  } : memref<1x512x128xf32> = memref<1x128x72xf4E2M1FN> scaled by memref<1x72x128xf8E8M0FNU> * memref<1x512x72xf4E2M1FN> scaled by memref<1x72x512xf8E8M0FNU>
  func.return
}

// -----

// CHECK-LABEL: func.func @gemm_scaled_fp4_with_f32_scales
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf4E2M1FN>, %[[b:.*]]: memref<1x72x512xf4E2M1FN>, %[[c:.*]]: memref<1x128x512xf32>, %[[scaleA:.*]]: memref<1x128x72xf32>, %[[scaleB:.*]]: memref<1x72x512xf32>)
// CHECK-SAME: grid_size = 16 : i32
func.func @gemm_scaled_fp4_with_f32_scales(%a: memref<1x72x128xf4E2M1FN>, %b: memref<1x72x512xf4E2M1FN>, %c: memref<1x128x512xf32>, %scaleA: memref<1x128x72xf32>, %scaleB: memref<1x72x512xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %[[normalizeScaleA:.*]] = rock.transform %[[scaleA]] by {{.*}} : memref<1x128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK: %[[allocScaleA:.*]] = memref.alloc() : memref<1x72x128xf8E8M0FNU>
  // CHECK: linalg.generic {{{.*}}} ins(%[[normalizeScaleA]] : memref<1x72x128xf32{{.*}}>) outs(%[[allocScaleA]] : memref<1x72x128xf8E8M0FNU>)
  // CHECK: %[[allocScaleB:.*]] = memref.alloc() : memref<1x72x512xf8E8M0FNU>
  // CHECK: linalg.generic {{{.*}}} ins(%[[scaleB]] : memref<1x72x512xf32>) outs(%[[allocScaleB]] : memref<1x72x512xf8E8M0FNU>)
  // CHECK: rock.gridwise_gemm_accel(%[[a]], %[[b]], %[[c]], %[[allocScaleA]], %[[allocScaleB]])
  rock.gemm %c = tr %a scaled by %scaleA * %b scaled by %scaleB features = mfma storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 16 : i32,
    params = #xdlops_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf4E2M1FN> scaled by memref<1x128x72xf32> * memref<1x72x512xf4E2M1FN> scaled by memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_scaled_fp4_splitk
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf4E2M1FN>, %[[b:.*]]: memref<1x72x512xf4E2M1FN>, %[[c:.*]]: memref<1x128x512xf32> {rock.prefill = 0.000000e+00 : f32}, %[[scaleA:.*]]: memref<1x128x72xf8E8M0FNU>, %[[scaleB:.*]]: memref<1x72x512xf8E8M0FNU>)
// CHECK-SAME: grid_size = 32 : i32
func.func @gemm_scaled_fp4_splitk(%a: memref<1x72x128xf4E2M1FN>, %b: memref<1x72x512xf4E2M1FN>, %c: memref<1x128x512xf32>, %scaleA: memref<1x128x72xf8E8M0FNU>, %scaleB: memref<1x72x512xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // Transpose scaleA from MxK to KxM
  // CHECK: rock.transform %[[scaleA]] by {{.*}} : memref<1x128x72xf8E8M0FNU> to memref<1x72x128xf8E8M0FNU>
  
  // Padding K from 72 to 96
  // CHECK-DAG: rock.transform %[[a]] by {{.*}} : memref<1x72x128xf4E2M1FN> to memref<1x96x128xf4E2M1FN>
  // CHECK-DAG: rock.transform %[[b]] by {{.*}} : memref<1x72x512xf4E2M1FN> to memref<1x96x512xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x72x128xf8E8M0FNU> to memref<1x96x128xf8E8M0FNU>
  // CHECK-DAG: rock.transform %[[scaleB]] by {{.*}} : memref<1x72x512xf8E8M0FNU> to memref<1x96x512xf8E8M0FNU>
  
  // Split K into 2 parts (96/2 = 48 per split)
  // CHECK-DAG: rock.transform {{.*}} : memref<1x96x128xf4E2M1FN> to memref<1x2x48x128xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x2x48x128xf4E2M1FN> to memref<2x48x128xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x96x512xf4E2M1FN> to memref<1x2x48x512xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x2x48x512xf4E2M1FN> to memref<2x48x512xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x96x128xf8E8M0FNU> to memref<1x2x48x128xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x2x48x128xf8E8M0FNU> to memref<2x48x128xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x96x512xf8E8M0FNU> to memref<1x2x48x512xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x2x48x512xf8E8M0FNU> to memref<2x48x512xf8E8M0FNU>
  
  // Split and merge C
  // CHECK-DAG: rock.transform %[[c]] by {{.*}} : memref<1x128x512xf32> to memref<1x2x128x512xf32>
  // CHECK-DAG: rock.transform {{.*}} : memref<1x2x128x512xf32> to memref<2x128x512xf32>
  
  // CHECK: rock.gridwise_gemm_accel({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) storeMethod( atomic_add) features =  mfma {{.*}} : memref<2x48x128xf4E2M1FN>, memref<2x48x512xf4E2M1FN>, memref<2x128x512xf32>, memref<2x48x128xf8E8M0FNU>, memref<2x48x512xf8E8M0FNU>
  rock.gemm %c = tr %a scaled by %scaleA * %b scaled by %scaleB features = mfma storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 16 : i32,
    params = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 2, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>
  } : memref<1x128x512xf32> = memref<1x72x128xf4E2M1FN> scaled by memref<1x128x72xf8E8M0FNU> * memref<1x72x512xf4E2M1FN> scaled by memref<1x72x512xf8E8M0FNU>
  func.return
}

// CHECK-LABEL: func.func @gemm_scaled_fp4_splitk_odd
// CHECK-SAME: (%[[aRaw:.*]]: memref<589824xf4E2M1FN>, %[[bRaw:.*]]: memref<589824xf4E2M1FN>, %[[cRaw:.*]]: memref<196608xf32> {rock.prefill = 0.000000e+00 : f32}, %[[scaleARaw:.*]]: memref<18432xf8E8M0FNU>, %[[scaleBRaw:.*]]: memref<18432xf8E8M0FNU>)
// CHECK-SAME: grid_size = 240 : i32
func.func @gemm_scaled_fp4_splitk_odd(%arg0: memref<589824xf4E2M1FN>, %arg1: memref<589824xf4E2M1FN>, %arg2: memref<196608xf32>, %arg3: memref<18432xf8E8M0FNU>, %arg4: memref<18432xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK-DAG: rock.transform %[[aRaw]] by {{.*}} : memref<589824xf4E2M1FN> to memref<3x256x768xf4E2M1FN>
  // CHECK-DAG: rock.transform %[[bRaw]] by {{.*}} : memref<589824xf4E2M1FN> to memref<3x768x256xf4E2M1FN>
  // CHECK-DAG: rock.transform %[[cRaw]] by {{.*}} : memref<196608xf32> to memref<3x256x256xf32>
  // CHECK-DAG: rock.transform %[[scaleARaw]] by {{.*}} : memref<18432xf8E8M0FNU> to memref<3x256x24xf8E8M0FNU>
  // CHECK-DAG: rock.transform %[[scaleBRaw]] by {{.*}} : memref<18432xf8E8M0FNU> to memref<3x24x256xf8E8M0FNU>
  
  // Scale broadcasting through AddDim, Broadcast, and Merge transformations
  // CHECK-DAG: rock.transform {{.*}} : memref<3x256x24xf8E8M0FNU> to memref<3x256x24x1xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x256x24x1xf8E8M0FNU> to memref<3x256x24x32xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x256x24x32xf8E8M0FNU> to memref<3x256x768xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x24x256xf8E8M0FNU> to memref<3x24x1x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x24x1x256xf8E8M0FNU> to memref<3x24x32x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x24x32x256xf8E8M0FNU> to memref<3x768x256xf8E8M0FNU>
  
  // Transpose A and scaleA from MxK to KxM
  // CHECK-DAG: rock.transform {{.*}} : memref<3x256x768xf4E2M1FN> to memref<3x768x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x256x768xf8E8M0FNU> to memref<3x768x256xf8E8M0FNU>
  
  // Padding K from 768 to 800
  // CHECK-DAG: rock.transform {{.*}} : memref<3x768x256xf4E2M1FN> to memref<3x800x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x768x256xf4E2M1FN> to memref<3x800x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x768x256xf8E8M0FNU> to memref<3x800x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x768x256xf8E8M0FNU> to memref<3x800x256xf8E8M0FNU>
  
  // Split K into 5 parts (800/5 = 160 per split)
  // CHECK-DAG: rock.transform {{.*}} : memref<3x800x256xf4E2M1FN> to memref<3x5x160x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x5x160x256xf4E2M1FN> to memref<15x160x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x800x256xf4E2M1FN> to memref<3x5x160x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x5x160x256xf4E2M1FN> to memref<15x160x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x800x256xf8E8M0FNU> to memref<3x5x160x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x5x160x256xf8E8M0FNU> to memref<15x160x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x800x256xf8E8M0FNU> to memref<3x5x160x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x5x160x256xf8E8M0FNU> to memref<15x160x256xf8E8M0FNU>
  
  // Split and merge C
  // CHECK-DAG: rock.transform {{.*}} : memref<3x256x256xf32> to memref<3x5x256x256xf32>
  // CHECK-DAG: rock.transform {{.*}} : memref<3x5x256x256xf32> to memref<15x256x256xf32>
  
  // Final padding K from 160 to 512
  // CHECK-DAG: rock.transform {{.*}} : memref<15x160x256xf4E2M1FN> to memref<15x512x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<15x160x256xf4E2M1FN> to memref<15x512x256xf4E2M1FN>
  // CHECK-DAG: rock.transform {{.*}} : memref<15x160x256xf8E8M0FNU> to memref<15x512x256xf8E8M0FNU>
  // CHECK-DAG: rock.transform {{.*}} : memref<15x160x256xf8E8M0FNU> to memref<15x512x256xf8E8M0FNU>
  
  // CHECK: rock.gridwise_gemm_accel({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) storeMethod( atomic_add) features =  mfma {{.*}} : memref<15x512x256xf4E2M1FN>, memref<15x512x256xf4E2M1FN>, memref<15x256x256xf32>, memref<15x512x256xf8E8M0FNU>, memref<15x512x256xf8E8M0FNU>
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 768 + d2)> by [<Unmerge{3, 256, 768} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 768] -> [589824]> : memref<589824xf4E2M1FN> to memref<3x256x768xf4E2M1FN>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 256 + d2)> by [<Unmerge{3, 768, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 256] -> [589824]> : memref<589824xf4E2M1FN> to memref<3x768x256xf4E2M1FN>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 256 + d2)> by [<Unmerge{3, 256, 256} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 256] -> [196608]> : memref<196608xf32> to memref<3x256x256xf32>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 24 + d2)> by [<Unmerge{3, 256, 24} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 24] -> [18432]> : memref<18432xf8E8M0FNU> to memref<3x256x24xf8E8M0FNU>
  %4 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> ((d0 * 24 + d1) * 256 + d2)> by [<Unmerge{3, 24, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 24, 256] -> [18432]> : memref<18432xf8E8M0FNU> to memref<3x24x256xf8E8M0FNU>
  %5 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)> by [<AddDim{1} ["block"] at [3] -> [] at []>, <PassThrough ["g", "m", "kScale"] at [0, 1, 2] -> ["g", "m", "kScale"] at [0, 1, 2]>] bounds = [3, 256, 24, 1] -> [3, 256, 24]> : memref<3x256x24xf8E8M0FNU> to memref<3x256x24x1xf8E8M0FNU>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)> by [<Broadcast{1} ["block"] at [3] -> ["block"] at [3]>, <PassThrough ["g", "m", "kScale"] at [0, 1, 2] -> ["g", "m", "kScale"] at [0, 1, 2]>] bounds = [3, 256, 24, 32] -> [3, 256, 24, 1]> : memref<3x256x24x1xf8E8M0FNU> to memref<3x256x24x32xf8E8M0FNU>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 32, d2 mod 32)> by [<Merge{24, 32} ["k"] at [2] -> ["kScale", "block"] at [2, 3]>, <PassThrough ["g", "m"] at [0, 1] -> ["g", "m"] at [0, 1]>] bounds = [3, 256, 768] -> [3, 256, 24, 32]> : memref<3x256x24x32xf8E8M0FNU> to memref<3x256x768xf8E8M0FNU>
  %8 = rock.transform %4 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)> by [<AddDim{1} ["block"] at [2] -> [] at []>, <PassThrough ["g", "kScale", "n"] at [0, 1, 3] -> ["g", "kScale", "n"] at [0, 1, 2]>] bounds = [3, 24, 1, 256] -> [3, 24, 256]> : memref<3x24x256xf8E8M0FNU> to memref<3x24x1x256xf8E8M0FNU>
  %9 = rock.transform %8 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)> by [<Broadcast{1} ["block"] at [2] -> ["block"] at [2]>, <PassThrough ["g", "kScale", "n"] at [0, 1, 3] -> ["g", "kScale", "n"] at [0, 1, 3]>] bounds = [3, 24, 32, 256] -> [3, 24, 1, 256]> : memref<3x24x1x256xf8E8M0FNU> to memref<3x24x32x256xf8E8M0FNU>
  %10 = rock.transform %9 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 32, d1 mod 32, d2)> by [<PassThrough ["g", "n"] at [0, 2] -> ["g", "n"] at [0, 3]>, <Merge{24, 32} ["k"] at [1] -> ["kScale", "block"] at [1, 2]>] bounds = [3, 768, 256] -> [3, 24, 32, 256]> : memref<3x24x32x256xf8E8M0FNU> to memref<3x768x256xf8E8M0FNU>
  rock.gemm %2 = %0 scaled by %7 * %1 scaled by %10 features = mfma storeMethod = set {
    derivedBlockSize = 256 : i32,
    gridSize = 12 : i32,
    params = #rock.mfma_gemm_params<kpackPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kpack = 32, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 5, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0>
  } : memref<3x256x256xf32> = memref<3x256x768xf4E2M1FN> scaled by memref<3x256x768xf8E8M0FNU> * memref<3x768x256xf4E2M1FN> scaled by memref<3x768x256xf8E8M0FNU>
  func.return
}
