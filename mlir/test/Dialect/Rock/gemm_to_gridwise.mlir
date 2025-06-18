// Ensures that the padding application, group application, etc. in gemm-to-gridwise
// function as expected.

// RUN: rocmlir-opt -rock-gemm-to-gridwise %s | FileCheck %s

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#general_gemm_params_splitk = #rock.general_gemm_params<blockSize = 64, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 2, scheduleVersion = 1, outputSwizzle = 2>
#general_gemm_params1 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#xdlops_gemm_params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#xdlops_gemm_params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 64, nPerWave = 64, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#xdlops_gemm_params3 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 64, mnPerXdl = 32, forceUnroll = true, splitKFactor = 3, scheduleVersion = 1, outputSwizzle = 2>
#xldops_attn_params_g0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 1, mPerBlock = 32, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#xldops_attn_params_g1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>

// CHECK-LABEL: func.func @gemm_easy_case_from_conv
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
func.func @gemm_easy_case_from_conv(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK-NEXT: rock.gridwise_gemm %[[c]] = %[[a]] * %[[b]]
  rock.gemm %c = tr %a * %b features = none storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_splitk
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
func.func @gemm_splitk(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %c = tr %a * %b features = atomic_add storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    gridSize = 4 : i32,
    params = #general_gemm_params_splitk
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_easy_case_from_conv_xdlops
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
func.func @gemm_easy_case_from_conv_xdlops(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK-NEXT: rock.gridwise_gemm_accel(%[[a]], %[[b]], %[[c]])
  rock.gemm %c = tr %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx908",
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_most_general_padding_case
// CHECK-SAME: (%[[a:.*]]: memref<1x1x1xf32>, %[[b:.*]]: memref<1x1x1xf32>, %[[c:.*]]: memref<1x1x1xf32>)
func.func @gemm_most_general_padding_case(%a: memref<1x1x1xf32>, %b: memref<1x1x1xf32>, %c: memref<1x1x1xf32>) {
  // CHECK-DAG: %[[padA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x1x1xf32> to memref<1x16x64xf32{{.*}}>
  // CHECK-DAG: %[[padB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x1x1xf32> to memref<1x16x64xf32{{.*}}>
  // CHECK-DAG: %[[padC:.*]] = rock.transform %[[c]] by {{.*}} : memref<1x1x1xf32> to memref<1x64x64xf32{{.*}}>
  // CHECK: rock.gridwise_gemm %[[padC]] = %[[padA]] * %[[padB]]
  rock.gemm %c = tr %a * %b features = none storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    gridSize = 1 : i32,
    params = #general_gemm_params1
  } : memref<1x1x1xf32> = memref<1x1x1xf32> * memref<1x1x1xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_in_standard_form
// CHECK-SAME: (%[[a:.*]]: memref<128x72xf32>, %[[b:.*]]: memref<72x512xf32>, %[[c:.*]]: memref<128x512xf32>)
func.func @gemm_in_standard_form(%a: memref<128x72xf32>, %b: memref<72x512xf32>, %c: memref<128x512xf32>) {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] by {{.*}} : memref<128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<72x512xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] by {{.*}} : memref<128x512xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm %[[normalizeC]] = %[[normalizeA]] * %[[normalizeB]]
  rock.gemm %c = %a * %b features = none storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : memref<128x512xf32> = memref<128x72xf32> * memref<72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_transposed_from_gridwise
// CHECK-SAME: (%[[a:.*]]: memref<1x128x72xf32>, %[[b:.*]]: memref<1x512x72xf32>, %[[c:.*]]: memref<1x512x128xf32>)
func.func @gemm_transposed_from_gridwise(%a: memref<1x128x72xf32>, %b: memref<1x512x72xf32>, %c: memref<1x512x128xf32>) {
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] {{.*}} : memref<1x128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] {{.*}} : memref<1x512x72xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] {{.*}} : memref<1x512x128xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm %[[normalizeC]] = %[[normalizeA]] * %[[normalizeB]]
  rock.gemm tr %c = %a * tr %b features = none storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : memref<1x512x128xf32> = memref<1x128x72xf32> * memref<1x512x72xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_pad_for_split_k
// CHECK-SAME: (%[[a:.*]]: memref<1x128x238xf32>, %[[b:.*]]: memref<1x238x512xf32>, %[[c:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
func.func @gemm_pad_for_split_k(%a: memref<1x128x238xf32>, %b: memref<1x238x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x128x238xf32> to memref<1x238x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : memref<1x238x128xf32> to memref<1x240x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x238x512xf32> to memref<1x240x512xf32{{.*}}>
  // CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : memref<1x240x128xf32> to memref<1x3x80x128xf32{{.*}}>
  // CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : memref<1x240x512xf32> to memref<1x3x80x512xf32{{.*}}>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x128x512xf32> = memref<1x128x238xf32> * memref<1x238x512xf32>
  memref.copy %alloc, %c : memref<1x128x512xf32> to memref<1x128x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_reduce_and_split_k
// CHECK-SAME: (%[[a:.*]]: memref<1x128x238xf32>, %[[b:.*]]: memref<1x238x512xf32>, %[[c:.*]]: memref<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
func.func @gemm_reduce_and_split_k(%a: memref<1x128x238xf32>, %b: memref<1x238x512xf32>, %c: memref<1x128x1xf32>, %d: memref<1x128x512xf32>) {
  // CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x128x238xf32> to memref<1x238x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : memref<1x238x128xf32> to memref<1x240x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x238x512xf32> to memref<1x240x512xf32{{.*}}>
  // CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : memref<1x240x128xf32> to memref<1x3x80x128xf32{{.*}}>
  // CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : memref<1x240x512xf32> to memref<1x3x80x512xf32{{.*}}>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
  %alloc2 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x128x512xf32> = memref<1x128x238xf32> * memref<1x238x512xf32>
  rock.reduce sum %alloc into %alloc2 features = mfma|dot|atomic_add|atomic_add_f16 {axis = 2 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<1x128x512xf32> into memref<1x128x1xf32>
  memref.copy %alloc, %d : memref<1x128x512xf32> to memref<1x128x512xf32>
  memref.copy %alloc2, %c : memref<1x128x1xf32> to memref<1x128x1xf32>

  func.return
}

// CHECK-LABEL: func.func @gemm_reduce_and_split_k_return_reduce_directly
// CHECK-SAME: (%[[a:.*]]: memref<1x128x238xf32>, %[[b:.*]]: memref<1x238x512xf32>, %[[c:.*]]: memref<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: memref<1x128x512xf32> {rock.prefill = {{.*}} : f32})
func.func @gemm_reduce_and_split_k_return_reduce_directly(%a: memref<1x128x238xf32>, %b: memref<1x238x512xf32>, %c: memref<1x128x1xf32>, %d: memref<1x128x512xf32>) {
  // CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : memref<1x128x238xf32> to memref<1x238x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : memref<1x238x128xf32> to memref<1x240x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : memref<1x238x512xf32> to memref<1x240x512xf32{{.*}}>
  // CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : memref<1x240x128xf32> to memref<1x3x80x128xf32{{.*}}>
  // CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : memref<1x240x512xf32> to memref<1x3x80x512xf32{{.*}}>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x512xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %a * %b features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : memref<1x128x512xf32> = memref<1x128x238xf32> * memref<1x238x512xf32>
  rock.reduce sum %alloc into %c features = mfma|dot|atomic_add|atomic_add_f16 {axis = 2 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<1x128x512xf32> into memref<1x128x1xf32>
  memref.copy %alloc, %d : memref<1x128x512xf32> to memref<1x128x512xf32>

  func.return
}

// CHECK-LABEL: func.func @gemm_fusion_to_f32_split_k
// CHECK-SAME: (%[[a:.*]]: memref<1x5x4xf16>, %[[b:.*]]: memref<1x4x3xf16>, %[[c:.*]]: memref<1x5x3xf16>, %[[d:.*]]: memref<1x5x3xf32> {rock.prefill = 0.000000e+00 : f32})
func.func @gemm_fusion_to_f32_split_k(%arg0: memref<1x5x4xf16>, %arg1: memref<1x4x3xf16>, %arg2: memref<1x5x3xf16>, %arg3: memref<1x5x3xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5x3xf16>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %arg0 * %arg1 features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
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
func.func @gemm_fusion_to_f16_split_k(%arg0: memref<1x5x4xf32>, %arg1: memref<1x4x3xf32>, %arg2: memref<1x5x3xf32>, %arg3: memref<1x5x3xf16>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5x3xf32>
  // CHECK: rock.gridwise_gemm
  // CHECK-SAME: storeMethod( atomic_add)
  rock.gemm %alloc = %arg0 * %arg1 features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
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
func.func @rock_attention_simple(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]])
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     %arg3 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    arch = "amdgcn-amd-amdhsa:gfx908", 
    features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>,
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIdx = 0 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_tr_padded
// CHECK-SAME: (%[[q:.*]]: memref<1x49x7xf32>, %[[k:.*]]: memref<1x7x49xf32>, %[[v:.*]]: memref<1x49x7xf32>, %[[o:.*]]: memref<1x49x7xf32>)
func.func @rock_attention_tr_padded(%arg0: memref<1x49x7xf32>, %arg1: memref<1x7x49xf32>, %arg2: memref<1x49x7xf32>, %arg3: memref<1x49x7xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 2 : i32} {
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
    arch = "amdgcn-amd-amdhsa:gfx908", 
    features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>,
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIdx = 0 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_kvcache
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x1024x64xf32>, %[[currentSeqLen:.*]]: memref<1xi32>)
func.func @rock_attention_kvcache(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>, %arg4: memref<1xi32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[currentSeqLen]], %[[o]])
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     currentSeqLen = (%arg4 : memref<1xi32>)
     %arg3 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    arch = "amdgcn-amd-amdhsa:gfx908",
    features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>,
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIdx = 0 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_causal
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x1024x64xf32>)
func.func @rock_attention_causal(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]])
  // CHECK-NEXT: , causal,
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     causal
     %arg3 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    arch = "amdgcn-amd-amdhsa:gfx908",
    features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>,
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIdx = 0 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_attention_lse
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[lse:.*]]: memref<1x1024xf32>, %[[o:.*]]: memref<1x1024x64xf32>)
func.func @rock_attention_lse(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024xf32>, %arg4: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]], %[[lse]])
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     lse = %arg3 : memref<1x1024xf32>
     %arg4 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    arch = "amdgcn-amd-amdhsa:gfx908",
    features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>,
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIdx = 0 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_simple
// CHECK-SAME: (%[[a:.*]]: memref<1x64x1024xf32>, %[[b:.*]]: memref<1x64x1024xf32>, %[[c:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x1024x64xf32>)
func.func @rock_gemmelementwisegemm_simple(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x1024x64xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 1024 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[a]], %[[b]], %[[c]], %[[o]])
  // CHECK-NEXT: enableSoftmax = false
  rock.gemm_elementwise_gemm{
     ab = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     %arg3 = ab * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } {
    arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", 
    features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>,
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIdx = 0 : i32
  }
  return
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_tr_padded
// CHECK-SAME: (%[[a:.*]]: memref<1x49x7xf32>, %[[b:.*]]: memref<1x7x49xf32>, %[[c:.*]]: memref<1x49x7xf32>, %[[o:.*]]: memref<1x49x7xf32>)
func.func @rock_gemmelementwisegemm_tr_padded(%arg0: memref<1x49x7xf32>, %arg1: memref<1x7x49xf32>, %arg2: memref<1x49x7xf32>, %arg3: memref<1x49x7xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 64 : i32, grid_size = 2 : i32} {
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
    arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", 
    features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>,
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIdx = 0 : i32
  }
  return
}
