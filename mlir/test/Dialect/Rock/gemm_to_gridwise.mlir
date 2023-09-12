// Ensures that the padding application, group application, etc. in gemm-to-gridwise
// function as expected.

// RUN: rocmlir-opt -rock-gemm-to-gridwise %s | FileCheck %s

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1>
#general_gemm_params1 = #rock.general_gemm_params<blockSize = 64, kPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1>
#xdlops_gemm_params0 = #rock.xdlops_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32, forceUnroll = true>
#xdlops_gemm_params1 = #rock.xdlops_gemm_params<kpackPerBlock = 4, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 64, nPerWave = 64, forceUnroll = true>
#xldops_attn_params = #rock.xdlops_gemm_params<kpackPerBlock = 1, mPerBlock = 32, nPerBlock = 4, kpack = 4, mPerWave = 32, nPerWave = 32, forceUnroll = true>

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

// CHECK-LABEL: func.func @gemm_easy_case_from_conv_xdlops
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
func.func @gemm_easy_case_from_conv_xdlops(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK-NEXT: rock.gridwise_gemm_accel(%[[a]], %[[b]], %[[c]])
  rock.gemm %c = tr %a * %b features = mfma|dot|atomic_add storeMethod = set {
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

// CHECK-LABEL: func.func @rock_attention_simple
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x1024x64xf32>, %[[o:.*]]: memref<1x64x1024xf32>)
func.func @rock_attention_simple(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x1024x64xf32>, %arg3: memref<1x64x1024xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 8 : i32, grid_size = 32 : i32} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]])
  rock.attention(%arg0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    params = #xldops_attn_params,
    qTransposed
  } : memref<1x64x1024xf32>, memref<1x64x1024xf32>, memref<1x1024x64xf32>, memref<1x64x1024xf32>
  return
}

// CHECK-LABEL: func.func @rock_attention_tr_padded
// CHECK-SAME: (%[[q:.*]]: memref<1x49x7xf32>, %[[k:.*]]: memref<1x7x49xf32>, %[[v:.*]]: memref<1x49x7xf32>, %[[o:.*]]: memref<1x49x7xf32>)
func.func @rock_attention_tr_padded(%arg0: memref<1x49x7xf32>, %arg1: memref<1x7x49xf32>, %arg2: memref<1x49x7xf32>, %arg3: memref<1x49x7xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908", block_size = 8 : i32, grid_size = 2 : i32} {
  // CHECK-DAG: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : memref<1x49x7xf32> to memref<1x7x49xf32>
  // CHECK-DAG: %[[paddedTrQ:.*]] = rock.transform %[[trQ]] by {{.*}} : memref<1x7x49xf32> to memref<1x8x64xf32>
  // CHECK-DAG: %[[paddedK:.*]] = rock.transform %[[k]] by {{.*}} : memref<1x7x49xf32> to memref<1x8x52xf32>
  // CHECK-DAG: %[[paddedV:.*]] = rock.transform %[[v]] by {{.*}} : memref<1x49x7xf32> to memref<1x52x8xf32>
  // CHECK-DAG: %[[paddedO:.*]] = rock.transform %[[o]] by {{.*}} : memref<1x49x7xf32> to memref<1x64x8xf32>
  // CHECK: rock.gridwise_attention_accel(%[[paddedTrQ]], %[[paddedK]], %[[paddedV]], %[[paddedO]])
  rock.attention(%arg0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    params = #xldops_attn_params
  } :  memref<1x49x7xf32>, memref<1x7x49xf32>, memref<1x49x7xf32>, memref<1x49x7xf32>
  return
}
