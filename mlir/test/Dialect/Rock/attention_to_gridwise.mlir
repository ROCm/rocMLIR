// Ensures that the padding application, group application, etc. in gemm-to-gridwise
// function as expected.

// RUN: rocmlir-opt -rock-attention-to-gridwise %s | FileCheck %s

#xldops_gemm_params = #rock.xdlops_gemm_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 32, forceUnroll = true>

// CHECK-LABEL: func.func @rock_attention_simple
// CHECK-SAME: (%[[q:.*]]: memref<1x64x1024xf32>, %[[k:.*]]: memref<1x64x1024xf32>, %[[v:.*]]: memref<1x64x1024xf32>, %[[o:.*]]: memref<1x64x1024xf32>)
// CHECK-SAME: block_size = 64
// CHECK-SAME: grid_size = 32
func.func @rock_attention_simple(%arg0: memref<1x64x1024xf32>, %arg1: memref<1x64x1024xf32>, %arg2: memref<1x64x1024xf32>, %arg3: memref<1x64x1024xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.gridwise_attention_accel(%[[q]], %[[k]], %[[v]], %[[o]])
  rock.attention(%arg0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    params = #xldops_gemm_params,
    qTransposed,
    vTransposed
  } : memref<1x64x1024xf32>, memref<1x64x1024xf32>, memref<1x64x1024xf32>, memref<1x64x1024xf32>
  return
}

// CHECK-LABEL: func.func @rock_attention_tr_padded
// CHECK-SAME: (%[[q:.*]]: memref<1x1024x32xf32>, %[[k:.*]]: memref<1x32x1024xf32>, %[[v:.*]]: memref<1x1024x32xf32>, %[[o:.*]]: memref<1x1024x32xf32>)
func.func @rock_attention_tr_padded(%arg0: memref<1x1024x32xf32>, %arg1: memref<1x32x1024xf32>, %arg2: memref<1x1024x32xf32>, %arg3: memref<1x1024x32xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : memref<1x1024x32xf32> to memref<1x32x1024xf32>
  // CHECK-DAG: %[[paddedTrQ:.*]] = rock.transform %[[trQ]] by {{.*}} : memref<1x32x1024xf32> to memref<1x64x1024xf32>
  // CHECK-DAG: %[[paddedK:.*]] = rock.transform %[[k]] by {{.*}} : memref<1x32x1024xf32> to memref<1x64x1024xf32>
  // CHECK-DAG: %[[trV:.*]] = rock.transform %[[v]] by {{.*}} : memref<1x1024x32xf32> to memref<1x32x1024xf32>
  // CHECK-DAG: %[[paddedTrV:.*]] = rock.transform %[[trV]] by {{.*}} : memref<1x32x1024xf32> to memref<1x64x1024xf32>
  // CHECK: rock.gridwise_attention_accel(%[[paddedTrQ]], %[[paddedK]], %[[paddedTrV]], %[[o]])
  rock.attention(%arg0, %arg1, %arg2, %arg3) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    params = #xldops_gemm_params
  } : memref<1x1024x32xf32>, memref<1x32x1024xf32>, memref<1x1024x32xf32>, memref<1x1024x32xf32>
  return
}
