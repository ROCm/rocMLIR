// RUN: rocmlir-opt --rock-view-to-transform -rock-affix-params -rock-conv-to-gemm -rock-grid-config -rock-gemm-to-gridwise %s | FileCheck %s
// RUN: rocmlir-driver -kernel-pipeline=gpu --verify-passes %s | rocmlir-opt

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: memref<1x1024x769xf16>, %arg1: memref<1x769x512xf32>, %arg2: memref<1x1024x512xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
    rock.gemm %arg2 = %arg0 * %arg1 features =  dot|atomic_add|atomic_fmax_f32 storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx1100"} : memref<1x1024x512xf32> = memref<1x1024x769xf16> * memref<1x769x512xf32>
    return
  }
}

// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1x1024x769xf32>
// CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x1024x769xf16>) outs(%[[ALLOC]] : memref<1x1024x769xf32>)
