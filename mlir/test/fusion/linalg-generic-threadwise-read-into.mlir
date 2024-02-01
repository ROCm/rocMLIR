// RUN: rocmlir-driver -kernel-pipeline=gpu --verify-passes %s | rocmlir-opt

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: memref<1x1x1xf16>, %arg1: memref<1x1x1xf32>, %arg2: memref<1x1x1xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
    %0 = memref.alloc() : memref<1x1x1xf32>
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel", "parallel"]} ins(%arg0 : memref<1x1x1xf16>) outs(%0 : memref<1x1x1xf32>) {
    ^bb0(%arg3: f16, %arg4: f32):
        %1 = arith.extf %arg3 : f16 to f32
        linalg.yield %1 : f32
    }
    rock.gemm %arg2 = %0 * %arg1 features =  dot|atomic_add|atomic_fmax_f32 storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx1100"} : memref<1x1x1xf32> = memref<1x1x1xf32> * memref<1x1x1xf32>
    return
  }
}
