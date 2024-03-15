// RUN: rocmlir-opt --rock-view-to-transform -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise --rock-regularize --rock-gridwise-gemm-to-blockwise --rock-linalg-align %s -verify-diagnostics

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {

  func.func @rock_gemm(%arg0: memref<1x1024x1024xf32>, %arg1: memref<1x1024x512xf32>, %arg2: memref<1x1024x512xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
    %0 = memref.alloc() : memref<1x1024x512xf32>
    rock.gemm %0 = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod = atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a"} : memref<1x1024x512xf32> = memref<1x1024x1024xf32> * memref<1x1024x512xf32>

    // expected-error @+1 {{'linalg.generic' op lingalg generic ops are only allowed to operate with `Set` store method}}
    linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel", "parallel"]} ins(%0 : memref<1x1024x512xf32>) outs(%arg2 : memref<1x1024x512xf32>) {
    ^bb0(%i: f32, %o: f32):
        %c0 = arith.constant 2.0 : f32
        %r = arith.addf %i, %c0 : f32
        linalg.yield %r : f32
    }
    return
  }
}
