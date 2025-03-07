// RUN: rocmlir-opt -rock-conv-to-gemm -rock-gemm-to-gridwise --rock-regularize --rock-gridwise-gemm-to-blockwise --rock-linalg-align %s -verify-diagnostics

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {

  func.func @rock_gemm(%arg0: memref<1x1024x1024xf32>, %arg1: memref<1x1024x512xf32>, %arg2: memref<1x1024x512xi16> {rock.prefill = 0.000000e+00 : f32}) attributes {block_size = 256 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
    %alloc = memref.alloc() : memref<1x1024x512xf32>
    rock.gemm %alloc = %arg0 * %arg1 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a", derivedBlockSize = 256 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 2, mPerBlock = 256, nPerBlock = 256, kpack = 4, mPerWave = 128, nPerWave = 128, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x1024x512xf32> = memref<1x1024x1024xf32> * memref<1x1024x512xf32>
    
    // expected-error @+1 {{'linalg.generic' op is infusible with non-`Set` store method}}
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<1x1024x512xf32>) outs(%arg2 : memref<1x1024x512xi16>) {
    ^bb0(%in: f32, %out: i16):
      %cst = arith.constant 2.000000e+00 : f32
      %0 = arith.addf %in, %cst : f32
      %1 = arith.fptosi %0 : f32 to i16
      linalg.yield %1 : i16
    }
    return
  }
}
