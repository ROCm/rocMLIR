// This test checks a case where reduction is happening in a middle dimension

// RUN: cat %s | rocmlir-gen -ph -print-results -fut test_reduce -verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func private @test_gemm_reduce_fusion__part_0(%arg0: memref<1x1x256xf32> {func.write_access}) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg0 : memref<1x1x256xf32>)
    return
  }
  func.func private @test_gemm_reduce_fusion__part_1(%arg0: memref<1x128x64xf32> {func.read_access}, %arg1: memref<1x64x256xf32> {func.read_access}, %arg2: memref<1x1x256xf32> {func.read_access, func.write_access}) {
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x128x256xf32>
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%0 : memref<1x128x256xf32>)
    linalg.batch_matmul ins(%arg0, %arg1 : memref<1x128x64xf32>, memref<1x64x256xf32>) outs(%0 : memref<1x128x256xf32>)
    %1 = memref.collapse_shape %0 [[0, 1], [2]] : memref<1x128x256xf32> into memref<128x256xf32>
    %2 = memref.collapse_shape %arg2 [[0, 1, 2]] : memref<1x1x256xf32> into memref<256xf32>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%1 : memref<128x256xf32>) outs(%2 : memref<256xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
    }
    return
  }
  func.func @test_reduce(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x1x256xf32>) attributes {arch = ""} {
    %token0 = async.launch @test_gemm_reduce_fusion__part_0 (%arg2) : (memref<1x1x256xf32>) -> ()
    %token1 = async.launch @test_gemm_reduce_fusion__part_1 [%token0] (%arg0, %arg1, %arg2) : (memref<1x128x64xf32>, memref<1x64x256xf32>, memref<1x1x256xf32>) -> ()
    async.await %token1 : !async.token
    return
  }
  module @__xmodule_gfx908 attributes {xmodel.arch = "gfx908", xmodel.module} {
    func.func private @test_gemm_reduce_fusion__part_0(%arg0: memref<1x1x256xf32> {func.write_access}) attributes {kernel, original_func = @test_gemm_reduce_fusion__part_0, grid_size = 1, block_size = 128} {
      rock.zero_init_kernel %arg0 features =  mfma|dot|atomic_add {arch = "", blockSize = 128 : i32, elemsPerThread = 1 : index, gridSize = 1 : i32} : memref<1x1x256xf32>
      return
    }
    func.func private @test_gemm_reduce_fusion__part_1(%arg0: memref<1x128x64xf32> {func.read_access}, %arg1: memref<1x64x256xf32> {func.read_access}, %arg2: memref<1x1x256xf32> {func.read_access, func.write_access}) attributes {kernel, original_func = @test_gemm_reduce_fusion__part_1} {
      %0 = memref.alloc() : memref<1x128x256xf32>
      rock.gemm %0 = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx908"} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
      rock.reduce sum %0 into %arg2 features = mfma|dot|atomic_add {axis = 1 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x1x256xf32>
      return
    }
  }
}
