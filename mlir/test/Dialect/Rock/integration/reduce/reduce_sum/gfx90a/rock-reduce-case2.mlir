// This test is checking sensitivity for reduction in a higher dimension

// RUN: cat %s | rocmlir-gen -ph -print-results -fut test_reduce -verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  func.func private @zero_init(%arg0: memref<10x1x20xf32> {func.write_access}) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg0 : memref<10x1x20xf32>)
    return
  }
  func.func private @test_reduce__part_1(%arg0: memref<10x30x20xf32> {func.read_access}, %arg1: memref<10x1x20xf32> {func.read_access, func.write_access}) {
    %0 = memref.collapse_shape %arg1 [[0], [1, 2]] : memref<10x1x20xf32> into memref<10x20xf32>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0 : memref<10x30x20xf32>) outs(%0 : memref<10x20xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.addf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    }
    return
  }
  func.func @test_reduce(%arg0: memref<10x30x20xf32>, %arg1: memref<10x1x20xf32>) attributes {arch = ""} {
    call @zero_init (%arg1) : (memref<10x1x20xf32>) -> ()
    %token1 = mhal.launch @test_reduce__part_1 (%arg0, %arg1) : (memref<10x30x20xf32>, memref<10x1x20xf32>)
    mhal.await %token1 : !mhal.token
    return
  }
  module @__xmodule_gfx90a attributes {mhal.arch = "gfx90a",mhal.module} {
    func.func private @test_reduce__part_1(%arg0: memref<10x30x20xf32> {func.read_access}, %arg1: memref<10x1x20xf32> {func.read_access, func.write_access}) attributes {kernel, original_func = @test_reduce__part_1, grid_size = 1, block_size = 256} {
      rock.reduce sum %arg0 into %arg1 features = mfma|dot|atomic_add {axis = 1 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<10x30x20xf32> into memref<10x1x20xf32>
      return
    }
  }
}
