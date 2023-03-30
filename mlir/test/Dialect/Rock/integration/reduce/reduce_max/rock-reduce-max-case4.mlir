// This test is checking for larger reductions with larger block and grid sizes

// RUN: sed -e 's/##TOKEN_ARCH##/%arch/g; s/##TOKEN_FEATURES##/%features/g' %s | rocmlir-gen -ph -print-results -fut test_reduce -verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext,%linalg_test_lib_dir/%prefix_mlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  func.func private @init_output(%arg0: memref<1x250x100xf32> {func.write_access}) {
    %cst = arith.constant 0xff800000 : f32 
    linalg.fill ins(%cst : f32) outs(%arg0 : memref<1x250x100xf32>)
    return
  }
  func.func private @test_reduce__part_1(%arg0: memref<1000x250x100xf32> {func.read_access}, %arg1: memref<1x250x100xf32> {func.read_access, func.write_access}) {
    %0 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<1x250x100xf32> into memref<250x100xf32>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%arg0 : memref<1000x250x100xf32>) outs(%0 : memref<250x100xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maxf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    }
    return
  }
  func.func @test_reduce(%arg0: memref<1000x250x100xf32>, %arg1: memref<1x250x100xf32>) attributes {arch = ""} {
    call @init_output (%arg1) : (memref<1x250x100xf32>) -> ()
    %token1 = async.launch @test_reduce__part_1 (%arg0, %arg1) : (memref<1000x250x100xf32>, memref<1x250x100xf32>) -> ()
    async.await %token1 : !async.token
    return
  }
  module @__xmodule_ attributes {xmodel.arch = "##TOKEN_ARCH##", xmodel.module} {
    func.func private @test_reduce__part_1(%arg0: memref<1000x250x100xf32> {func.read_access}, %arg1: memref<1x250x100xf32> {func.read_access, func.write_access}) attributes {kernel, original_func = @test_reduce__part_1, grid_size = 16, block_size = 1024} {
      rock.reduce max %arg0 into %arg1 features = ##TOKEN_FEATURES## {axis = 0 : index, blockSize = 1024 : i32, gridSize = 16 : i32} : memref<1000x250x100xf32> into memref<1x250x100xf32>
      return
    }
  }
}
