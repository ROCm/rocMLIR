// RUN: rocmlir-gen -ph -print-results -rand none %s | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2


// CHECK:  73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73,      73
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
module {
  func.func @test_fusion(%arg0: memref<1x1x32x32x8xf32>, %arg1: memref<1x16x3x3x8xf32>, %arg2: memref<16xf32>, %arg3: memref<1x1x30x30x16xf32>) attributes {kernel, arch = ""} {
    %0 = memref.alloc() : memref<1x1x30x30x16xf32>
    rock.conv2d(%arg1, %arg0, %0) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["gi", "ni", "hi", "wi", "ci"], output_layout = ["go", "no", "ho", "wo", "ko"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x16x3x3x8xf32>, memref<1x1x32x32x8xf32>, memref<1x1x30x30x16xf32>
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0, %arg2 : memref<1x1x30x30x16xf32>, memref<16xf32>) outs(%arg3 : memref<1x1x30x30x16xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %8 = arith.addf %arg4, %arg5 : f32
      linalg.yield %8 : f32
    }
    return
  }
}
