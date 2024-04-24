// RUN: rocmlir-driver -c -arch %arch %s | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

#map = affine_map<(d0) -> (d0 mod 3)>
module {
  func.func @rock_conv_gkc012_ngc012_ngk012_0(%arg0: memref<1x1x1x3x3x3xf32>, %arg1: memref<1x1x1x8x8x8xf32>, %arg2: memref<1x1x1x6x6x6xf32>) attributes {kernel = 0 : i32} {
    rock.conv(%arg0, %arg1, %arg2) features =  dot {arch = "", dilations = [1 : index, 1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1", "2"], input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"], numCU = 104 : i32, output_layout = ["no", "go", "ko", "0o", "1o", "2o"], padding = [0 : index, 0 : index, 0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index, 1 : index]} : memref<1x1x1x3x3x3xf32>, memref<1x1x1x8x8x8xf32>, memref<1x1x1x6x6x6xf32>
    return
  }
  func.func @linalg_conv_3d(%arg0: memref<1x1x1x8x8x8xf32> , %arg1: memref<1x1x1x3x3x3xf32>, %arg2: memref<1x1x1x6x6x6xf32>) {
    %a0 = memref.collapse_shape %arg0 [[0, 1, 2, 3], [4], [5]] : memref<1x1x1x8x8x8xf32> into memref<8x8x8xf32>
    %a1 = memref.collapse_shape %arg1 [[0, 1, 2, 3], [4], [5]] : memref<1x1x1x3x3x3xf32> into memref<3x3x3xf32>
    %a2 = memref.collapse_shape %arg2 [[0, 1, 2, 3], [4], [5]] : memref<1x1x1x6x6x6xf32> into memref<6x6x6xf32>
    linalg.conv_3d ins (%a0, %a1: memref<8x8x8xf32>, memref<3x3x3xf32>)
                  outs (%a2: memref<6x6x6xf32>)
    return
  }
  func.func @main() {
    %two = arith.constant 2.00000e+00 : f32
    %alloc = memref.alloc() : memref<1x1x1x3x3x3xf32>
    %collapse_shape = memref.collapse_shape %alloc [[0, 1, 2, 3, 4, 5]] : memref<1x1x1x3x3x3xf32> into memref<27xf32>
    linalg.fill ins(%two : f32) outs(%collapse_shape : memref<27xf32>)

    %alloc_0 = memref.alloc() : memref<1x1x1x8x8x8xf32>
    %collapse_shape_1 = memref.collapse_shape %alloc_0 [[0, 1, 2, 3, 4, 5]] : memref<1x1x1x8x8x8xf32> into memref<512xf32>
    linalg.fill ins(%two : f32) outs(%collapse_shape_1 : memref<512xf32>)
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %f0 = arith.constant 0.00000e+00 : f32
    %f10 = arith.constant 10.00000e+00 : f32
    memref.store %f10, %alloc_0[%c0, %c0, %c0, %c0, %c0, %c3] : memref<1x1x1x8x8x8xf32>

    %alloc_2 = memref.alloc() : memref<1x1x1x6x6x6xf32>
    %collapse_shape_3 = memref.collapse_shape %alloc_2 [[0, 1, 2, 3, 4, 5]] : memref<1x1x1x6x6x6xf32> into memref<216xf32>
    linalg.fill ins(%f0 : f32) outs(%collapse_shape_3 : memref<216xf32>)

    call @rock_conv_gkc012_ngc012_ngk012_0_gpu(%alloc, %alloc_0, %alloc_2) : (memref<1x1x1x3x3x3xf32>, memref<1x1x1x8x8x8xf32>, memref<1x1x1x6x6x6xf32>) -> ()
    %cast = memref.cast %alloc_2 : memref<1x1x1x6x6x6xf32> to memref<*xf32>

    %alloc_4 = memref.alloc() : memref<1x1x1x6x6x6xf32>
    %collapse_shape_4 = memref.collapse_shape %alloc_4 [[0, 1, 2, 3, 4, 5]] : memref<1x1x1x6x6x6xf32> into memref<216xf32>
    linalg.fill ins(%f0 : f32) outs(%collapse_shape_4 : memref<216xf32>)
    call @linalg_conv_3d(%alloc_0, %alloc, %alloc_4) : (memref<1x1x1x8x8x8xf32>, memref<1x1x1x3x3x3xf32>, memref<1x1x1x6x6x6xf32>) -> ()
    %cast4 = memref.cast %alloc_4 : memref<1x1x1x6x6x6xf32> to memref<*xf32>

    call @conv3d_verify(%collapse_shape_3, %collapse_shape_4) : (memref<216xf32>, memref<216xf32>) -> ()

    memref.dealloc %alloc : memref<1x1x1x3x3x3xf32>
    memref.dealloc %alloc_0 : memref<1x1x1x8x8x8xf32>
    memref.dealloc %alloc_2 : memref<1x1x1x6x6x6xf32>
    memref.dealloc %alloc_4 : memref<1x1x1x6x6x6xf32>
    return
  }
  func.func @rock_conv_gkc012_ngc012_ngk012_0_gpu(%arg0: memref<1x1x1x3x3x3xf32>, %arg1: memref<1x1x1x8x8x8xf32>, %arg2: memref<1x1x1x6x6x6xf32>) {
    %memref = gpu.alloc  () : memref<1x1x1x3x3x3xf32>
    gpu.memcpy  %memref, %arg0 : memref<1x1x1x3x3x3xf32>, memref<1x1x1x3x3x3xf32>
    %memref_0 = gpu.alloc  () : memref<1x1x1x8x8x8xf32>
    gpu.memcpy  %memref_0, %arg1 : memref<1x1x1x8x8x8xf32>, memref<1x1x1x8x8x8xf32>
    %memref_1 = gpu.alloc  () : memref<1x1x1x6x6x6xf32>
    gpu.memcpy  %memref_1, %arg2 : memref<1x1x1x6x6x6xf32>, memref<1x1x1x6x6x6xf32>
    call @rock_conv_gkc012_ngc012_ngk012_0(%memref, %memref_0, %memref_1) : (memref<1x1x1x3x3x3xf32>, memref<1x1x1x8x8x8xf32>, memref<1x1x1x6x6x6xf32>) -> ()
    gpu.memcpy  %arg0, %memref : memref<1x1x1x3x3x3xf32>, memref<1x1x1x3x3x3xf32>
    gpu.dealloc  %memref : memref<1x1x1x3x3x3xf32>
    gpu.memcpy  %arg1, %memref_0 : memref<1x1x1x8x8x8xf32>, memref<1x1x1x8x8x8xf32>
    gpu.dealloc  %memref_0 : memref<1x1x1x8x8x8xf32>
    gpu.memcpy  %arg2, %memref_1 : memref<1x1x1x6x6x6xf32>, memref<1x1x1x6x6x6xf32>
    gpu.dealloc  %memref_1 : memref<1x1x1x6x6x6xf32>
    return
  }

  func.func @conv3d_verify(%arg0: memref<216xf32>, %arg1: memref<216xf32>) {
    %print_verify = arith.constant 1 : i8
    %rms = arith.constant 1.000000e-03 : f32
    %abs = arith.constant 1.000000e+02 : f32
    %rel = arith.constant 1.000000e-06 : f32
    %cast = memref.cast %arg0 : memref<216xf32> to memref<?xf32>
    %cast_1 = memref.cast %arg1 : memref<216xf32> to memref<?xf32>
    call @mcpuVerifyFloat(%cast, %cast_1, %rms, %abs, %rel, %print_verify) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8) -> ()
    return
  }
  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8)
}
