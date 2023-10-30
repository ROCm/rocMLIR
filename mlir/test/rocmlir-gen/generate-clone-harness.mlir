// RUN: rocmlir-gen -fut mlir_reshape_convolution  --arch %arch --clone-harness %s| rocmlir-driver -kernel-pipeline=migraphx -host-pipeline=migraphx,highlevel |rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_reshape_convolution_wrapper --verifier clone - | FileCheck %s
// RUN: rocmlir-gen -fut mlir_reshape_convolution  --arch %arch --clone-harness %s| rocmlir-driver -kernel-pipeline=migraphx -host-pipeline=migraphx,highlevel |rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_reshape_convolution_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full| FileCheck %s  --check-prefixes=CHECK_FULL


func.func private @mlir_reshape_convolution(%arg0: tensor<1x1x16x1x16x1xf32>, %arg1: tensor<1x1x3x3xf32>) -> (tensor<1x1x32x32xf32>) {
    %0 = migraphx.multibroadcast(%arg0) {out_dyn_dims = [], out_lens = [1, 1, 16, 2, 16, 2]} : (tensor<1x1x16x1x16x1xf32>) -> tensor<1x1x16x2x16x2xf32>
    %1 = migraphx.reshape(%0) {dims = [2, 4, 32, 32]} : (tensor<1x1x16x2x16x2xf32>) -> tensor<1x1x32x32xf32>
    %2 = migraphx.convolution(%1, %arg1) {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : (tensor<1x1x32x32xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x32x32xf32>
    return %2 : tensor<1x1x32x32xf32>
  }

// CHECK-LABEL: module
// CHECK: func.func private @mlir_reshape_convolution({{.*}}: memref<1x1x16x1x16x1xf32> {func.read_access}, {{.*}}: memref<1x1x3x3xf32> {func.read_access}, {{.*}}: memref<1x1x32x32xf32> {func.write_access})
// CHECK: func.func @mlir_reshape_convolution_wrapper({{.*}}: memref<1x1x16x1x16x1xf32>, {{.*}}: memref<1x1x3x3xf32>, {{.*}}: memref<1x1x32x32xf32>)
// CHECK: module @__xmodule_ attributes {mhal.arch = "{{.*}}", mhal.module}
// CHECK: func.func private @mlir_reshape_convolution({{.*}}: memref<1x1x16x1x16x1xf32> {func.read_access}, {{.*}}: memref<1x1x3x3xf32> {func.read_access}, {{.*}}: memref<1x1x32x32xf32> {func.write_access}) attributes {kernel, original_func = @mlir_reshape_convolution}
// CHECK-LABEL: @main
// CHECK: call @mlir_reshape_convolution_wrapper({{.*}}, {{.*}}, {{.*}}) : (memref<1x1x16x1x16x1xf32>, memref<1x1x3x3xf32>, memref<1x1x32x32xf32>) -> ()
// CHECK: call @mlir_reshape_convolution_wrapper_cloned({{.*}}, {{.*}}, {{.*}}) : (memref<1x1x16x1x16x1xf32>, memref<1x1x3x3xf32>, memref<1x1x32x32xf32>) -> ()
// CHECK: call @mlir_reshape_convolution_wrapper_verify2({{.*}}, {{.*}}) : (memref<1x1x32x32xf32>, memref<1x1x32x32xf32>) -> ()
// CHECK: func.func @mlir_reshape_convolution_wrapper_cloned({{.*}}: memref<1x1x16x1x16x1xf32>, {{.*}}: memref<1x1x3x3xf32>, {{.*}}: memref<1x1x32x32xf32>)

// CHECK_FULL-LABEL: module
// CHECK_FULL: func.func private @mlir_reshape_convolution
// CHECK_FULL-SAME: attributes {mhal.targets = [#mhal.kernel_pkg<GPU = {{.*}} : mlir_reshape_convolution [16, 64] -> #mhal.target_obj
