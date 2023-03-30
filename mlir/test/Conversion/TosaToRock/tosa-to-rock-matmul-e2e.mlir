// RUN:  rocmlir-driver -host-pipeline highlevel %s | rocmlir-gen -ph -rand=none -print-results - \
// RUN: | rocmlir-driver -kernel-pipeline full -host-pipeline runner -arch %arch \
// RUN: | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 3 offset = 0 sizes = [1, 128, 256] strides = [32768, 256, 1] data =
// CHECK-NEXT: 64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,

func.func @test_fusion(%a: tensor<1x128x64xf32>, %b: tensor<1x64x256xf32>) -> tensor<1x128x256xf32> attributes {kernel, arch = ""} {
  %0 = "tosa.matmul"(%a, %b) {} : (tensor<1x128x64xf32>, tensor<1x64x256xf32>) -> tensor<1x128x256xf32>

  return %0 : tensor<1x128x256xf32>
}

// -----
