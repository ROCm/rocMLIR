// RUN:  mlir-miopen-driver -host-pipeline highlevel %s | miopen-gen -ph -rand=none -print-results - | mlir-miopen-driver -kernel-pipeline gpu | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 3 offset = 0 sizes = [10, 128, 256] strides = [32768, 256, 1] data =
// CHECK-NEXT: 64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,    64,

func.func @test_fusion(%a: tensor<10x128x64xf32>, %b: tensor<10x64x256xf32>) -> tensor<10x128x256xf32> attributes {kernel, arch = "gfx908"} {
  %0 = "tosa.matmul"(%a, %b) {} : (tensor<10x128x64xf32>, tensor<10x64x256xf32>) -> tensor<10x128x256xf32>

  return %0 : tensor<10x128x256xf32>
}

// -----
