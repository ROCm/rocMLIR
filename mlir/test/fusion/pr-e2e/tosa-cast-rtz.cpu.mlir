// RUN: rocmlir-driver -host-pipeline=highlevel %s | rocmlir-gen -rand=none -ph -pr -fut test_cast_rtz - \
// RUN: | rocmlir-driver -host-pipeline=runner \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // This test directly tags a tosa.cast with the RTZ FusedLoc metadata, so we
  // can drive deterministic input values via tosa.const and check the exact
  // numerical output. The migraphx.convert -> tosa.cast tag-insertion path
  // is covered by the lit test in
  // test/Conversion/FixTosaCastRounding/fix-tosa-cast-rounding.mlir.
  //
  // RTZ truncates toward zero: 3.5 -> 3, -3.5 -> -3 (RNE would be 4 and -4).
  // %arg0 is a dummy input required by rocmlir-gen's host-placeholder driver
  // (-ph); the kernel itself only consumes the tosa.const above.
  // CHECK: Unranked Memref {{.*}}
  // CHECK-NEXT: [2,  3,  -2,  -3,  2,  -2]
  func.func @test_cast_rtz(%arg0: tensor<1xf32>) -> tensor<6xi32> {
    %0 = "tosa.const"() {values = dense<[2.7, 3.5, -2.5, -3.5, 2.5, -2.7]> : tensor<6xf32>} : () -> tensor<6xf32>
    %1 = tosa.cast %0 : (tensor<6xf32>) -> tensor<6xi32> loc(fused<"rocmlir.rtz_cast">["cast_rtz":0:0])
    return %1 : tensor<6xi32>
  }
}
