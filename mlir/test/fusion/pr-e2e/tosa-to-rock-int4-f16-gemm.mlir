// RUN: rocmlir-gen --clone-harness -arch %arch -fut gemmi4f16 %s | rocmlir-driver -host-pipeline highlevel -targets %arch | rocmlir-gen -ph -fut gemmi4f16_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full --arch %arch | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// ALLOW-RETRIES: 2

!aCompressedFlat = tensor<4096xi4>
!bFlat = tensor<4096xf16>
!cFlat = tensor<4096xf16>
!aCompressed = tensor<1x64x64xi4>
!a = tensor<1x64x64xf16>
!b = tensor<1x64x64xf16>
!c = tensor<1x64x64xf16>

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @gemmi4f16(%arg0: !aCompressedFlat, %arg1: !bFlat) -> !cFlat {
  %const_shape = "tosa.const_shape"() { values = dense<[1, 64, 64]> : tensor<3xindex> } : () -> !tosa.shape<3>
  %0 = tosa.reshape %arg0, %const_shape : (!aCompressedFlat, !tosa.shape<3>) -> !aCompressed
  %1 = tosa.reshape %arg1, %const_shape : (!bFlat, !tosa.shape<3>) -> !b
  %2 = tensor.empty() : !a
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0 : !aCompressed) outs(%2 : !a) {
    ^bb0(%in : i4, %out: f16):
      %4 = arith.extui %in : i4 to i8
      %5 = arith.uitofp %4 : i8 to f16
      linalg.yield %5 : f16
  } -> !a
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %6 = tosa.matmul %3, %1, %a_zp, %b_zp : (!a, !b, tensor<1xf16>, tensor<1xf16>) -> !c
  %const_shape2 = "tosa.const_shape"() { values = dense<[4096]> : tensor<1xindex> } : () -> !tosa.shape<1>
  %7 = tosa.reshape %6, %const_shape2 : (!c, !tosa.shape<1>) -> !cFlat
  func.return %7 : !cFlat
}
