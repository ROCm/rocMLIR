// ALLOW_RETRIES: 2
// MLIR#765: Bug in TosaMakeBroadcastable, or LinalgElementwiseOpFusion
// XFAIL: *

// RUN: rocmlir-driver -host-pipeline partition -targets %arch %s | FileCheck %s
// RUN-DISABLE: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | rocmlir-gen -ph -print-results -verifier clone -fut forward - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers %shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext,%linalg_test_lib_dir/%prefix_mlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%prefix_mlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// COM: CHECK: [1 1 1]
// CHECK: forward__part_13

module attributes {torch.debug_module_name = "BertTinyWrapper"} {
  func.func @forward(%arg0: tensor<2x128xi64>) -> tensor<2x128x30522xf32> {
    %0 = "tosa.const"() {value = dense<1.23e-01> : tensor<128xf32>} : () -> tensor<128xf32>
    %1 = "tosa.const"() {value = dense<1.23e-01> : tensor<128x128xf32>} : () -> tensor<128x128xf32>
    %2 = "tosa.const"() {value = dense<1.23e-01> : tensor<30522xf32>} : () -> tensor<30522xf32>
    %3 = "tosa.const"() {value = dense<1.23e-01> : tensor<128x512xf32>} : () -> tensor<128x512xf32>
    %4 = "tosa.const"() {value = dense<1.23e-01> : tensor<512x128xf32>} : () -> tensor<512x128xf32>
    %5 = "tosa.const"() {value = dense<1.23e-01> : tensor<512xf32>} : () -> tensor<512xf32>
    %6 = "tosa.const"() {value = dense<8.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %7 = "tosa.const"() {value = dense<2.303890e-01> : tensor<2x128xf32>} : () -> tensor<2x128xf32>
    %8 = "tosa.const"() {value = dense<7> : tensor<1x512xi64>} : () -> tensor<1x512xi64>
    %9 = "tosa.const"() {value = dense<2.783930e-01> : tensor<30522x128xf32>} : () -> tensor<30522x128xf32>
    %10 = "tosa.const"() {value = dense<1.280000e+02> : tensor<1xf32>} : () -> tensor<1xf32>
    %11 = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %12 = "tosa.const"() {value = dense<[1, 0, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
    %13 = "tosa.const"() {value = dense<[0, 2, 1, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
    %14 = "tosa.const"() {value = dense<[0, 1, 3, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %15 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %16 = "tosa.const"() {value = dense<7.810800e-02> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %17 = "tosa.const"() {value = dense<9.720000e-04> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %18 = "tosa.const"() {value = dense<2.303890e-01> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %19 = "tosa.const"() {value = dense<2.783930e-01> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %20 = "tosa.const"() {value = dense<0.707106769> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %21 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %22 = "tosa.const"() {value = dense<9.99999996E-13> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %23 = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
    %24 = "tosa.const"() {value = dense<-3.4028234663852886E+38> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %25 = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %26 = "tosa.const"() {value = dense<0> : tensor<1x128xi32>} : () -> tensor<1x128xi32>
    %27 = "tosa.const"() {value = dense<1.000000e+00> : tensor<2x1x1x128xf32>} : () -> tensor<2x1x1x128xf32>
    %28 = "tosa.sub"(%25, %27) : (tensor<1x1x1x1xf32>, tensor<2x1x1x128xf32>) -> tensor<2x1x1x128xf32>
    %29 = "tosa.mul"(%28, %24) {shift = 0 : i32} : (tensor<2x1x1x128xf32>, tensor<1x1x1x1xf32>) -> tensor<2x1x1x128xf32>
    %30 = "tosa.slice"(%8) {size = array<i64: 1, 128>, start = array<i64: 0, 0>} : (tensor<1x512xi64>) -> tensor<1x128xi64>
    %31 = "tosa.reshape"(%9) {new_shape = array<i64: 1, 30522, 128>} : (tensor<30522x128xf32>) -> tensor<1x30522x128xf32>
    %32 = "tosa.reshape"(%arg0) {new_shape = array<i64: 1, 256>} : (tensor<2x128xi64>) -> tensor<1x256xi64>
    %33 = "tosa.cast"(%32) : (tensor<1x256xi64>) -> tensor<1x256xi32>
    %34 = "tosa.gather"(%31, %33) : (tensor<1x30522x128xf32>, tensor<1x256xi32>) -> tensor<1x256x128xf32>
    %35 = "tosa.reshape"(%34) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %36 = "tosa.reshape"(%7) {new_shape = array<i64: 1, 2, 128>} : (tensor<2x128xf32>) -> tensor<1x2x128xf32>
    %37 = "tosa.gather"(%36, %26) : (tensor<1x2x128xf32>, tensor<1x128xi32>) -> tensor<1x128x128xf32>
    
    // FIXME : invalid reshape
    %38 = "tosa.reshape"(%37) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x128x128xf32>) -> tensor<2x128x128xf32>
    %39 = "tosa.add"(%35, %38) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %40 = "tosa.reshape"(%4) {new_shape = array<i64: 1, 512, 128>} : (tensor<512x128xf32>) -> tensor<1x512x128xf32>
    %41 = "tosa.cast"(%30) : (tensor<1x128xi64>) -> tensor<1x128xi32>
    %42 = "tosa.gather"(%40, %41) : (tensor<1x512x128xf32>, tensor<1x128xi32>) -> tensor<1x128x128xf32>
    %43 = "tosa.add"(%39, %42) : (tensor<2x128x128xf32>, tensor<1x128x128xf32>) -> tensor<2x128x128xf32>
    %44 = "tosa.reciprocal"(%10) : (tensor<1xf32>) -> tensor<1xf32>
    %45 = "tosa.reduce_sum"(%43) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %46 = "tosa.reshape"(%44) {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
    %47 = "tosa.mul"(%45, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %48 = "tosa.sub"(%43, %47) : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %49 = "tosa.mul"(%48, %48) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %50 = "tosa.reduce_sum"(%49) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %51 = "tosa.mul"(%50, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %52 = "tosa.reshape"(%0) {new_shape = array<i64: 1, 1, 128>} : (tensor<128xf32>) -> tensor<1x1x128xf32>
    %53 = "tosa.add"(%51, %22) : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %54 = "tosa.rsqrt"(%53) : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %55 = "tosa.mul"(%48, %54) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %56 = "tosa.mul"(%55, %52) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %57 = "tosa.add"(%56, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %58 = "tosa.transpose"(%1, %11) : (tensor<128x128xf32>, tensor<2xi32>) -> tensor<128x128xf32>
    %59 = "tosa.reshape"(%58) {new_shape = array<i64: 1, 128, 128>} : (tensor<128x128xf32>) -> tensor<1x128x128xf32>
    %60 = "tosa.reshape"(%57) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x128xf32>) -> tensor<1x256x128xf32>
    %61 = "tosa.transpose"(%59, %12) : (tensor<1x128x128xf32>, tensor<3xi32>) -> tensor<128x1x128xf32>
    %62 = "tosa.reshape"(%61) {new_shape = array<i64: 1, 128, 128>} : (tensor<128x1x128xf32>) -> tensor<1x128x128xf32>
    %63 = "tosa.matmul"(%60, %62) : (tensor<1x256x128xf32>, tensor<1x128x128xf32>) -> tensor<1x256x128xf32>
    %64 = "tosa.reshape"(%63) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %65 = "tosa.add"(%64, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %66 = "tosa.reshape"(%65) {new_shape = array<i64: 2, 128, 2, 64>} : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %67 = "tosa.transpose"(%66, %13) : (tensor<2x128x2x64xf32>, tensor<4xi64>) -> tensor<2x2x128x64xf32>
    %68 = "tosa.transpose"(%67, %14) : (tensor<2x2x128x64xf32>, tensor<4xi32>) -> tensor<2x2x64x128xf32>
    %69 = "tosa.reshape"(%67) {new_shape = array<i64: 4, 128, 64>} : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %70 = "tosa.reshape"(%68) {new_shape = array<i64: 4, 64, 128>} : (tensor<2x2x64x128xf32>) -> tensor<4x64x128xf32>
    %71 = "tosa.matmul"(%69, %70) : (tensor<4x128x64xf32>, tensor<4x64x128xf32>) -> tensor<4x128x128xf32>
    %72 = "tosa.reshape"(%71) {new_shape = array<i64: 2, 2, 128, 128>} : (tensor<4x128x128xf32>) -> tensor<2x2x128x128xf32>
    %73 = "tosa.reciprocal"(%6) : (tensor<f32>) -> tensor<f32>
    %74 = "tosa.reshape"(%73) {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %75 = "tosa.mul"(%72, %74) {shift = 0 : i32} : (tensor<2x2x128x128xf32>, tensor<1x1x1x1xf32>) -> tensor<2x2x128x128xf32>
    %76 = "tosa.add"(%75, %29) : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tensor<2x2x128x128xf32>
    %77 = "tosa.reduce_max"(%76) {axis = 3 : i64} : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x1xf32>
    %78 = "tosa.sub"(%76, %77) : (tensor<2x2x128x128xf32>, tensor<2x2x128x1xf32>) -> tensor<2x2x128x128xf32>
    %79 = "tosa.exp"(%78) : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %80 = "tosa.reduce_sum"(%79) {axis = 3 : i64} : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x1xf32>
    %81 = "tosa.reciprocal"(%80) : (tensor<2x2x128x1xf32>) -> tensor<2x2x128x1xf32>
    %82 = "tosa.mul"(%79, %81) {shift = 0 : i32} : (tensor<2x2x128x128xf32>, tensor<2x2x128x1xf32>) -> tensor<2x2x128x128xf32>
    %83 = "tosa.reshape"(%82) {new_shape = array<i64: 4, 128, 128>} : (tensor<2x2x128x128xf32>) -> tensor<4x128x128xf32>
    %84 = "tosa.matmul"(%83, %69) : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %85 = "tosa.reshape"(%84) {new_shape = array<i64: 2, 2, 128, 64>} : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %86 = "tosa.transpose"(%85, %13) : (tensor<2x2x128x64xf32>, tensor<4xi64>) -> tensor<2x128x2x64xf32>
    %87 = "tosa.reshape"(%86) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x2x64xf32>) -> tensor<1x256x128xf32>
    %88 = "tosa.matmul"(%87, %62) : (tensor<1x256x128xf32>, tensor<1x128x128xf32>) -> tensor<1x256x128xf32>
    %89 = "tosa.reshape"(%88) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %90 = "tosa.add"(%89, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %91 = "tosa.add"(%90, %57) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %92 = "tosa.reduce_sum"(%91) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %93 = "tosa.mul"(%92, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %94 = "tosa.sub"(%91, %93) : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %95 = "tosa.mul"(%94, %94) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %96 = "tosa.reduce_sum"(%95) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %97 = "tosa.mul"(%96, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %98 = "tosa.add"(%97, %22) : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %99 = "tosa.rsqrt"(%98) : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %100 = "tosa.mul"(%94, %99) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %101 = "tosa.mul"(%100, %52) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %102 = "tosa.add"(%101, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %103 = "tosa.transpose"(%4, %11) : (tensor<512x128xf32>, tensor<2xi32>) -> tensor<128x512xf32>
    %104 = "tosa.reshape"(%103) {new_shape = array<i64: 1, 128, 512>} : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %105 = "tosa.reshape"(%102) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x128xf32>) -> tensor<1x256x128xf32>
    %106 = "tosa.transpose"(%104, %12) : (tensor<1x128x512xf32>, tensor<3xi32>) -> tensor<128x1x512xf32>
    %107 = "tosa.reshape"(%106) {new_shape = array<i64: 1, 128, 512>} : (tensor<128x1x512xf32>) -> tensor<1x128x512xf32>
    %108 = "tosa.matmul"(%105, %107) : (tensor<1x256x128xf32>, tensor<1x128x512xf32>) -> tensor<1x256x512xf32>
    %109 = "tosa.reshape"(%108) {new_shape = array<i64: 2, 128, 512>} : (tensor<1x256x512xf32>) -> tensor<2x128x512xf32>
    %110 = "tosa.reshape"(%5) {new_shape = array<i64: 1, 1, 512>} : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %111 = "tosa.add"(%109, %110) : (tensor<2x128x512xf32>, tensor<1x1x512xf32>) -> tensor<2x128x512xf32>
    %112 = "tosa.sub"(%111, %21) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %113 = "tosa.mul"(%112, %20) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %114 = "tosa.abs"(%113) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %115 = "tosa.mul"(%114, %19) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %116 = "tosa.add"(%115, %23) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %117 = "tosa.mul"(%114, %114) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %118 = "tosa.mul"(%117, %18) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %119 = "tosa.add"(%116, %118) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %120 = "tosa.mul"(%117, %114) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %121 = "tosa.mul"(%120, %17) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %122 = "tosa.add"(%119, %121) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %123 = "tosa.mul"(%120, %114) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %124 = "tosa.mul"(%123, %16) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %125 = "tosa.add"(%122, %124) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %126 = "tosa.reciprocal"(%125) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %127 = "tosa.mul"(%126, %126) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %128 = "tosa.mul"(%127, %127) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %129 = "tosa.sub"(%23, %128) : (tensor<1x1x1xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %130 = "tosa.greater_equal"(%113, %21) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xi1>
    %131 = "tosa.negate"(%129) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %132 = "tosa.select"(%130, %129, %131) : (tensor<2x128x512xi1>, tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %133 = "tosa.add"(%132, %23) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %134 = "tosa.mul"(%133, %15) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %135 = "tosa.mul"(%111, %134) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %136 = "tosa.transpose"(%3, %11) : (tensor<128x512xf32>, tensor<2xi32>) -> tensor<512x128xf32>
    %137 = "tosa.reshape"(%136) {new_shape = array<i64: 1, 512, 128>} : (tensor<512x128xf32>) -> tensor<1x512x128xf32>
    %138 = "tosa.reshape"(%135) {new_shape = array<i64: 1, 256, 512>} : (tensor<2x128x512xf32>) -> tensor<1x256x512xf32>
    %139 = "tosa.transpose"(%137, %12) : (tensor<1x512x128xf32>, tensor<3xi32>) -> tensor<512x1x128xf32>
    %140 = "tosa.reshape"(%139) {new_shape = array<i64: 1, 512, 128>} : (tensor<512x1x128xf32>) -> tensor<1x512x128xf32>
    %141 = "tosa.matmul"(%138, %140) : (tensor<1x256x512xf32>, tensor<1x512x128xf32>) -> tensor<1x256x128xf32>
    %142 = "tosa.reshape"(%141) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %143 = "tosa.add"(%142, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %144 = "tosa.add"(%143, %102) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %145 = "tosa.reduce_sum"(%144) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %146 = "tosa.mul"(%145, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %147 = "tosa.sub"(%144, %146) : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %148 = "tosa.mul"(%147, %147) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %149 = "tosa.reduce_sum"(%148) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %150 = "tosa.mul"(%149, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %151 = "tosa.add"(%150, %22) : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %152 = "tosa.rsqrt"(%151) : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %153 = "tosa.mul"(%147, %152) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %154 = "tosa.mul"(%153, %52) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %155 = "tosa.add"(%154, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %156 = "tosa.reshape"(%155) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x128xf32>) -> tensor<1x256x128xf32>
    %157 = "tosa.matmul"(%156, %62) : (tensor<1x256x128xf32>, tensor<1x128x128xf32>) -> tensor<1x256x128xf32>
    %158 = "tosa.reshape"(%157) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %159 = "tosa.add"(%158, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %160 = "tosa.reshape"(%159) {new_shape = array<i64: 2, 128, 2, 64>} : (tensor<2x128x128xf32>) -> tensor<2x128x2x64xf32>
    %161 = "tosa.transpose"(%160, %13) : (tensor<2x128x2x64xf32>, tensor<4xi64>) -> tensor<2x2x128x64xf32>
    %162 = "tosa.transpose"(%161, %14) : (tensor<2x2x128x64xf32>, tensor<4xi32>) -> tensor<2x2x64x128xf32>
    %163 = "tosa.reshape"(%161) {new_shape = array<i64: 4, 128, 64>} : (tensor<2x2x128x64xf32>) -> tensor<4x128x64xf32>
    %164 = "tosa.reshape"(%162) {new_shape = array<i64: 4, 64, 128>} : (tensor<2x2x64x128xf32>) -> tensor<4x64x128xf32>
    %165 = "tosa.matmul"(%163, %164) : (tensor<4x128x64xf32>, tensor<4x64x128xf32>) -> tensor<4x128x128xf32>
    %166 = "tosa.reshape"(%165) {new_shape = array<i64: 2, 2, 128, 128>} : (tensor<4x128x128xf32>) -> tensor<2x2x128x128xf32>
    %167 = "tosa.mul"(%166, %74) {shift = 0 : i32} : (tensor<2x2x128x128xf32>, tensor<1x1x1x1xf32>) -> tensor<2x2x128x128xf32>
    %168 = "tosa.add"(%167, %29) : (tensor<2x2x128x128xf32>, tensor<2x1x1x128xf32>) -> tensor<2x2x128x128xf32>
    %169 = "tosa.reduce_max"(%168) {axis = 3 : i64} : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x1xf32>
    %170 = "tosa.sub"(%168, %169) : (tensor<2x2x128x128xf32>, tensor<2x2x128x1xf32>) -> tensor<2x2x128x128xf32>
    %171 = "tosa.exp"(%170) : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x128xf32>
    %172 = "tosa.reduce_sum"(%171) {axis = 3 : i64} : (tensor<2x2x128x128xf32>) -> tensor<2x2x128x1xf32>
    %173 = "tosa.reciprocal"(%172) : (tensor<2x2x128x1xf32>) -> tensor<2x2x128x1xf32>
    %174 = "tosa.mul"(%171, %173) {shift = 0 : i32} : (tensor<2x2x128x128xf32>, tensor<2x2x128x1xf32>) -> tensor<2x2x128x128xf32>
    %175 = "tosa.reshape"(%174) {new_shape = array<i64: 4, 128, 128>} : (tensor<2x2x128x128xf32>) -> tensor<4x128x128xf32>
    %176 = "tosa.matmul"(%175, %163) : (tensor<4x128x128xf32>, tensor<4x128x64xf32>) -> tensor<4x128x64xf32>
    %177 = "tosa.reshape"(%176) {new_shape = array<i64: 2, 2, 128, 64>} : (tensor<4x128x64xf32>) -> tensor<2x2x128x64xf32>
    %178 = "tosa.transpose"(%177, %13) : (tensor<2x2x128x64xf32>, tensor<4xi64>) -> tensor<2x128x2x64xf32>
    %179 = "tosa.reshape"(%178) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x2x64xf32>) -> tensor<1x256x128xf32>
    %180 = "tosa.matmul"(%179, %62) : (tensor<1x256x128xf32>, tensor<1x128x128xf32>) -> tensor<1x256x128xf32>
    %181 = "tosa.reshape"(%180) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %182 = "tosa.add"(%181, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %183 = "tosa.add"(%182, %155) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %184 = "tosa.reduce_sum"(%183) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %185 = "tosa.mul"(%184, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %186 = "tosa.sub"(%183, %185) : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %187 = "tosa.mul"(%186, %186) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %188 = "tosa.reduce_sum"(%187) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %189 = "tosa.mul"(%188, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %190 = "tosa.add"(%189, %22) : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %191 = "tosa.rsqrt"(%190) : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %192 = "tosa.mul"(%186, %191) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %193 = "tosa.mul"(%192, %52) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %194 = "tosa.add"(%193, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %195 = "tosa.reshape"(%194) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x128xf32>) -> tensor<1x256x128xf32>
    %196 = "tosa.matmul"(%195, %107) : (tensor<1x256x128xf32>, tensor<1x128x512xf32>) -> tensor<1x256x512xf32>
    %197 = "tosa.reshape"(%196) {new_shape = array<i64: 2, 128, 512>} : (tensor<1x256x512xf32>) -> tensor<2x128x512xf32>
    %198 = "tosa.add"(%197, %110) : (tensor<2x128x512xf32>, tensor<1x1x512xf32>) -> tensor<2x128x512xf32>
    %199 = "tosa.sub"(%198, %21) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %200 = "tosa.mul"(%199, %20) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %201 = "tosa.abs"(%200) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %202 = "tosa.mul"(%201, %19) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %203 = "tosa.add"(%202, %23) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %204 = "tosa.mul"(%201, %201) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %205 = "tosa.mul"(%204, %18) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %206 = "tosa.add"(%203, %205) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %207 = "tosa.mul"(%204, %201) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %208 = "tosa.mul"(%207, %17) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %209 = "tosa.add"(%206, %208) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %210 = "tosa.mul"(%207, %201) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %211 = "tosa.mul"(%210, %16) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %212 = "tosa.add"(%209, %211) : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %213 = "tosa.reciprocal"(%212) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %214 = "tosa.mul"(%213, %213) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %215 = "tosa.mul"(%214, %214) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %216 = "tosa.sub"(%23, %215) : (tensor<1x1x1xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %217 = "tosa.greater_equal"(%200, %21) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xi1>
    %218 = "tosa.negate"(%216) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %219 = "tosa.select"(%217, %216, %218) : (tensor<2x128x512xi1>, tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %220 = "tosa.add"(%219, %23) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %221 = "tosa.mul"(%220, %15) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    %222 = "tosa.mul"(%198, %221) {shift = 0 : i32} : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
    %223 = "tosa.reshape"(%222) {new_shape = array<i64: 1, 256, 512>} : (tensor<2x128x512xf32>) -> tensor<1x256x512xf32>
    %224 = "tosa.matmul"(%223, %140) : (tensor<1x256x512xf32>, tensor<1x512x128xf32>) -> tensor<1x256x128xf32>
    %225 = "tosa.reshape"(%224) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %226 = "tosa.add"(%225, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %227 = "tosa.add"(%226, %194) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %228 = "tosa.reduce_sum"(%227) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %229 = "tosa.mul"(%228, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %230 = "tosa.sub"(%227, %229) : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %231 = "tosa.mul"(%230, %230) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %232 = "tosa.reduce_sum"(%231) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %233 = "tosa.mul"(%232, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %234 = "tosa.add"(%233, %22) : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %235 = "tosa.rsqrt"(%234) : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %236 = "tosa.mul"(%230, %235) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %237 = "tosa.mul"(%236, %52) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %238 = "tosa.add"(%237, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %239 = "tosa.reshape"(%238) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x128xf32>) -> tensor<1x256x128xf32>
    %240 = "tosa.matmul"(%239, %62) : (tensor<1x256x128xf32>, tensor<1x128x128xf32>) -> tensor<1x256x128xf32>
    %241 = "tosa.reshape"(%240) {new_shape = array<i64: 2, 128, 128>} : (tensor<1x256x128xf32>) -> tensor<2x128x128xf32>
    %242 = "tosa.add"(%241, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %243 = "tosa.sub"(%242, %21) : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %244 = "tosa.mul"(%243, %20) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %245 = "tosa.abs"(%244) : (tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %246 = "tosa.mul"(%245, %19) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %247 = "tosa.add"(%246, %23) : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %248 = "tosa.mul"(%245, %245) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %249 = "tosa.mul"(%248, %18) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %250 = "tosa.add"(%247, %249) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %251 = "tosa.mul"(%248, %245) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %252 = "tosa.mul"(%251, %17) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %253 = "tosa.add"(%250, %252) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %254 = "tosa.mul"(%251, %245) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %255 = "tosa.mul"(%254, %16) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %256 = "tosa.add"(%253, %255) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %257 = "tosa.reciprocal"(%256) : (tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %258 = "tosa.mul"(%257, %257) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %259 = "tosa.mul"(%258, %258) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %260 = "tosa.sub"(%23, %259) : (tensor<1x1x1xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %261 = "tosa.greater_equal"(%244, %21) : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xi1>
    %262 = "tosa.negate"(%260) : (tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %263 = "tosa.select"(%261, %260, %262) : (tensor<2x128x128xi1>, tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %264 = "tosa.add"(%263, %23) : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %265 = "tosa.mul"(%264, %15) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x1xf32>) -> tensor<2x128x128xf32>
    %266 = "tosa.mul"(%242, %265) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %267 = "tosa.reduce_sum"(%266) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %268 = "tosa.mul"(%267, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %269 = "tosa.sub"(%266, %268) : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %270 = "tosa.mul"(%269, %269) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
    %271 = "tosa.reduce_sum"(%270) {axis = 2 : i64} : (tensor<2x128x128xf32>) -> tensor<2x128x1xf32>
    %272 = "tosa.mul"(%271, %46) {shift = 0 : i32} : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %273 = "tosa.add"(%272, %22) : (tensor<2x128x1xf32>, tensor<1x1x1xf32>) -> tensor<2x128x1xf32>
    %274 = "tosa.rsqrt"(%273) : (tensor<2x128x1xf32>) -> tensor<2x128x1xf32>
    %275 = "tosa.mul"(%269, %274) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x1xf32>) -> tensor<2x128x128xf32>
    %276 = "tosa.mul"(%275, %52) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %277 = "tosa.add"(%276, %52) : (tensor<2x128x128xf32>, tensor<1x1x128xf32>) -> tensor<2x128x128xf32>
    %278 = "tosa.transpose"(%9, %11) : (tensor<30522x128xf32>, tensor<2xi32>) -> tensor<128x30522xf32>
    %279 = "tosa.reshape"(%278) {new_shape = array<i64: 1, 128, 30522>} : (tensor<128x30522xf32>) -> tensor<1x128x30522xf32>
    %280 = "tosa.reshape"(%277) {new_shape = array<i64: 1, 256, 128>} : (tensor<2x128x128xf32>) -> tensor<1x256x128xf32>
    %281 = "tosa.transpose"(%279, %12) : (tensor<1x128x30522xf32>, tensor<3xi32>) -> tensor<128x1x30522xf32>
    %282 = "tosa.reshape"(%281) {new_shape = array<i64: 1, 128, 30522>} : (tensor<128x1x30522xf32>) -> tensor<1x128x30522xf32>
    %283 = "tosa.matmul"(%280, %282) : (tensor<1x256x128xf32>, tensor<1x128x30522xf32>) -> tensor<1x256x30522xf32>
    %284 = "tosa.reshape"(%283) {new_shape = array<i64: 2, 128, 30522>} : (tensor<1x256x30522xf32>) -> tensor<2x128x30522xf32>
    %285 = "tosa.reshape"(%2) {new_shape = array<i64: 1, 1, 30522>} : (tensor<30522xf32>) -> tensor<1x1x30522xf32>
    %286 = "tosa.add"(%284, %285) : (tensor<2x128x30522xf32>, tensor<1x1x30522xf32>) -> tensor<2x128x30522xf32>
    return %286 : tensor<2x128x30522xf32>
  }
}

