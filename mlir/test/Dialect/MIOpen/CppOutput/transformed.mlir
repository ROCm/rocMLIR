// RUN: mlir-translate -mlir-to-miopen-cpp %s | FileCheck %s

// CHECK:  __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_implicit_gemm_v4r4_kcyx_niciwihi_nokohowo
func @miopen_transformed_conv2d(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  // filter tensor
  %filter_gemmK_gemmM = miopen.transform(%filter) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmK"],
        transformation = "merge",
        source_dimensions = [1, 2, 3],
        source_names = ["c", "y", "x"]
      },
      {
        dimensions = [1],
        names = ["gemmM"],
        transformation = "passthrough",
        source_dimensions = [0],
        source_names = ["n"]
      }
    ],
    source_layout = ["k", "c", "y", "x"]
  } : memref<?x?x?x?xf32> to memref<?x?xf32>

  // input tensor
  %input_n_c_hipad_wipad = miopen.transform(%input) {
    layout = [
      {
        dimensions = [0],
        names = ["n"],
        transformation = "passthorugh",
        source_dimensions = [0],
        source_names = ["ni"]
      },
      {
        dimensions = [1],
        names = ["c"],
        transformation = "passthorugh",
        source_dimensions = [1],
        source_names = ["ci"]
      },
      {
        dimensions = [2],
        names = ["hipad"],
        transformation = "pad",
        parameters = [0, 0],
        source_dimensions = [2],
        source_names = ["hi"]
      },
      {
        dimensions = [3],
        names = ["wipad"],
        transformation = "pad",
        parameters = [0, 0],
        source_dimensions = [3],
        source_names = ["wi"]
      }
    ],
    source_layout = ["ni", "ci", "wi", "hi"]
  } : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
  
  %input_n_c_y_ho_x_wo = miopen.transform(%input_n_c_hipad_wipad) {
    layout = [
      {
        dimensions = [0],
        names = ["n"],
        transformation = "passthrough",
        source_dimensions = [0],
        source_names = ["n"]
      },
      {
        dimensions = [1],
        names = ["c"],
        transformation = "passthrough",
        source_dimensions = [1],
        source_names = ["c"]
      },
      {
        dimensions = [2, 3],
        names = ["y", "ho"],
        transformation = "embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [2],
        source_names = ["hipad"]
      },
      {
        dimensions = [4, 5],
        names = ["x", "wo"],
        transformation = "embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [2],
        source_names = ["wipad"]
      }
    ],
    intermediate_layout = ["n", "c", "hipad", "wipad"]
  } : memref<?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32>
  
  %input_gemmK_gemmN = miopen.transform(%input_n_c_y_ho_x_wo) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmK"],
        transformation = "merge",
        source_dimensions = [1, 2, 4],
        source_names = ["c", "y", "x"]
      },
      {
        dimensions = [1],
        names = ["gemmN"],
        transformation = "merge",
        source_dimensions = [0, 3, 5],
        source_names = ["n", "hipad", "wipad"]
      }
    ],
    intermediate_layout = ["n", "c", "y", "hipad", "x", "wipad"]
  } : memref<?x?x?x?x?x?x?xf32> to memref<?x?xf32>
  
  // output tensor
  %output_gemmM_gemmN = miopen.transform(%output) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmM"],
        transformation = "passthrough",
        source_dimensions = [1],
        source_names = ["ko"]
      },
      {
        dimensions = [1],
        names = ["gemmN"],
        transformation = "merge",
        source_dimensions = [0, 2, 3],
        source_names = ["no", "ho", "wo"]
      }
    ],
    source_layout = ["no", "ko", "ho", "wo"]
  } : memref<?x?x?x?xf32> to memref<?x?xf32>
  
  // apply gridwise GEMM
  miopen.gridwise_gemm(%filter_gemmK_gemmM, %input_gemmK_gemmN, %output_gemmM_gemmN) {
    parameters = [
      // tuning parameters
    ]
  } : memref<?x?xf32>,
      memref<?x?xf32>,
      memref<?x?xf32>

  return
}
// CHECK:    constexpr auto weight_k_c_y_x_desc = make_native_tensor_descriptor(Sequence<k, c, y, x>{}, Sequence<stride_k, stride_c, stride_y, stride_x>{});
// CHECK:     constexpr auto input_ni_ci_wi_hi_desc = make_native_tensor_descriptor(Sequence<ni, ci, wi, hi>{}, Sequence<stride_ni, stride_ci, stride_wi, stride_hi>{});
// CHECK:     constexpr auto output_no_ko_ho_wo_desc = make_native_tensor_descriptor(Sequence<no, ko, ho, wo>{}, Sequence<stride_no, stride_ko, stride_ho, stride_wo>{});
// CHECK:         constexpr auto gridwise_conv = GridwiseConvolutionImplicitGemm_v4r4_kcyx_niciwihi_nokohowo
// CHECK:        decltype(weight_k_c_y_x_desc),
// CHECK:        decltype(input_ni_ci_wi_hi_desc),
// CHECK:        decltype(output_no_ko_ho_wo_desc),
