// RUN: mlir-translate -mlir-to-miopen-cpp %s | FileCheck -check-prefix=MIOPEN-CPP %s
// RUN: mlir-translate -mlir-to-miopen-hpp %s | FileCheck -check-prefix=MIOPEN-HPP %s

// MIOPEN-CPP:  __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void mlir_gen_igemm_conv2d_cpp_v1r1_bwd
// MIOPEN-CPP:  FLOAT* const __restrict__ p_in_global
// MIOPEN-HPP: struct MlirGenIgemmConv2dV1r1Bwd
func @miopen_transformed_conv2d(%filter : memref<?x?x?x?xf32>, %input : memref<?x?x?x?xf32>, %output : memref<?x?x?x?xf32>) {
  // filter tensor
  %filter_gemmK_gemmM = miopen.transform(%filter) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmK"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["k"]
      },
      {
        dimensions = [1],
        names = ["gemmM"],
        transformation = "Merge",
        source_dimensions = [1, 2, 3],
        source_names = ["c", "y", "x"]
      }
    ],
    source_layout = ["k", "c", "y", "x"],
    output_layout = ["gemmK", "gemmM"],
    gridwise_gemm_argument_position = 0
  } : memref<?x?x?x?xf32> to memref<?x?xf32>

  // input tensor
  %input_ni_ci_hipad_wipad = miopen.transform(%input) {
    layout = [
      {
        dimensions = [0],
        names = ["ni"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["ni"]
      },
      {
        dimensions = [1],
        names = ["ci"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["ci"]
      },
      {
        dimensions = [2, 3],
        names = ["hipad", "wipad"],
        transformation = "Pad",
        parameters = [0, 0],
        source_dimensions = [2, 3],
        source_names = ["hi", "wi"]
      }
    ],
    source_layout = ["ni", "ci", "hi", "wi"],
    output_layout = ["ni", "ci", "hipad", "wipad"]
  } : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
  
  %input_ni_ci_y_ho_x_wo = miopen.transform(%input_ni_ci_hipad_wipad) {
    layout = [
      {
        dimensions = [0],
        names = ["ni"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["ni"]
      },
      {
        dimensions = [1],
        names = ["ci"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["ci"]
      },
      {
        dimensions = [2, 3],
        names = ["y", "ho"],
        transformation = "Embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [2],
        source_names = ["hipad"]
      },
      {
        dimensions = [4, 5],
        names = ["x", "wo"],
        transformation = "Embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [3],
        source_names = ["wipad"]
      }
    ],
    intermediate_layout = ["ni", "ci", "hipad", "wipad"],
    output_layout = ["ni", "ci", "y", "ho", "x", "wo"]
  } : memref<?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32>
  
  %input_gemmM_gemmN = miopen.transform(%input_ni_ci_y_ho_x_wo) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmM"],
        transformation = "Merge",
        source_dimensions = [1, 2, 4],
        source_names = ["ci", "y", "x"]
      },
      {
        dimensions = [1],
        names = ["gemmN"],
        transformation = "Merge",
        source_dimensions = [0, 3, 5],
        source_names = ["ni", "ho", "wo"]
      }
    ],
    intermediate_layout = ["ni", "ci", "y", "ho", "x", "wo"],
    output_layout = ["gemmM", "gemmN"],
    gridwise_gemm_argument_position = 2
  } : memref<?x?x?x?x?x?x?xf32> to memref<?x?xf32>
  
  // output tensor
  %output_gemmK_gemmN = miopen.transform(%output) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmK"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["ko"]
      },
      {
        dimensions = [1],
        names = ["gemmN"],
        transformation = "Merge",
        source_dimensions = [0, 2, 3],
        source_names = ["no", "ho", "wo"]
      }
    ],
    source_layout = ["no", "ko", "ho", "wo"],
    output_layout = ["gemmK", "gemmN"],
    gridwise_gemm_argument_position = 1
  } : memref<?x?x?x?xf32> to memref<?x?xf32>
  
  // apply gridwise GEMM
  miopen.gridwise_gemm(%filter_gemmK_gemmM, %output_gemmK_gemmN, %input_gemmM_gemmN) {
    // tuning parameters
    kernel_algorithm = "backward_data_v1r1",
    filter_dimension = [128, 8, 3, 3],
    filter_layout = ["k", "c", "y", "x"],
    input_dimension = [128, 8, 32, 32],
    input_layout = ["ni", "ci", "hi", "wi"],
    output_dimension = [128, 128, 30, 30],
    output_layout = ["no", "ko", "ho", "wo"]
  } : memref<?x?xf32>,
      memref<?x?xf32>,
      memref<?x?xf32>

  return
}
// MIOPEN-CPP:    constexpr auto weight_k_c_y_x_desc = make_native_tensor_descriptor(Sequence<k, c, y, x>{}, Sequence<stride_k, stride_c, stride_y, stride_x>{});
// MIOPEN-CPP:     constexpr auto input_ni_ci_hi_wi_desc = make_native_tensor_descriptor(Sequence<ni, ci, hi, wi>{}, Sequence<stride_ni, stride_ci, stride_hi, stride_wi>{});
// MIOPEN-CPP:     constexpr auto output_no_ko_ho_wo_desc = make_native_tensor_descriptor(Sequence<no, ko, ho, wo>{}, Sequence<stride_no, stride_ko, stride_ho, stride_wo>{});
// MIOPEN-CPP:         constexpr auto gridwise_conv = MlirGenIgemmConv2dV1r1Bwd
// MIOPEN-CPP:        decltype(input_ni_ci_hi_wi_desc),
// MIOPEN-CPP:        decltype(weight_k_c_y_x_desc),
// MIOPEN-CPP:        decltype(output_no_ko_ho_wo_desc),
