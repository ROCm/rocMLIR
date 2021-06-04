// RUN: mlir-translate -mlir-to-miopen-cpp %s | FileCheck -check-prefix=MIOPEN-CPP %s
// RUN: mlir-translate -mlir-to-miopen-hpp %s | FileCheck -check-prefix=MIOPEN-HPP %s

// MIOPEN-CPP:  __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void mlir_gen_igemm_conv2d_cpp_v4r1_bwd
// MIOPEN-CPP:  FLOAT* const __restrict__ p_in_global
// MIOPEN-HPP: struct MlirGenIgemmConv2dV1r1Bwd
func @miopen_transformed_conv2d(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  // filter tensor
  %filter_gemmK_gemmM = miopen.transform(%filter) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["gemmG"],
        transformation = "PassThrough",
        lower_layer_dimensions = [0],
        lower_layer_names = ["g"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["gemmK"],
        transformation = "PassThrough",
        lower_layer_dimensions = [1],
        lower_layer_names = ["k"]
      },
      {
        upper_layer_dimensions = [2],
        upper_layer_names = ["gemmM"],
        transformation = "Merge",
        lower_layer_dimensions = [2, 3, 4],
        lower_layer_names = ["c", "y", "x"]
      }
    ],
    lower_layer_layout = ["g", "k", "c", "y", "x"],
    upper_layer_layout = ["gemmG", "gemmK", "gemmM"],
    gridwise_gemm_argument_position = 0,
    lowest_layer = true
  } : memref<?x?x?x?x?xf32> to memref<?x?x?xf32>

  // input tensor
  %input_ni_gi_ci_hipad_wipad = miopen.transform(%input) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["gi"],
        transformation = "PassThrough",
        lower_layer_dimensions = [1],
        lower_layer_names = ["gi"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["ni"],
        transformation = "PassThrough",
        lower_layer_dimensions = [0],
        lower_layer_names = ["ni"]
      },
      {
        upper_layer_dimensions = [2],
        upper_layer_names = ["ci"],
        transformation = "PassThrough",
        lower_layer_dimensions = [2],
        lower_layer_names = ["ci"]
      },
      {
        upper_layer_dimensions = [3, 4],
        upper_layer_names = ["hipad", "wipad"],
        transformation = "Pad",
        parameters = [0, 0, 0, 0],
        lower_layer_dimensions = [3, 4],
        lower_layer_names = ["hi", "wi"]
      }
    ],
    lower_layer_layout = ["ni", "gi", "ci", "hi", "wi"],
    upper_layer_layout = ["gi", "ni", "ci", "hipad", "wipad"],
    lowest_layer = true
  } : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?xf32>
  
  %input_gi_ni_ci_y_ho_x_wo = miopen.transform(%input_ni_gi_ci_hipad_wipad) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["gi"],
        transformation = "PassThrough",
        lower_layer_dimensions = [0],
        lower_layer_names = ["gi"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["ni"],
        transformation = "PassThrough",
        lower_layer_dimensions = [1],
        lower_layer_names = ["ni"]
      },
      {
        upper_layer_dimensions = [2],
        upper_layer_names = ["ci"],
        transformation = "PassThrough",
        lower_layer_dimensions = [2],
        lower_layer_names = ["ci"]
      },
      {
        upper_layer_dimensions = [3, 4],
        upper_layer_names = ["y", "ho"],
        transformation = "Embed",
        parameters = [2, [1, 1, 0]],
        lower_layer_dimensions = [3],
        lower_layer_names = ["hipad"]
      },
      {
        upper_layer_dimensions = [5, 6],
        upper_layer_names = ["x", "wo"],
        transformation = "Embed",
        parameters = [2, [1, 1, 0]],
        lower_layer_dimensions = [4],
        lower_layer_names = ["wipad"]
      }
    ],
    lower_layer_layout = ["ni", "gi", "ci", "hipad", "wipad"],
    upper_layer_layout = ["gi", "ni", "ci", "y", "ho", "x", "wo"]
  } : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?x?x?x?xf32>
  
  %input_gemmM_gemmN = miopen.transform(%input_gi_ni_ci_y_ho_x_wo) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["gemmG"],
        transformation = "Merge",
        lower_layer_dimensions = [0],
        lower_layer_names = ["gi"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["gemmM"],
        transformation = "Merge",
        lower_layer_dimensions = [2, 3, 5],
        lower_layer_names = ["ci", "y", "x"]
      },
      {
        upper_layer_dimensions = [2],
        upper_layer_names = ["gemmN"],
        transformation = "Merge",
        lower_layer_dimensions = [1, 4, 6],
        lower_layer_names = ["ni", "ho", "wo"]
      }
    ],
    lower_layer_layout = ["gi", "ni", "ci", "y", "ho", "x", "wo"],
    upper_layer_layout = ["gemmG", "gemmM", "gemmN"],
    gridwise_gemm_argument_position = 2
  } : memref<?x?x?x?x?x?x?x?xf32> to memref<?x?x?xf32>
  
  // output tensor
  %output_gemmK_gemmN = miopen.transform(%output) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["gemmG"],
        transformation = "PassThrough",
        lower_layer_dimensions = [1],
        lower_layer_names = ["go"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["gemmK"],
        transformation = "PassThrough",
        lower_layer_dimensions = [2],
        lower_layer_names = ["ko"]
      },
      {
        upper_layer_dimensions = [2],
        upper_layer_names = ["gemmN"],
        transformation = "Merge",
        lower_layer_dimensions = [0, 3, 4],
        lower_layer_names = ["no", "ho", "wo"]
      }
    ],
    lower_layer_layout = ["no", "go", "ko", "ho", "wo"],
    upper_layer_layout = ["gemmG", "gemmK", "gemmN"],
    gridwise_gemm_argument_position = 1,
    lowest_layer = true
  } : memref<?x?x?x?x?xf32> to memref<?x?x?xf32>
  
  // apply gridwise GEMM
  miopen.gridwise_gemm(%filter_gemmK_gemmM, %output_gemmK_gemmN, %input_gemmM_gemmN) {
    // tuning parameters
    kernel_algorithm = "backward_data_v4r1",
    filter_dimension = [1, 128, 8, 3, 3],
    filter_layout = ["g", "k", "c", "y", "x"],
    input_dimension = [128, 1, 8, 32, 32],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_dimension = [128, 1, 128, 30, 30],
    output_layout = ["no", "go", "ko", "ho", "wo"]
  } : memref<?x?x?xf32>,
      memref<?x?x?xf32>,
      memref<?x?x?xf32>

  return
}
// MIOPEN-CPP:    constexpr auto weight_g_k_c_y_x_desc = make_native_tensor_descriptor(Sequence<g, k, c, y, x>{}, Sequence<stride_g, stride_k, stride_c, stride_y, stride_x>{});
// MIOPEN-CPP:     constexpr auto input_ni_gi_ci_hi_wi_desc = make_native_tensor_descriptor(Sequence<ni, gi, ci, hi, wi>{}, Sequence<stride_ni, stride_gi, stride_ci, stride_hi, stride_wi>{});
// MIOPEN-CPP:     constexpr auto output_no_go_ko_ho_wo_desc = make_native_tensor_descriptor(Sequence<no, go, ko, ho, wo>{}, Sequence<stride_no, stride_go, stride_ko, stride_ho, stride_wo>{});
// MIOPEN-CPP:         constexpr auto gridwise_conv = MlirGenIgemmConv2dV1r1Bwd
// MIOPEN-CPP:        decltype(input_ni_gi_ci_hi_wi_desc),
// MIOPEN-CPP:        decltype(weight_g_k_c_y_x_desc),
// MIOPEN-CPP:        decltype(output_no_go_ko_ho_wo_desc),
