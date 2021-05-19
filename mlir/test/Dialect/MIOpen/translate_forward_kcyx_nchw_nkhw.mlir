// RUN: mlir-translate -mlir-to-miopen-cpp %s | FileCheck -check-prefix=MIOPEN-CPP %s
// RUN: mlir-translate -mlir-to-miopen-hpp %s | FileCheck -check-prefix=MIOPEN-HPP %s

// MIOPEN-CPP:  __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void mlir_gen_igemm_conv2d_cpp_v4r4_fwd
// MIOPEN-CPP:  FLOAT* const __restrict__ p_out_global
// MIOPEN-HPP: struct MlirGenIgemmConv2dV4r4Fwd
func @miopen_transformed_conv2d(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  // filter tensor
  %filter_gemmG_gemmK_gemmM = miopen.transform(%filter) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmG"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["k"]
      },
      {
        dimensions = [1],
        names = ["gemmK"],
        transformation = "Merge",
        source_dimensions = [2, 3, 4],
        source_names = ["c", "y", "x"]
      },
      {
        dimensions = [2],
        names = ["gemmM"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["k"]
      }
    ],
    source_layout = ["g", "k", "c", "y", "x"],
    output_layout = ["gemmG", "gemmK", "gemmM"],
    gridwise_gemm_argument_position = 0
  } : memref<?x?x?x?x?xf32> to memref<?x?x?xf32>

  // input tensor
  %input_ni_gi_ci_hipad_wipad = miopen.transform(%input) {
    layout = [
      {
        dimensions = [0],
        names = ["gi"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["gi"]
      },
      {
        dimensions = [1],
        names = ["ni"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["ni"]
      },
      {
        dimensions = [2],
        names = ["ci"],
        transformation = "PassThrough",
        source_dimensions = [2],
        source_names = ["ci"]
      },
      {
        dimensions = [3, 4],
        names = ["hipad", "wipad"],
        transformation = "Pad",
        parameters = [0, 0, 0 , 0],
        source_dimensions = [3, 4],
        source_names = ["hi", "wi"]
      }
    ],
    source_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["gi", "ni", "ci", "hipad", "wipad"]
  } : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?xf32>
  
  %input_gi_ni_ci_y_ho_x_wo = miopen.transform(%input_ni_gi_ci_hipad_wipad) {
    layout = [
      {
        dimensions = [0],
        names = ["gi"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["gi"]
      },
      {
        dimensions = [1],
        names = ["ni"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["ni"]
      },
      {
        dimensions = [2],
        names = ["ci"],
        transformation = "PassThrough",
        source_dimensions = [2],
        source_names = ["ci"]
      },
      {
        dimensions = [3, 4],
        names = ["y", "ho"],
        transformation = "Embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [3],
        source_names = ["hipad"]
      },
      {
        dimensions = [5, 6],
        names = ["x", "wo"],
        transformation = "Embed",
        parameters = [2, [1, 1, 0]],
        source_dimensions = [4],
        source_names = ["wipad"]
      }
    ],
    intermediate_layout = ["gi", "ni", "ci", "hipad", "wipad"],
    output_layout = ["gi", "ni", "ci", "y", "ho", "x", "wo"]
  } : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?x?x?x?xf32>
  
  %input_gemmG_gemmK_gemmN = miopen.transform(%input_gi_ni_ci_y_ho_x_wo) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmG"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["Gi"]
      },
      {
        dimensions = [1],
        names = ["gemmK"],
        transformation = "Merge",
        source_dimensions = [2, 3, 5],
        source_names = ["ci", "y", "x"]
      },
      {
        dimensions = [2],
        names = ["gemmN"],
        transformation = "Merge",
        source_dimensions = [1, 4, 6],
        source_names = ["ni", "ho", "wo"]
      }
    ],
    intermediate_layout = ["gi", "ni", "ci", "y", "ho", "x", "wo"],
    output_layout = ["gemmG", "gemmK", "gemmN"],
    gridwise_gemm_argument_position = 1
  } : memref<?x?x?x?x?x?x?x?xf32> to memref<?x?x?xf32>
  
  // output tensor
  %output_gemmG_gemmM_gemmN = miopen.transform(%output) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmG"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["go"]
      },
      {
        dimensions = [1],
        names = ["gemmM"],
        transformation = "PassThrough",
        source_dimensions = [2],
        source_names = ["ko"]
      },
      {
        dimensions = [2],
        names = ["gemmN"],
        transformation = "Merge",
        source_dimensions = [0, 3, 4],
        source_names = ["no", "ho", "wo"]
      }
    ],
    source_layout = ["no", "go", "ko", "ho", "wo"],
    output_layout = ["gemmG", "gemmM", "gemmN"],
    gridwise_gemm_argument_position = 2
  } : memref<?x?x?x?x?xf32> to memref<?x?x?xf32>
  
  // apply gridwise GEMM
  miopen.gridwise_gemm(%filter_gemmG_gemmK_gemmM, %input_gemmG_gemmK_gemmN, %output_gemmG_gemmM_gemmN) {
    // tuning parameters
    kernel_algorithm = "v4r4",
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
// MIOPEN-CPP:         constexpr auto gridwise_conv = MlirGenIgemmConv2dV4r4Fwd
// MIOPEN-CPP:        decltype(input_ni_gi_ci_hi_wi_desc),
// MIOPEN-CPP:        decltype(weight_g_k_c_y_x_desc),
// MIOPEN-CPP:        decltype(output_no_go_ko_ho_wo_desc),
