// RUN: mlir-translate -mlir-to-miopencpp %s | FileCheck %s

// CHECK: Hello World
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
    ]
  } : memref<?x?x?x?xf32> to memref<?x?xf32>

  // input tensor
  %input_n_c_hipad_wipad = miopen.transform(%input) {
    layout = [
      {
        dimensions = [0],
        names = ["n"],
        transformation = "passthorugh",
        source_dimensions = [0],
        source_names = ["n"]
      },
      {
        dimensions = [1],
        names = ["c"],
        transformation = "passthorugh",
        source_dimensions = [1],
        source_names = ["c"]
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
    ]
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
    ]
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
        source_names = ["n", "ho", "wo"]
      }
    ]
  } : memref<?x?x?x?x?x?x?xf32> to memref<?x?xf32>
  
  // output tensor
  %output_gemmM_gemmN = miopen.transform(%output) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmM"],
        transformation = "passthrough",
        source_dimensions = [1],
        source_names = ["k"]
      },
      {
        dimensions = [1],
        names = ["gemmN"],
        transformation = "merge",
        source_dimensions = [0, 2, 3],
        source_names = ["n", "ho", "wo"]
      }
    ]
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
