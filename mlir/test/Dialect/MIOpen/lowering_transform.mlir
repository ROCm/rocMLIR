// RUN: mlir-opt -miopen-affine-transform %s | FileCheck %s

// test 1-1 dimension mappings.
func @miopen_transform_1_to_1(%memref: memref<1x2x3x4xf32>) {
  %transformed_memref = miopen.transform(%memref) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["n"],
        transformation = "PassThrough",
        lower_layer_dimensions = [0],
        lower_layer_names = ["n"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["c"],
        transformation = "PassThrough",
        lower_layer_dimensions = [1],
        lower_layer_names = ["c"]
      },
      {
        upper_layer_dimensions = [2],
        upper_layer_names = ["hipad"],
        transformation = "Pad",
        parameters = [1, 1],
        lower_layer_dimensions = [2],
        lower_layer_names = ["hi"]
      },
      {
        upper_layer_dimensions = [3],
        upper_layer_names = ["wipad"],
        transformation = "Pad",
        parameters = [1, 1],
        lower_layer_dimensions = [3],
        lower_layer_names = ["wi"]
      }
    ],
    lower_layer_layout = ["n", "c", "hi", "wi"],
    upper_layer_layout = ["n", "c", "hipad", "wipad"],
    lower_layer_bounds = [1, 2, 3, 4],
    upper_layer_bounds = [1, 2, 5, 6]
  } : memref<1x2x3x4xf32> to memref<1x2x5x6xf32>
  return
}
// CHECK-LABEL: func @miopen_transform_1_to_1
//  CHECK-NEXT: %{{.*}} = miopen.transform(%{{.*}}) {{{.*}}lower_layer_bounds = [1, 2, 3, 4]{{.*}}upper_layer_bounds = [1, 2, 5, 6]{{.*}}} : memref<1x2x3x4xf32> to memref<1x2x5x6xf32, #{{.*}}>

// test multiple source dimensions map to 1 target dimension.
func @miopen_transform_n_to_1(%memref : memref<1x2x3x4xf32>) {
  %transformed_memref = miopen.transform(%memref) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["gemmK"],
        transformation = "Merge",
        lower_layer_dimensions = [1, 2, 3],
        lower_layer_names = ["c", "y", "x"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["gemmM"],
        transformation = "PassThrough",
        lower_layer_dimensions = [0],
        lower_layer_names = ["k"]
      }
    ],
    lower_layer_layout = ["k", "c", "y", "x"],
    upper_layer_layout = ["gemmK", "gemmM"],
    gridwise_gemm_argument_pos = 0,
    lower_layer_bounds = [1, 2, 3, 4],
    upper_layer_bounds = [24, 1]
  } : memref<1x2x3x4xf32> to memref<24x1xf32>
  return
}
// CHECK-LABEL: func @miopen_transform_n_to_1
//  CHECK-NEXT: %{{.*}} = miopen.transform(%{{.*}}) {{{.*}}lower_layer_bounds = [1, 2, 3, 4]{{.*}}upper_layer_bounds = [24, 1]{{.*}}} : memref<1x2x3x4xf32> to memref<24x1xf32, #{{.*}}>

// test 1 source dimension map to multiple target dimensions.
func @miopen_transform_1_to_n(%memref : memref<1x2x3x4xf32>) {
  %transformed_memref = miopen.transform(%memref) {
    layout = [
      {
        upper_layer_dimensions = [0],
        upper_layer_names = ["n"],
        transformation = "PassThrough",
        lower_layer_dimensions = [0],
        lower_layer_names = ["n"]
      },
      {
        upper_layer_dimensions = [1],
        upper_layer_names = ["c"],
        transformation = "PassThrough",
        lower_layer_dimensions = [1],
        lower_layer_names = ["c"]
      },
      {
        upper_layer_dimensions = [2, 3],
        upper_layer_names = ["y", "ho"],
        transformation = "Embed",
        parameters = [1, 1, 0],
        lower_layer_dimensions = [2],
        lower_layer_names = ["hipad"]
      },
      {
        upper_layer_dimensions = [4, 5],
        upper_layer_names = ["x", "wo"],
        transformation = "Embed",
        parameters = [1, 1, 0],
        lower_layer_dimensions = [3],
        lower_layer_names = ["wipad"]
      }
    ],
    lower_layer_layout = ["n", "c", "hipad", "wipad"],
    upper_layer_layout = ["n", "c", "y", "ho", "x", "wo"],
    lower_layer_bounds = [1, 2, 3, 4],
    upper_layer_bounds = [1, 2, 3, 3, 3, 4]
  } : memref<1x2x3x4xf32> to memref<1x2x3x3x3x4xf32>
  return
}

// CHECK-LABEL: func @miopen_transform_1_to_n
//  CHECK-NEXT: %{{.*}} = miopen.transform(%{{.*}}) {{{.*}}lower_layer_bounds = [1, 2, 3, 4]{{.*}}upper_layer_bounds = [1, 2, 3, 3, 3, 4]{{.*}}} : memref<1x2x3x4xf32> to memref<1x2x3x3x3x4xf32, #{{.*}}>
