// RUN: mlir-opt -miopen-affine-transform %s | FileCheck %s

// test 1-1 dimension mappings.
func @miopen_transform_1_to_1(%memref: memref<1x2x3x4xf32>) {
  %transformed_memref = miopen.transform(%memref) {
    layout = [
      {
        dimensions = [0],
        names = ["n"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["n"]
      },
      {
        dimensions = [1],
        names = ["c"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["c"]
      },
      {
        dimensions = [2],
        names = ["hi"],
        transformation = "Pad",
        parameters = [1, 1],
        source_dimensions = [2],
        source_names = ["hipad"]
      },
      {
        dimensions = [3],
        names = ["wi"],
        transformation = "Pad",
        parameters = [1, 1],
        source_dimensions = [3],
        source_names = ["wipad"]
      }
    ],
    source_layout = ["n", "c", "hi", "wi"],
    output_layout = ["n", "c", "hipad", "wipad"],
    domain = [1, 2, 3, 4],
    range = [1, 2, 5, 6]
  } : memref<1x2x3x4xf32> to memref<1x2x5x6xf32>
  return
}
// CHECK-LABEL: func @miopen_transform_1_to_1
//  CHECK-NEXT: %{{.*}} = miopen.transform(%{{.*}}) {domain = [1, 2, 3, 4]{{.*}}range = [1, 2, 5, 6]{{.*}}} : memref<1x2x3x4xf32> to memref<1x2x5x6xf32, #{{.*}}>

// test multiple source dimensions map to 1 target dimension.
func @miopen_transform_n_to_1(%memref : memref<1x2x3x4xf32>) {
  %transformed_memref = miopen.transform(%memref) {
    layout = [
      {
        dimensions = [0],
        names = ["gemmK"],
        transformation = "Merge",
        source_dimensions = [1, 2, 3],
        source_names = ["c", "y", "x"]
      },
      {
        dimensions = [1],
        names = ["gemmM"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["k"]
      }
    ],
    source_layout = ["k", "c", "y", "x"],
    output_layout = ["gemmK", "gemmM"],
    gridwise_gemm_argument_pos = 0,
    domain = [1, 2, 3, 4],
    range = [24, 1]
  } : memref<1x2x3x4xf32> to memref<24x1xf32>
  return
}
// CHECK-LABEL: func @miopen_transform_n_to_1
//  CHECK-NEXT: %{{.*}} = miopen.transform(%{{.*}}) {domain = [1, 2, 3, 4]{{.*}}range = [24, 1]{{.*}}} : memref<1x2x3x4xf32> to memref<24x1xf32, #{{.*}}>

// test 1 source dimension map to multiple target dimensions.
func @miopen_transform_1_to_n(%memref : memref<1x2x3x4xf32>) {
  %transformed_memref = miopen.transform(%memref) {
    layout = [
      {
        dimensions = [0],
        names = ["n"],
        transformation = "PassThrough",
        source_dimensions = [0],
        source_names = ["n"]
      },
      {
        dimensions = [1],
        names = ["c"],
        transformation = "PassThrough",
        source_dimensions = [1],
        source_names = ["c"]
      },
      {
        dimensions = [2, 3],
        names = ["y", "ho"],
        transformation = "Embed",
        parameters = [1, 1, 0],
        source_dimensions = [2],
        source_names = ["hipad"]
      },
      {
        dimensions = [4, 5],
        names = ["x", "wo"],
        transformation = "Embed",
        parameters = [1, 1, 0],
        source_dimensions = [3],
        source_names = ["wipad"]
      }
    ],
    intermediate_layout = ["n", "c", "hipad", "wipad"],
    output_layout = ["n", "c", "y", "ho", "x", "wo"],
    domain = [1, 2, 3, 4],
    range = [1, 2, 3, 3, 3, 4]
  } : memref<1x2x3x4xf32> to memref<1x2x3x3x3x4xf32>
  return
}

// CHECK-LABEL: func @miopen_transform_1_to_n
//  CHECK-NEXT: %{{.*}} = miopen.transform(%{{.*}}) {domain = [1, 2, 3, 4]{{.*}}range = [1, 2, 3, 3, 3, 4]{{.*}}} : memref<1x2x3x4xf32> to memref<1x2x3x3x3x4xf32, #{{.*}}>
