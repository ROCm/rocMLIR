// RUN: mlir-opt -miopen-lowering-step4 %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 4 + d2)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 4 + d3)>

#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 4 + d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 8 + d2 * 4 + d3, d4)>
#map9 = affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, (d2 mod 196) floordiv 14, (d2 mod 196) mod 14)>

// CHECK-LABEL: func @miopen_threadwise_copy_v2
func @miopen_threadwise_copy_v2(%source_offset : i32,
                                %source : vector<32xf32>,
                                %dest1D : memref<32xf32>,
                                %dest5D : memref<128x1x1024x14x14xf32>) {
  %c0_i32 = constant 0 : i32

  // A simplified usage of threadwise_copy_v2.
  // Source vector has a transformation.
  // Source vector has no offset.
  // Source vector has a bound.
  // Dest memref has a transformation.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest1D, %c0_i32,
                            %c0_i32, %c0_i32, %c0_i32,
                            %c0_i32, %c0_i32, %c0_i32) {
    dim_access_order = [0 : i32, 1 : i32, 2 : i32],
    source_data_per_read = 1,
    dest_data_per_write = 1,
    vector_read_write_dim = 0,
    bound = [1 : i32, 8 : i32, 4 : i32],
    coord_transforms = [
      {
        operand = 0 : i32, transforms = [#map0],
        metadata = [
          {
            map = [#map0],
            layout = [
              {
                 lower_layer_dimensions = [0 : i32],
                 lower_layer_names = ["raw"],
                 transformation = "UnMerge",
                 parameters = [32 : i32, 4 : i32, 1 : i32],
                 upper_layer_dimensions = [0 : i32, 1 : i32, 2 : i32],
                 upper_layer_names = ["no", "ho", "wo"]
              }
            ],
            lower_layer_bounds = [32 : i32],
            lower_layer_layout = ["vector"],
            lowest_layer = true,
            upper_layer_bounds = [1 : i32, 8 : i32, 4 : i32],
            upper_layer_layout = ["no", "ho", "wo"]
          }
        ]
      }, 
      {
        operand = 1 : i32, transforms = [#map0], 
        domain = [1 : i32, 8 : i32, 4 : i32],
        metadata = [
          {
            map = [#map0],
            layout = [
              {
                 lower_layer_dimensions = [0 : i32],
                 lower_layer_names = ["raw"],
                 transformation = "UnMerge",
                 parameters = [32 : i32, 4 : i32, 1 : i32],
                 upper_layer_dimensions = [0 : i32, 1 : i32, 2 : i32],
                 upper_layer_names = ["no", "ho", "wo"]
              }
            ], 
            lower_layer_bounds = [32 : i32], 
            lower_layer_layout = ["vector"], 
            lowest_layer = true,
            upper_layer_bounds = [1 : i32, 8 : i32, 4 : i32],
            upper_layer_layout = ["no", "ho", "wo"]
          }
        ]
      }
    ]
  } : vector<32xf32>, memref<32xf32>

  // A real use case of threadwise_copy_v2.
  // Source vector has a transformation.
  // Source vector has offset and bound.
  // Dest memref has 2 transformations.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest5D, %source_offset, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) {bound = [1 : i32, 4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{metadata = [{layout = [{lower_layer_dimensions = [0 : i32], lower_layer_names = ["raw"], parameters = [16 : i32, 4 : i32, 4 : i32, 1 : i32, 1 : i32], transformation = "UnMerge", upper_layer_dimensions = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], upper_layer_names = ["dim0", "m3", "dim2", "m2", "dim4"]}], lower_layer_bounds = [16 : i32], lower_layer_layout = ["raw"], map = [#map7], upper_layer_bounds = [1 : i32, 4 : i32, 1 : i32, 4 : i32, 1 : i32], upper_layer_layout = ["dim0", "m3", "dim2", "m2", "dim4"]}], operand = 0 : i32, transforms = [#map7]}, {domain = [1 : i32, 128 : i32, 2 : i32, 4 : i32, 25088 : i32], metadata = [{layout = [{lower_layer_dimensions = [0 : i32], lower_layer_names = ["gemmG"], transformation = "PassThrough", upper_layer_dimensions = [0 : i32], upper_layer_names = ["g"]}, {lower_layer_dimensions = [1 : i32], lower_layer_names = ["gemmM"], parameters = [8 : i32, 4 : i32, 1 : i32], transformation = "UnMerge", upper_layer_dimensions = [1 : i32, 2 : i32, 3 : i32], upper_layer_names = ["m0", "m1", "m2"]}, {lower_layer_dimensions = [2 : i32], lower_layer_names = ["gemmN"], transformation = "PassThrough", upper_layer_dimensions = [4 : i32], upper_layer_names = ["n"]}], lower_layer_bounds = [1 : i32, 1024 : i32, 25088 : i32], lower_layer_layout = ["gemmG", "gemmM", "gemmN"], map = [#map8], upper_layer_bounds = [1 : i32, 128 : i32, 2 : i32, 4 : i32, 25088 : i32], upper_layer_layout = ["g", "m0", "m1", "m2", "n"]}, {extraPad = false, gemmMExtra = 0 : i32, gemmNExtra = 0 : i32, gridwise_gemm_argument_position = 2 : i32, layout = [{lower_layer_dimensions = [1 : i32], lower_layer_names = ["go"], transformation = "PassThrough", upper_layer_dimensions = [0 : i32], upper_layer_names = ["gemmG"]}, {lower_layer_dimensions = [2 : i32], lower_layer_names = ["ko"], transformation = "PassThrough", upper_layer_dimensions = [1 : i32], upper_layer_names = ["gemmM"]}, {lower_layer_dimensions = [0 : i32, 3 : i32, 4 : i32], lower_layer_names = ["no", "ho", "wo"], transformation = "Merge", upper_layer_dimensions = [2 : i32], upper_layer_names = ["gemmN"]}], lower_layer_bounds = [128 : i32, 1 : i32, 1024 : i32, 14 : i32, 14 : i32], lower_layer_layout = ["no", "go", "ko", "ho", "wo"], lowest_layer = true, map = [#map9], upper_layer_bounds = [1 : i32, 1024 : i32, 25088 : i32], upper_layer_layout = ["gemmG", "gemmM", "gemmN"]}], operand = 1 : i32, transforms = [#map8, #map9]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 4 : i32} : vector<32xf32>, memref<128x1x1024x14x14xf32>

  return
}

