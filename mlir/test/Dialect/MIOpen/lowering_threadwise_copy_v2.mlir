// RUN: mlir-opt -miopen-lowering-step4 %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 999 + d1 * 998)>

#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 8 + d2 * 4 + d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func @miopen_threadwise_copy_v2
func @miopen_threadwise_copy_v2(%source_offset : i32,
                                %source_coord : memref<2xi32, 5>,
                                %dest_coord : memref<2xi32, 5>,
                                %source : vector<32xf32>,
                                %dest1D : memref<32xf32>,
                                %dest2D : memref<?x?xf32>,
                                %dest_with_embedded_affine : memref<?x?xf32, #map0>,
                                %dest_with_externally_defined_affine : memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %c0_i32 = constant 0 : i32

  // check dest as a vanilla memref.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest1D, %c0_i32, %c0_i32, %c0_i32) {
    dim_access_order = [0 : i32],
    source_data_per_read = 1,
    dest_data_per_write = 1,
    vector_read_write_dim = 0
  } : vector<32xf32>, memref<32xf32>


  // check dest as a vanilla memref.
  // source has offset and bound.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest1D, %source_offset, %c0_i32, %c0_i32) {
    dim_access_order = [0],
    source_data_per_read = 1,
    dest_data_per_write = 1,
    vector_read_write_dim = 0,
    coord_transforms = [
      { operand = 0, transforms = [affine_map<(d0) -> (d0)>] }
    ],
    bound = [16 : i32]
  } : vector<32xf32>, memref<32xf32>

  // ----- 

  // check source with one externally defined affine map.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest2D, %source_offset, %c0_i32, %c0_i32, %c0_i32, %c0_i32) {
    dim_access_order = [0, 1],
    source_data_per_read = 1,
    dest_data_per_write = 1,
    vector_read_write_dim = 1,

    coord_transforms = [
      { operand = 0, transforms = [#map0] }
    ],
    bound = [8 : i32, 4 : i32]
  } : vector<32xf32>, memref<?x?xf32>

  // check source with multiple externally defined affine map.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest2D, %source_offset, %c0_i32, %c0_i32, %c0_i32, %c0_i32) {
    dim_access_order = [0, 1],
    source_data_per_read = 1,
    dest_data_per_write = 1,
    vector_read_write_dim = 1,

    coord_transforms = [
      { operand = 0, transforms = [#map0, #map1] }
    ],
    bound = [8 : i32, 4 : i32]
  } : vector<32xf32>, memref<?x?xf32>

  // -----

  // check source and destination with one externally defined affine map.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) {
    dim_access_order = [0, 1, 2, 3],
    source_data_per_read = 1,
    dest_data_per_write = 1,
    vector_read_write_dim = 3,

    coord_transforms = [
      { operand = 0, transforms = [#map2] },
      { operand = 1, transforms = [#map3] }
    ],
    bound = [2, 2, 2, 4]
  } : vector<32xf32>, memref<?x?x?x?xf32>

  // check source and destination with one externally defined affine map.
  // only read half of the source vector with bound attribute.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) {
    dim_access_order = [0, 1, 2, 3],
    source_data_per_read = 1,
    dest_data_per_write = 1,
    vector_read_write_dim = 3,

    coord_transforms = [
      { operand = 0, transforms = [#map2] },
      { operand = 1, transforms = [#map3] }
    ],
    bound = [2, 2, 2, 2]
  } : vector<32xf32>, memref<?x?x?x?xf32>

  return
}

