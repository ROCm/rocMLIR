// RUN: mlir-opt -miopen-lowering-step4 %s | FileCheck %s

#map11 = affine_map<(d0, d1) -> (d1, d0, d1, d0)>

#map12 = affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>

func @miopen_threadwise_copy_v2(%source_coord : i32, %dest_coord : memref<2xi32, 5>,
                                %source : vector<32xf32>, %dest : memref<?x?xf32>,
                                %dest_with_embedded_affine : memref<?x?xf32, #map11>,
                                %dest_with_externally_defined_affine : memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %dest_coord_y = load %dest_coord[%c0] : memref<2xi32, 5>
  %dest_coord_x = load %dest_coord[%c0] : memref<2xi32, 5>

  // check dest as a vanilla memref.
  miopen.threadwise_copy_v2(%source, %dest, %source_coord, %dest_coord_x, %dest_coord_y) : vector<32xf32>, memref<?x?xf32>

  // -----

  // check dest with embedded affine maps.
  miopen.threadwise_copy_v2(%source, %dest_with_embedded_affine, %source_coord, %dest_coord_x, %dest_coord_y) : vector<32xf32>, memref<?x?xf32, #map11>

  // -----

  // check destination with one externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_coord, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 1, transforms = [#map12] } ] } : vector<32xf32>, memref<?x?x?x?xf32>

  // check destination with multiple externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_coord, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 1, transforms = [#map12, #map13] } ] } : vector<32xf32>, memref<?x?x?x?xf32>

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy_v2
//  CHECK: miopen.threadwise_copy_v2
