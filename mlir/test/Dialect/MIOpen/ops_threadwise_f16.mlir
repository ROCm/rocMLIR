// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1, d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0, d1, d0)>

#map2 = affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>

func @miopen_threadwise_copy_f16(%source_coord : memref<2xi32, 5>, %dest_coord : memref<2xi32, 5>,
                             %source : memref<?x?xf16, 5>, %dest : memref<?x?xf16, 5>,
                             %source_with_embedded_affine : memref<?x?xf16, #map0, 3>,
                             %dest_with_embedded_affine : memref<?x?xf16, #map1, 3>,
                             %source_with_externally_defined_affine : memref<?x?x?x?xf16>,
                             %dest_with_externally_defined_affine : memref<?x?x?x?xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_coord_y = memref.load %source_coord[%c0] : memref<2xi32, 5>
  %source_coord_x = memref.load %source_coord[%c0] : memref<2xi32, 5>
  %dest_coord_y = memref.load %dest_coord[%c0] : memref<2xi32, 5>
  %dest_coord_x = memref.load %dest_coord[%c0] : memref<2xi32, 5>

  // check source and dest as vanilla memrefs.
  miopen.threadwise_copy(%source, %dest,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
  : memref<?x?xf16, 5>, memref<?x?xf16, 5>, i32, i32, i32, i32

  // -----

  // check source with embedded affine maps.
  miopen.threadwise_copy(%source_with_embedded_affine, %dest,
  %source_coord_x, %source_coord_y,
  %dest_coord_x, %dest_coord_y)
  : memref<?x?xf16, #map0, 3>, memref<?x?xf16, 5>, i32, i32, i32, i32

  // check dest with embedded affine maps.
  miopen.threadwise_copy(%source, %dest_with_embedded_affine,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    : memref<?x?xf16, 5>, memref<?x?xf16, #map1, 3>, i32, i32, i32, i32

  // check source and dest with embedded affine maps.
  miopen.threadwise_copy(%source_with_embedded_affine, %dest_with_embedded_affine,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    : memref<?x?xf16, #map0, 3>, memref<?x?xf16, #map1, 3>, i32, i32, i32, i32

  // -----

  // check source with one externally defined affine map.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [ { operand = 0, transforms = [#map2] } ]
    } : memref<?x?x?x?xf16>, memref<?x?xf16, 5>, i32, i32, i32, i32

  // check source with multiple externally defined affine maps.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [ { operand = 0, transforms = [#map2, #map3] } ]
    } : memref<?x?x?x?xf16>, memref<?x?xf16, 5>, i32, i32, i32, i32

  // check destination with one externally defined affine map.
  miopen.threadwise_copy(%source, %dest_with_externally_defined_affine,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [ { operand = 1, transforms = [#map2] } ]
    } : memref<?x?xf16, 5>, memref<?x?x?x?xf16>, i32, i32, i32, i32

  // check destination with multiple externally defined affine map.
  miopen.threadwise_copy(%source, %dest_with_externally_defined_affine,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [ { operand = 1, transforms = [#map2, #map3] } ]
    } : memref<?x?xf16, 5>, memref<?x?x?x?xf16>, i32, i32, i32, i32

  // -----

  // check source and destination with one externally defined affine map.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest_with_externally_defined_affine,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [
        { operand = 0, transforms = [#map2] },
        { operand = 1, transforms = [#map2] }
      ]
    } : memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i32, i32, i32, i32

  // check source and destination with multiple externally defined affine maps.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest_with_externally_defined_affine,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [
        { operand = 0, transforms = [#map2, #map3] },
        { operand = 1, transforms = [#map2, #map3] }
      ]
    } : memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i32, i32, i32, i32

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy_f16
//  CHECK: miopen.threadwise_copy

#map11 = affine_map<(d0, d1) -> (d1, d0, d1, d0)>

#map12 = affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>

func @miopen_threadwise_copy_v2_f16(%source_offset : i32, %source_coord : memref<2xi32, 5>, %dest_coord : memref<2xi32, 5>,
                                %source : vector<32xf16>, %dest : memref<?x?xf16>,
                                %dest_with_embedded_affine : memref<?x?xf16, #map11>,
                                %dest_with_externally_defined_affine : memref<?x?x?x?xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32

  %source_coord_y = memref.load %source_coord[%c0] : memref<2xi32, 5>
  %source_coord_x = memref.load %source_coord[%c1] : memref<2xi32, 5>
  %dest_coord_y = memref.load %dest_coord[%c0] : memref<2xi32, 5>
  %dest_coord_x = memref.load %dest_coord[%c1] : memref<2xi32, 5>

  // check dest as a vanilla memref.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset,
    %c0_i32,
    %dest_coord_x, %dest_coord_y)
    : vector<32xf16>, memref<?x?xf16>, i32, i32, i32, i32

  // -----

  // check source with one externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [ { operand = 0, transforms = [#map12] } ]
    } : vector<32xf16>, memref<?x?xf16>, i32, i32, i32, i32, i32

  // check source with multiple externally defined affine maps.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [ { operand = 0, transforms = [#map12, #map13] } ]
    } : vector<32xf16>, memref<?x?xf16>, i32, i32, i32, i32, i32

  // -----

  // check source and destination with one externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [
        { operand = 0, transforms = [#map12] },
        { operand = 1, transforms = [#map12] }
      ]
    } : vector<32xf16>, memref<?x?x?x?xf16>, i32, i32, i32, i32, i32

  // check source and destination with multiple externally defined affine maps.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset,
    %source_coord_x, %source_coord_y,
    %dest_coord_x, %dest_coord_y)
    {
      coord_transforms = [
        { operand = 0, transforms = [#map12, #map13] },
        { operand = 1, transforms = [#map12, #map13] }
      ]
    } : vector<32xf16>, memref<?x?x?x?xf16>, i32, i32, i32, i32, i32

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy_v2_f16
//  CHECK: miopen.threadwise_copy_v2

func @miopen_threadwise_gemm_f16(%lhs : memref<1x4x8xf16>, %rhs : memref<1x4x8xf16>, %output : memref<1x8x8xf16>) {
  miopen.threadwise_gemm(%lhs, %rhs, %output) : memref<1x4x8xf16>, memref<1x4x8xf16>, memref<1x8x8xf16>
  return
}

// CHECK-LABEL: func @miopen_threadwise_gemm_f16
//  CHECK: miopen.threadwise_gemm
