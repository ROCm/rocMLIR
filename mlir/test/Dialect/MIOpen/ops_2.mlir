// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @miopen_gridwise_gemm_ex(%A : memref<?x?xf32>, %B : memref<?x?xf32>, %C : memref<?x?xf32>) {
  miopen.gridwise_gemm_ex(%A, %B, %C) {
    filter_layout = ["k", "c", "y", "x"],
    filter_dimension = [1, 2, 3, 4],
    input_layout = ["n", "c", "hi", "wi"],
    input_dimension = [5, 6, 7, 8],
    output_layout = ["n", "k", "ho", "wo"],
    output_dimension = [9, 10, 11, 12],
    strides = [1, 1],
    dilations = [1, 1],
    padding = [0, 0]
  } : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  return
}

// CHECK-LABEL: func @miopen_gridwise_gemm_ex
//  CHECK-NEXT: miopen.gridwise_gemm_ex

func @miopen_gpu_alloc() {
  // allocation on global.
  %buffer_global = miopen.gpu_alloc() : memref<1024xi8>

  // allocation on LDS.
  %buffer_lds = miopen.gpu_alloc() : memref<1024xi8, 3>

  // allocation on register (VGPR).
  %buffer_register = miopen.gpu_alloc() : memref<1024xi8, 5>

  return
}

// CHECK-LABEL: func @miopen_gpu_alloc
//   CHECK: miopen.gpu_alloc
//   CHECK-NEXT: miopen.gpu_alloc
//   CHECK-NEXT: miopen.gpu_alloc

func @miopen_subview(%buffer : memref<1024xi8>) {
  %c0 = constant 0 : index
  %c512 = constant 512 : index

  // 0 offset, same type.
  %view_0 = miopen.subview(%buffer, %c0) : memref<1024xi8> to memref<1024xi8>

  // 0 offset, different type.
  %view_1 = miopen.subview(%buffer, %c0) : memref<1024xi8> to memref<256xf32>

  // 0 offset, different type, different rank.
  %view_2 = miopen.subview(%buffer, %c0) { dimensions = [ 16, 16 ] } : memref<1024xi8> to memref<16x16xf32>

  // 512 offset, same type.
  %view_3 = miopen.subview(%buffer, %c512) : memref<1024xi8> to memref<512xi8>

  // 512 offset, different type.
  %view_4 = miopen.subview(%buffer, %c512) : memref<1024xi8> to memref<128xf32>

  // 512 offset, different type, different rank.
  %view_5 = miopen.subview(%buffer, %c512) { dimensions = [ 16, 8 ] } : memref<1024xi8> to memref<16x8xf32>

  return
}

// CHECK-LABEL: func @miopen_subview
//   CHECK: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview


func @miopen_fill(%buffer_f32 : memref<1024xf32, 5>, %buffer_i32 : memref<2xi32, 5>) {
  %cst = constant 0.0 : f32
  miopen.fill(%buffer_f32, %cst) : memref<1024xf32, 5>

  %c0 = constant 0 : i32
  miopen.fill(%buffer_i32, %c0) : memref<2xi32, 5>
  return
}

// CHECK-LABEL: func @miopen_fill
//   CHECK: miopen.fill

func @miopen_move_pos(%buffer_f32 : memref<2xf32, 5>, %buffer_i32 : memref<2xi32, 5>) {
  %deltaY_i32 = constant 16 : i32
  %deltaX_i32 = constant 8 : i32
  miopen.move_pos(%buffer_i32, %deltaY_i32, %deltaX_i32) : memref<2xi32, 5>

  %deltaY_f32 = constant 16.0 : f32
  %deltaX_f32 = constant 8.0 : f32
  miopen.move_pos(%buffer_f32, %deltaY_f32, %deltaX_f32) : memref<2xf32, 5>

  return
}

// CHECK-LABEL: func @miopen_move_pos
//   CHECK: miopen.move_pos
//   CHECK: miopen.move_pos

func @miopen_lds_barrier() {
  miopen.lds_barrier
  return
}

// CHECK-LABEL: func @miopen_lds_barrier
//   CHECK-NEXT: miopen.lds_barrier
 
func @miopen_blockwise_gemm(%A : memref<?x?xf32, 3>, %B : memref<?x?xf32, 3>, %C : memref<?x?xf32, 5>) {
  miopen.blockwise_gemm(%A, %B, %C) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?xf32, 3>, memref<?x?xf32, 3>, memref<?x?xf32, 5>
  return
}

// CHECK-LABEL: func @miopen_blockwise_gemm
//  CHECK-NEXT: miopen.blockwise_gemm

func @miopen_blockwise_copy(%source : memref<?x?xf32>, %dest : memref<?x?xf32, 3>, %source_coord : memref<2xi32>, %dest_coord : memref<2xi32>) {
  miopen.blockwise_copy(%source, %dest, %source_coord, %dest_coord) : memref<?x?xf32>, memref<?x?xf32, 3>, memref<2xi32>, memref<2xi32>
  miopen.blockwise_copy(%source, %dest, %source_coord, %dest_coord) { move_source_offset = 16 } : memref<?x?xf32>, memref<?x?xf32, 3>, memref<2xi32>, memref<2xi32>
  return
}

// CHECK-LABEL: func @miopen_blockwise_copy
//  CHECK-NEXT: miopen.blockwise_copy
//  CHECK-NEXT: miopen.blockwise_copy

#map0 = affine_map<(d0, d1) -> (d0, d1, d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0, d1, d0)>

#map2 = affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>

func @miopen_threadwise_copy(%source_coord : memref<2xi32, 5>, %dest_coord : memref<2xi32, 5>,
                             %source : memref<?x?xf32, 5>, %dest : memref<?x?xf32, 5>,
                             %source_with_embedded_affine : memref<?x?xf32, #map0, 3>,
                             %dest_with_embedded_affine : memref<?x?xf32, #map1, 3>,
                             %source_with_externally_defined_affine : memref<?x?x?x?xf32>,
                             %dest_with_externally_defined_affine : memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %source_coord_y = load %source_coord[%c0] : memref<2xi32, 5>
  %source_coord_x = load %source_coord[%c0] : memref<2xi32, 5>
  %dest_coord_y = load %dest_coord[%c0] : memref<2xi32, 5>
  %dest_coord_x = load %dest_coord[%c0] : memref<2xi32, 5>

  // check source and dest as vanilla memrefs.
  miopen.threadwise_copy(%source, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf32, 5>, memref<?x?xf32, 5>

  // -----

  // check source with embedded affine maps.
  miopen.threadwise_copy(%source_with_embedded_affine, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf32, #map0, 3>, memref<?x?xf32, 5>

  // check dest with embedded affine maps.
  miopen.threadwise_copy(%source, %dest_with_embedded_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf32, 5>, memref<?x?xf32, #map1, 3>

  // check source and dest with embedded affine maps.
  miopen.threadwise_copy(%source_with_embedded_affine, %dest_with_embedded_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf32, #map0, 3>, memref<?x?xf32, #map1, 3>

  // -----

  // check source with one externally defined affine map.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2] } ] } : memref<?x?x?x?xf32>, memref<?x?xf32, 5>

  // check source with multiple externally defined affine maps.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2, #map3] } ] } : memref<?x?x?x?xf32>, memref<?x?xf32, 5>

  // check destination with one externally defined affine map.
  miopen.threadwise_copy(%source, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 1, transforms = [#map2] } ] } : memref<?x?xf32, 5>, memref<?x?x?x?xf32>

  // check destination with multiple externally defined affine map.
  miopen.threadwise_copy(%source, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 1, transforms = [#map2, #map3] } ] } : memref<?x?xf32, 5>, memref<?x?x?x?xf32>

  // -----

  // check source and destination with one externally defined affine map.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2] }, { operand = 1, transforms = [#map2] } ] } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>

  // check source and destination with multiple externally defined affine maps.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2, #map3] }, { operand = 1, transforms = [#map2, #map3] } ] } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy
//  CHECK: miopen.threadwise_copy
