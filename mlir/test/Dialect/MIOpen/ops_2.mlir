// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @miopen_alloc() {
  // allocation on global.
  %buffer_global = miopen.alloc() : memref<1024xi8>

  // allocation on LDS.
  %buffer_lds = miopen.alloc() : memref<1024xi8, 3>

  // allocation on register (VGPR).
  %buffer_register = miopen.alloc() : memref<1024xi8, 5>

  return
}

// CHECK-LABEL: func @miopen_alloc
//   CHECK: miopen.alloc
//   CHECK-NEXT: miopen.alloc
//   CHECK-NEXT: miopen.alloc

func @miopen_subview(%buffer : memref<1024xi8>) {
  %c0 = constant 0 : index
  %c512 = constant 512 : index

  // 0 offset, same type.
  %view_0 = miopen.subview(%buffer, %c0) : memref<1024xi8> to memref<1024xi8>

  // 0 offset, different type.
  %view_1 = miopen.subview(%buffer, %c0) : memref<1024xi8> to memref<256xf32>

  // 0 offset, different type.
  %view_2 = miopen.subview(%buffer, %c0) : memref<1024xi8> to memref<256xf16>

  // 0 offset, different type, different rank.
  %view_3 = miopen.subview(%buffer, %c0) { dimensions = [ 16, 16 ] } : memref<1024xi8> to memref<16x16xf32>

  // 0 offset, different type, different rank.
  %view_4 = miopen.subview(%buffer, %c0) { dimensions = [ 16, 16 ] } : memref<1024xi8> to memref<16x16xf16>

  // 512 offset, same type.
  %view_5 = miopen.subview(%buffer, %c512) : memref<1024xi8> to memref<512xi8>

  // 512 offset, different type.
  %view_6 = miopen.subview(%buffer, %c512) : memref<1024xi8> to memref<128xf32>

  // 512 offset, different type.
  %view_7 = miopen.subview(%buffer, %c512) : memref<1024xi8> to memref<128xf16>

  // 512 offset, different type, different rank.
  %view_8 = miopen.subview(%buffer, %c512) { dimensions = [ 16, 8 ] } : memref<1024xi8> to memref<16x8xf32>

  // 512 offset, different type, different rank.
  %view_9 = miopen.subview(%buffer, %c512) { dimensions = [ 16, 8 ] } : memref<1024xi8> to memref<16x8xf16>

  return
}

// CHECK-LABEL: func @miopen_subview
//   CHECK: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview
//   CHECK-NEXT: miopen.subview


func @miopen_fill(%buffer_f32 : memref<1024xf32, 5>, %buffer_i32 : memref<2xi32, 5>, %buffer_f16 : memref<1024xf16, 5>) {
  %cst = constant 0.0 : f32
  miopen.fill(%buffer_f32, %cst) : memref<1024xf32, 5>

  %cst_f16 = constant 0.0 : f16
  miopen.fill(%buffer_f16, %cst_f16) : memref<1024xf16, 5>

  %c0 = constant 0 : i32
  miopen.fill(%buffer_i32, %c0) : memref<2xi32, 5>
  return
}

// CHECK-LABEL: func @miopen_fill
//   CHECK: miopen.fill
//   CHECK: miopen.fill
//   CHECK: miopen.fill

func @miopen_move_pos_v2(%vector_f32 : vector<2xf32>, %vector_i32 : vector<2xi32>) {
  %deltaY_i32 = constant 16 : i32
  %deltaX_i32 = constant 8 : i32
  %output1 = miopen.move_pos_v2(%vector_i32, %deltaY_i32, %deltaX_i32) : vector<2xi32>
 
  %deltaY_f32 = constant 16.0 : f32
  %deltaX_f32 = constant 8.0 : f32
  %output2 = miopen.move_pos_v2(%vector_f32, %deltaY_f32, %deltaX_f32) : vector<2xf32>

  return
}

// CHECK-LABEL: func @miopen_move_pos_v2
//   CHECK: %{{.*}} = miopen.move_pos_v2(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xi32>
//   CHECK: %{{.*}} = miopen.move_pos_v2(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>

func @miopen_workgroup_barrier() {
  miopen.workgroup_barrier
  return
}

// CHECK-LABEL: func @miopen_workgroup_barrier
//   CHECK-NEXT: miopen.workgroup_barrier

func @miopen_lds_barrier() {
  miopen.lds_barrier
  return
}

// CHECK-LABEL: func @miopen_lds_barrier
//   CHECK-NEXT: miopen.lds_barrier

func @miopen_indexing() {
  %0 = miopen.workgroup_id : index
  %1 = miopen.workitem_id : index
  return
}

// CHECK-LABEL: func @miopen_indexing
//   CHECK-NEXT: miopen.workgroup_id
//   CHECK-NEXT: miopen.workitem_id
 
func @miopen_blockwise_gemm(%A : memref<?x?x?xf32, 3>, %B : memref<?x?x?xf32, 3>, %C : memref<?x?x?xf32, 5>) {
  %c0 = constant 0 : index
  miopen.blockwise_gemm(%A, %B, %C, %c0, %c0) {
    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?x?xf32, 3>, memref<?x?x?xf32, 3>, memref<?x?x?xf32, 5>, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_gemm
//  CHECK: miopen.blockwise_gemm

func @miopen_blockwise_copy(%source : memref<?x?xf32>, %dest : memref<?x?xf32, 3>, %source_coord : vector<3xi32>, %dest_coord : vector<3xi32>) {
  miopen.blockwise_copy(%source, %dest, %source_coord, %dest_coord) : memref<?x?xf32>, memref<?x?xf32, 3>, vector<3xi32>, vector<3xi32>
  return
}

// CHECK-LABEL: func @miopen_blockwise_copy
//  CHECK: miopen.blockwise_copy(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<?x?xf32>, memref<?x?xf32, 3>, vector<3xi32>, vector<3xi32>

// --------------------------
// blockwise_load tests.

// f32 tests.

func @miopen_blockwise_load_f32(%source : memref<?x?xf32>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> f32  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xf32>, f32
  return %result : f32
}

// CHECK-LABEL: func @miopen_blockwise_load_f32
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xf32>, f32

func @miopen_blockwise_load_2xf32(%source : memref<?x?xf32>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<2xf32>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xf32>, vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @miopen_blockwise_load_2xf32
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xf32>, vector<2xf32>

func @miopen_blockwise_load_4xf32(%source : memref<?x?xf32>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<4xf32>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xf32>, vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func @miopen_blockwise_load_4xf32
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xf32>, vector<4xf32>

// f16 tests.

func @miopen_blockwise_load_f16(%source : memref<?x?xf16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> f16  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xf16>, f16
  return %result : f16
}

// CHECK-LABEL: func @miopen_blockwise_load_f16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xf16>, f16

func @miopen_blockwise_load_2xf16(%source : memref<?x?xf16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<2xf16>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xf16>, vector<2xf16>
  return %result : vector<2xf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_2xf16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xf16>, vector<2xf16>

func @miopen_blockwise_load_4xf16(%source : memref<?x?xf16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<4xf16>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xf16>, vector<4xf16>
  return %result : vector<4xf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_4xf16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xf16>, vector<4xf16>

func @miopen_blockwise_load_8xf16(%source : memref<?x?xf16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<8xf16>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xf16>, vector<8xf16>
  return %result : vector<8xf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_8xf16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xf16>, vector<8xf16>

// i16 tests.

func @miopen_blockwise_load_i16(%source : memref<?x?xi16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> i16  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xi16>, i16
  return %result : i16
}

// CHECK-LABEL: func @miopen_blockwise_load_i16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xi16>, i16

func @miopen_blockwise_load_2xi16(%source : memref<?x?xi16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<2xi16>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xi16>, vector<2xi16>
  return %result : vector<2xi16>
}

// CHECK-LABEL: func @miopen_blockwise_load_2xi16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xi16>, vector<2xi16>

func @miopen_blockwise_load_4xi16(%source : memref<?x?xi16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<4xi16>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xi16>, vector<4xi16>
  return %result : vector<4xi16>
}

// CHECK-LABEL: func @miopen_blockwise_load_4xi16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xi16>, vector<4xi16>

func @miopen_blockwise_load_8xi16(%source : memref<?x?xi16>, %source_coord : vector<2xi32>, %dest_coord : memref<2xi32>) -> vector<8xi16>  {
  %result = miopen.blockwise_load(%source, %source_coord) : memref<?x?xi16>, vector<8xi16>
  return %result : vector<8xi16>
}

// CHECK-LABEL: func @miopen_blockwise_load_8xi16
//  CHECK: %{{.*}} = miopen.blockwise_load(%{{.*}}, %{{.*}}) : memref<?x?xi16>, vector<8xi16>

// --------------------------
// blockwise_store tests.

// f32 tests.

func @miopen_blockwise_store_f32(%data : f32, %dest : memref<?x?xf32, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : f32, memref<?x?xf32, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_f32
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : f32, memref<?x?xf32, 3>

func @miopen_blockwise_store_2xf32(%data : vector<2xf32>, %dest : memref<?x?xf32, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<2xf32>, memref<?x?xf32, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_2xf32
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>, memref<?x?xf32, 3>

func @miopen_blockwise_store_4xf32(%data : vector<4xf32>, %dest : memref<?x?xf32, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<4xf32>, memref<?x?xf32, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_4xf32
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>, memref<?x?xf32, 3>

// f16 tests.

func @miopen_blockwise_store_f16(%data : f16, %dest : memref<?x?xf16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : f16, memref<?x?xf16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_f16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : f16, memref<?x?xf16, 3>

func @miopen_blockwise_store_2xf16(%data : vector<2xf16>, %dest : memref<?x?xf16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<2xf16>, memref<?x?xf16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_2xf16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf16>, memref<?x?xf16, 3>

func @miopen_blockwise_store_4xf16(%data : vector<4xf16>, %dest : memref<?x?xf16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<4xf16>, memref<?x?xf16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_4xf16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf16>, memref<?x?xf16, 3>

func @miopen_blockwise_store_8xf16(%data : vector<8xf16>, %dest : memref<?x?xf16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<8xf16>, memref<?x?xf16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_8xf16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<8xf16>, memref<?x?xf16, 3>

// i16 tests.

func @miopen_blockwise_store_i16(%data : i16, %dest : memref<?x?xi16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : i16, memref<?x?xi16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_i16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : i16, memref<?x?xi16, 3>

func @miopen_blockwise_store_2xi16(%data : vector<2xi16>, %dest : memref<?x?xi16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<2xi16>, memref<?x?xi16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_2xi16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xi16>, memref<?x?xi16, 3>

func @miopen_blockwise_store_4xi16(%data : vector<4xi16>, %dest : memref<?x?xi16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<4xi16>, memref<?x?xi16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_4xi16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xi16>, memref<?x?xi16, 3>

func @miopen_blockwise_store_8xi16(%data : vector<8xi16>, %dest : memref<?x?xi16, 3>, %dest_coord : vector<2xi32>) {
  miopen.blockwise_store(%data, %dest, %dest_coord) : vector<8xi16>, memref<?x?xi16, 3>
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_8xi16
//  CHECK: miopen.blockwise_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<8xi16>, memref<?x?xi16, 3>

// --------------------------
// threadwise_copy tests.

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

#map11 = affine_map<(d0, d1) -> (d1, d0, d1, d0)>

#map12 = affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>

func @miopen_threadwise_copy_v2(%source_offset : i32, %source_coord : memref<2xi32, 5>, %dest_coord : memref<2xi32, 5>,
                                %source : vector<32xf32>, %dest : memref<?x?xf32>,
                                %dest_with_embedded_affine : memref<?x?xf32, #map11>,
                                %dest_with_externally_defined_affine : memref<?x?x?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c0_i32 = constant 0 : i32

  %source_coord_y = load %source_coord[%c0] : memref<2xi32, 5>
  %source_coord_x = load %source_coord[%c1] : memref<2xi32, 5>
  %dest_coord_y = load %dest_coord[%c0] : memref<2xi32, 5>
  %dest_coord_x = load %dest_coord[%c1] : memref<2xi32, 5>

  // check dest as a vanilla memref.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset, %c0_i32, %dest_coord_x, %dest_coord_y) : vector<32xf32>, memref<?x?xf32>

  // -----

  // check source with one externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12] } ] } : vector<32xf32>, memref<?x?xf32>

  // check source with multiple externally defined affine maps.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12, #map13] } ] } : vector<32xf32>, memref<?x?xf32>

  // -----

  // check source and destination with one externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12] }, { operand = 1, transforms = [#map12] } ] } : vector<32xf32>, memref<?x?x?x?xf32>

  // check source and destination with multiple externally defined affine maps.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12, #map13] }, { operand = 1, transforms = [#map12, #map13] } ] } : vector<32xf32>, memref<?x?x?x?xf32>

  return
}
 
// CHECK-LABEL: func @miopen_threadwise_copy_v2
//  CHECK: miopen.threadwise_copy_v2

func @miopen_threadwise_gemm(%lhs : memref<1x4x8xf32>, %rhs : memref<1x4x8xf32>, %output : memref<1x8x8xf32>) {
  miopen.threadwise_gemm(%lhs, %rhs, %output) : memref<1x4x8xf32>, memref<1x4x8xf32>, memref<1x8x8xf32>
  return
}
 
// CHECK-LABEL: func @miopen_threadwise_gemm
//  CHECK: miopen.threadwise_gemm

// ----

func @miopen_mfma_v2_f32(%a : f32, %b : f32, %c : vector<32xf32>) -> vector<32xf32> {
  %d = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x1f32", imm = [1, 0, 0] } : f32, vector<32xf32>
  return %d : vector<32xf32>
}

// CHECK-LABEL: func @miopen_mfma_v2_f32
//   CHECK: miopen.mfma_v2

func @miopen_mfma_v2_f16(%a : vector<4xf16>, %b : vector<4xf16>, %c : vector<32xf32>) -> vector<32xf32> {
  %d = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x4f16", imm = [1, 0, 0] } : vector<4xf16>, vector<32xf32>
  return %d : vector<32xf32>
}

// CHECK-LABEL: func @miopen_mfma_v2_f16
//   CHECK: miopen.mfma_v2

func @miopen_mfma_v2_bf16(%a : vector<2xi16>, %b : vector<2xi16>, %c : vector<32xf32>) -> vector<32xf32> {
  %d = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x2bf16", imm = [1, 0, 0] } : vector<2xi16>, vector<32xf32>
  return %d : vector<32xf32>
}

// CHECK-LABEL: func @miopen_mfma_v2_bf16
//   CHECK: miopen.mfma_v2

// ----

func @miopen_xdlops_gemm_v2_one_result(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                       %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>) -> vector<32xf32> {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f32
  %vectorC0 = splat %c0f : vector<32xf32>
  %vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>
  return %vectorD0 : vector<32xf32>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_one_result
//  CHECK: miopen.xdlops_gemm_v2
 
// ----

func @miopen_xdlops_gemm_v2_two_results(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                        %bufferA : memref<32xf32, 5>, %bufferB: memref<16xf32, 5>) -> (vector<32xf32>, vector<32xf32>) {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f32
  %vectorC0 = splat %c0f : vector<32xf32>
  %vectorC1 = splat %c0f : vector<32xf32>
  %vectorD0, %vectorD1 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_two_results
//  CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_one_result(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                          %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>) -> vector<32xf32> {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f32
  %vectorC0 = splat %c0f : vector<32xf32>
  %vectorD0 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>
  return %vectorD0 : vector<32xf32>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_one_result
//  CHECK: miopen.blockwise_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_two_results(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                           %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>) -> (vector<32xf32>, vector<32xf32>) {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f32
  %vectorC0 = splat %c0f : vector<32xf32>
  %vectorC1 = splat %c0f : vector<32xf32>
  %vectorD0, %vectorD1 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_two_results
//  CHECK: miopen.blockwise_gemm_v2
