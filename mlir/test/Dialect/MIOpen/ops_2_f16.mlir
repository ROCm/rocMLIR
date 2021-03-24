// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Run: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

func @miopen_blockwise_gemm_f16(%A : memref<?x?xf16, 3>, %B : memref<?x?xf16, 3>, %C : memref<?x?xf16, 5>) {
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
  } : memref<?x?xf16, 3>, memref<?x?xf16, 3>, memref<?x?xf16, 5>, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_gemm_f16
//  CHECK: miopen.blockwise_gemm

func @miopen_blockwise_copy_f16(%source : memref<?x?xf16>, %dest : memref<?x?xf16, 3>, %source_coord : memref<2xi32>, %dest_coord : memref<2xi32>) {
  miopen.blockwise_copy(%source, %dest, %source_coord, %dest_coord) : memref<?x?xf16>, memref<?x?xf16, 3>, memref<2xi32>, memref<2xi32>
  miopen.blockwise_copy(%source, %dest, %source_coord, %dest_coord) { move_source_offset = 16 } : memref<?x?xf16>, memref<?x?xf16, 3>, memref<2xi32>, memref<2xi32>
  return
}

// CHECK-LABEL: func @miopen_blockwise_copy_f16
//  CHECK-NEXT: miopen.blockwise_copy
//  CHECK-NEXT: miopen.blockwise_copy

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
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %source_coord_y = load %source_coord[%c0] : memref<2xi32, 5>
  %source_coord_x = load %source_coord[%c0] : memref<2xi32, 5>
  %dest_coord_y = load %dest_coord[%c0] : memref<2xi32, 5>
  %dest_coord_x = load %dest_coord[%c0] : memref<2xi32, 5>

  // check source and dest as vanilla memrefs.
  miopen.threadwise_copy(%source, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf16, 5>, memref<?x?xf16, 5>

  // -----

  // check source with embedded affine maps.
  miopen.threadwise_copy(%source_with_embedded_affine, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf16, #map0, 3>, memref<?x?xf16, 5>

  // check dest with embedded affine maps.
  miopen.threadwise_copy(%source, %dest_with_embedded_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf16, 5>, memref<?x?xf16, #map1, 3>

  // check source and dest with embedded affine maps.
  miopen.threadwise_copy(%source_with_embedded_affine, %dest_with_embedded_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) : memref<?x?xf16, #map0, 3>, memref<?x?xf16, #map1, 3>

  // -----

  // check source with one externally defined affine map.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2] } ] } : memref<?x?x?x?xf16>, memref<?x?xf16, 5>

  // check source with multiple externally defined affine maps.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2, #map3] } ] } : memref<?x?x?x?xf16>, memref<?x?xf16, 5>

  // check destination with one externally defined affine map.
  miopen.threadwise_copy(%source, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 1, transforms = [#map2] } ] } : memref<?x?xf16, 5>, memref<?x?x?x?xf16>

  // check destination with multiple externally defined affine map.
  miopen.threadwise_copy(%source, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 1, transforms = [#map2, #map3] } ] } : memref<?x?xf16, 5>, memref<?x?x?x?xf16>

  // -----

  // check source and destination with one externally defined affine map.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2] }, { operand = 1, transforms = [#map2] } ] } : memref<?x?x?x?xf16>, memref<?x?x?x?xf16>

  // check source and destination with multiple externally defined affine maps.
  miopen.threadwise_copy(%source_with_externally_defined_affine, %dest_with_externally_defined_affine, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map2, #map3] }, { operand = 1, transforms = [#map2, #map3] } ] } : memref<?x?x?x?xf16>, memref<?x?x?x?xf16>

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
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c0_i32 = constant 0 : i32

  %source_coord_y = load %source_coord[%c0] : memref<2xi32, 5>
  %source_coord_x = load %source_coord[%c1] : memref<2xi32, 5>
  %dest_coord_y = load %dest_coord[%c0] : memref<2xi32, 5>
  %dest_coord_x = load %dest_coord[%c1] : memref<2xi32, 5>

  // check dest as a vanilla memref.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset, %c0_i32, %dest_coord_x, %dest_coord_y) : vector<32xf16>, memref<?x?xf16>

  // -----

  // check source with one externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12] } ] } : vector<32xf16>, memref<?x?xf16>

  // check source with multiple externally defined affine maps.
  miopen.threadwise_copy_v2(%source, %dest, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12, #map13] } ] } : vector<32xf16>, memref<?x?xf16>

  // -----

  // check source and destination with one externally defined affine map.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12] }, { operand = 1, transforms = [#map12] } ] } : vector<32xf16>, memref<?x?x?x?xf16>

  // check source and destination with multiple externally defined affine maps.
  miopen.threadwise_copy_v2(%source, %dest_with_externally_defined_affine, %source_offset, %source_coord_x, %source_coord_y, %dest_coord_x, %dest_coord_y) { coord_transforms = [ { operand = 0, transforms = [#map12, #map13] }, { operand = 1, transforms = [#map12, #map13] } ] } : vector<32xf16>, memref<?x?x?x?xf16>

  return
}
 
// CHECK-LABEL: func @miopen_threadwise_copy_v2_f16
//  CHECK: miopen.threadwise_copy_v2

func @miopen_threadwise_gemm_f16(%lhs : memref<4x8xf16>, %rhs : memref<4x8xf16>, %output : memref<8x8xf16>) {
  miopen.threadwise_gemm(%lhs, %rhs, %output) : memref<4x8xf16>, memref<4x8xf16>, memref<8x8xf16>
  return
}
 
// CHECK-LABEL: func @miopen_threadwise_gemm_f16
//  CHECK: miopen.threadwise_gemm

// ----

func @miopen_xdlops_gemm_v2_one_result_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                       %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> vector<32xf16> {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>
  return %vectorD0 : vector<32xf16>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_one_result_f16
//  CHECK: miopen.xdlops_gemm_v2
 
// ----

func @miopen_xdlops_gemm_v2_two_results_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                        %bufferA : memref<32xf16, 5>, %bufferB: memref<16xf16, 5>) -> (vector<32xf16>, vector<32xf16>) {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorC1 = splat %c0f : vector<32xf16>
  %vectorD0, %vectorD1 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>, vector<32xf16>
  return %vectorD0, %vectorD1 : vector<32xf16>, vector<32xf16>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_two_results_f16
//  CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_one_result_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                          %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> vector<32xf16> {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorD0 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>
  return %vectorD0 : vector<32xf16>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_one_result_f16
//  CHECK: miopen.blockwise_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_two_results_f16(%matrixA : memref<12288xf16, 3>, %matrixB : memref<12288xf16, 3>,
                                           %bufferA : memref<32xf16, 5>, %bufferB : memref<16xf16, 5>) -> (vector<32xf16>, vector<32xf16>) {
  %c0 = constant 0 : index
  %c0f = constant 0.0 : f16
  %vectorC0 = splat %c0f : vector<32xf16>
  %vectorC1 = splat %c0f : vector<32xf16>
  %vectorD0, %vectorD1 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf16, 3>, memref<12288xf16, 3>, index, index, memref<32xf16, 5>, memref<16xf16, 5>, vector<32xf16>, vector<32xf16>
  return %vectorD0, %vectorD1 : vector<32xf16>, vector<32xf16>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_two_results_f16
//  CHECK: miopen.blockwise_gemm_v2
