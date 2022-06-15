// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

#gemm_padding0 = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0>

#transform_map0 = #miopen.transform_map<
  affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)> by
  [#miopen.transform<PassThrough ["b"] at [1] -> ["w"] at [0]>,
   #miopen.transform<Merge{3, 3, 3} ["a"] at [0] -> ["x", "y", "z"] at [1, 2, 3]>
  ] bounds = [1, 27] -> [1, 3, 3, 3]>

#transform_map1 = #miopen.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)> by [
    #miopen.transform<PassThrough ["w", "x", "y", "z"] at [0, 1, 2, 3]
      -> ["x", "w", "y", "z"] at [1, 0, 2, 3]>
    ] bounds = [1, 3, 3, 3] -> [3, 1, 3, 3]>

func @miopen_threadwise_copy_f16(%source_coord : memref<2xindex, 5>, %dest_coord : memref<2xindex, 5>,
                             %source : memref<?x?xf16, 5>, %dest : memref<?x?xf16, 5>,
                             %source_with_transform_maps : memref<?x?x?x?xf16>,
                             %dest_with_transform_maps : memref<?x?x?x?xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_coord_y = memref.load %source_coord[%c0] : memref<2xindex, 5>
  %source_coord_x = memref.load %source_coord[%c0] : memref<2xindex, 5>
  %dest_coord_y = memref.load %dest_coord[%c0] : memref<2xindex, 5>
  %dest_coord_x = memref.load %dest_coord[%c0] : memref<2xindex, 5>

  // check source and dest as vanilla memrefs.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y]
    with [[], []]
    {paddingInfo = #gemm_padding0, oobDims=[false, false], bounds = [1 : index, 1 : index]}
    : memref<?x?xf16, 5>, index, index -> memref<?x?xf16, 5>, index, index

  // -----

  // check source with one coordinate transform.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0], []]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf16>, index, index -> memref<?x?xf16, 5>, index, index

  // check source with multiple coordinate transforms.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0, #transform_map1], []]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf16>, index, index -> memref<?x?xf16, 5>, index, index

  // check destination with one coordinate transform.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[], [#transform_map0]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      bounds = [1 : index, 1 : index] }
    : memref<?x?xf16, 5>, index, index -> memref<?x?x?x?xf16>, index, index

  // check destination with multiple coordinate transform.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[], [#transform_map0, #transform_map1]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      bounds = [1 : index, 1 : index] }
    : memref<?x?xf16, 5>, index, index -> memref<?x?x?x?xf16>, index, index

  // -----

  // check source and destination with one coordinate transform.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0], [#transform_map0]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf16>, index, index -> memref<?x?x?x?xf16>, index, index

  // check source and destination with multiple coordinate transforms.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0, #transform_map1], [#transform_map0, #transform_map1]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf16>, index, index -> memref<?x?x?x?xf16>, index, index

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy_f16
//  CHECK: miopen.threadwise_copy

func @miopen_threadwise_gemm_f16(%lhs : memref<32xf16, 5>, %rhs : memref<32xf16, 5>, %output : memref<256xf16, 5>) {
  miopen.threadwise_gemm %output = %lhs * %rhs
    { g = 1 : index, k = 4 : index, m = 8 : index, n = 8 : index, kPack = 1 : index }
    : memref<256xf16, 5> = memref<32xf16, 5> * memref<32xf16, 5>
  return
}

// CHECK-LABEL: func @miopen_threadwise_gemm_f16
//  CHECK: miopen.threadwise_gemm
