// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s


func.func @miopen_conv2d(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @miopen_conv2d
// CHECK-NEXT: miopen.conv2d

func.func @miopen_conv2d_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) {
  miopen.conv2d(%filter, %input, %output) {
    filter_layout = ["g" ,"k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}
// CHECK-LABEL: func.func @miopen_conv2d_f16
// CHECK-NEXT: miopen.conv2d

func.func @miopen_conv2d_bwd_data(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @miopen_conv2d_bwd_data
// CHECK-NEXT: miopen.conv2d_bwd_data

func.func @miopen_conv2d_bwd_data_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}
// CHECK-LABEL: func.func @miopen_conv2d_bwd_data_f16
// CHECK-NEXT: miopen.conv2d_bwd_data

func.func @miopen_conv2d_bwd_weight(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @miopen_conv2d_bwd_weight
// CHECK-NEXT: miopen.conv2d_bwd_weight

func.func @miopen_conv2d_bwd_weight_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0, 0]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}

// CHECK-LABEL: func.func @miopen_conv2d_bwd_weight_f16
// CHECK-NEXT: miopen.conv2d_bwd_weight

// Affine maps needed when testing transform
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d2, d3 - 1, d4 - 2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 512,
  (d1 mod 512) floordiv 16, d1 mod 16)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) ->
  (d1, d0, d2, d3 + d4, d5 + d6)>

// test 1-1 dimension mappings.
func.func @miopen_transform_1_to_1(%memref: memref<1x2x3x4x5xf32, 3>) {
  %transformed_memref = miopen.transform %memref by [
    #miopen.transform_map<#map0 by[
      #miopen.transform<PassThrough ["g"] at [0] -> ["g"] at [1]>,
      #miopen.transform<PassThrough ["n"] at [1] -> ["n"] at [0]>,
      #miopen.transform<PassThrough ["c"] at [2] -> ["c"] at [2]>,
      #miopen.transform<Pad{1, 1} ["hipad"] at [3] -> ["hi"] at [3]>,
      #miopen.transform<Pad{2, 2} ["wipad"] at [4] -> ["wi"] at [4]>
    ] bounds = [2, 1, 3, 6, 9] -> [1, 2, 3, 4, 5]>
  ] : memref<1x2x3x4x5xf32, 3> to memref<2x1x3x6x9xf32, #map0, 3>
  return
}
// CHECK-LABEL: func.func @miopen_transform_1_to_1
//  CHECK-NEXT: miopen.transform

// test multiple source dimensions map to 1 target dimension.
func.func @miopen_transform_n_to_1(%memref : memref<1x128x64x32x16xf32>) {
  %transformed_memref = miopen.transform %memref by [
    #miopen.transform_map<#map1 by[
      #miopen.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>,
      #miopen.transform<Merge{64, 32, 16} ["gemmK"] at [1] -> ["c", "y", "x"] at [2, 3, 4]>,
      #miopen.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>
    ] bounds = [1, 32768, 128] -> [1, 128, 64, 32, 16]>
  ] : memref<1x128x64x32x16xf32> to memref<1x32768x128xf32, #map1>
  return
}
// CHECK-LABEL: func.func @miopen_transform_n_to_1
//  CHECK-NEXT: miopen.transform

// test 1 source dimension map to multiple target dimensions.
func.func @miopen_transform_1_to_n(%memref : memref<?x?x?x?x?xf32>) {
  %transformed_memref = miopen.transform %memref by [
    #miopen.transform_map<#map2 by [
      #miopen.transform<PassThrough ["n", "g", "c"] at [0, 1, 2] ->
        ["n", "g", "c"] at [1, 0, 2]>,
      #miopen.transform<Embed{1, 1} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>,
      #miopen.transform<Embed{1, 1} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>
      // Note: fake data should work fine for now
     ] bounds = [0, 0, 0, 0, 0, 0, 0] -> [0, 0, 0, 0, 0]>
  ] : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32, #map2>
  return
}

// CHECK-LABEL: func.func @miopen_transform_1_to_n
//  CHECK-NEXT: miopen.transform

func.func @miopen_gridwise_gemm(%A : memref<?x?x?xf32>, %B : memref<?x?x?xf32>, %C : memref<?x?x?xf32>) {
  miopen.gridwise_gemm(%A, %B, %C) {
    paddingInfo =
      #miopen.padding_info<extraK = 0, extraM = 0, extraN = 0>,
    transforms = [[], [], []]
  } : memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>
  return
}

// CHECK-LABEL: func.func @miopen_gridwise_gemm
//  CHECK-NEXT: miopen.gridwise_gemm

func.func @miopen_gridwise_gemm_v2(%A : memref<?x?x?xf32>, %B : memref<?x?x?xf32>, %C : memref<?x?x?xf32>) {
  miopen.gridwise_gemm_v2(%A, %B, %C) {
    paddingInfo =
      #miopen.padding_info<extraK = 0, extraM = 0, extraN = 0>,
    transforms = [[], [], []],
    storeMethod = 0 : i32
  } : memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>
  return
}

// CHECK-LABEL: func.func @miopen_gridwise_gemm_v2
// CHECK-NEXT: miopen.gridwise_gemm_v2

func.func @miopen_extract_slice(%v : vector<32xf32>) -> vector<4xf32> {
  %i = arith.constant 0 : index
  %r = miopen.extract_slice %v[%i] : vector<32xf32> -> vector<4xf32>
  return %r : vector<4xf32>
}
// CHECK-LABEL: func.func @miopen_extract_slice
// CHECK: miopen.extract_slice

func.func @miopen_insert_slice(%u: vector<4xf32>, %v: vector<32xf32>) -> vector<32xf32> {
  %i = arith.constant 0 : index
  %w = miopen.insert_slice %u -> %v[%i] : vector<4xf32> -> vector<32xf32>
  return %w : vector<32xf32>
}
// CHECK-LABEL: func.func @miopen_insert_slice
// CHECK: miopen.insert_slice

func.func @miopen_buffer_load(%buffer: memref<128x128xf32>, %idx0: index, %idx1: index) -> vector<4xf32> {
  %ret = miopen.buffer_load %buffer[%idx0, %idx1] { leftOobDims = [], rightOobDims = [1 : i32] }
    : memref<128x128xf32>, index, index -> vector<4xf32>
  return %ret : vector<4xf32>
}
// CHECK-LABEL: func.func @miopen_buffer_load
// CHECK-NEXT: miopen.buffer_load

func.func @miopen_buffer_store(%buffer: memref<128x128xf32>, %data: vector<4xf32>, %idx0: index, %idx1: index) {
  miopen.buffer_store %data -> %buffer[%idx0, %idx1] { leftOobDims = [], rightOobDims = [1 : i32] }
  : vector<4xf32> -> memref<128x128xf32>, index, index
  return
}
// CHECK-LABEL: func.func @miopen_buffer_store
// CHECK-NEXT: miopen.buffer_store

func.func @miopen_in_bounds_load(%buffer: memref<128x128xf32, 3>, %idx0: index, %idx1: index) -> vector<4xf32> {
  %ret = miopen.in_bounds_load %buffer[%idx0, %idx1]
    : memref<128x128xf32, 3>, index, index -> vector<4xf32>
  return %ret : vector<4xf32>
}
// CHECK-LABEL: func.func @miopen_in_bounds_load
// CHECK-NEXT: miopen.in_bounds_load

func.func @miopen_in_bounds_store(%buffer: memref<128x128xf32, 3>, %data: vector<4xf32>, %idx0: index, %idx1: index) {
  miopen.in_bounds_store %data -> %buffer[%idx0, %idx1]
  : vector<4xf32> -> memref<128x128xf32, 3>, index, index
  return
}
// CHECK-LABEL: func.func @miopen_in_bounds_store
// CHECK-NEXT: miopen.in_bounds_store


func.func @miopen_in_warp_transpose(%v : vector<8xf32>) -> vector<8xf32> {
  %cst4 = arith.constant 4 : index
  %l = miopen.workitem_id : index
  %l2 = arith.remui %l, %cst4 : index
  %0 = miopen.in_warp_transpose { size = 4 : i32,
    inGroupPerm = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
  } %v, %l2 : vector<8xf32>, index
  return %0 : vector<8xf32>
}
// CHECK-LABEL: func.func @miopen_in_warp_transpose
// CHECK: miopen.in_warp_transpose
