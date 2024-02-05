// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// Run: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s


func.func @rock_conv2d(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  rock.conv2d(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1 : i32,  1 : i32],
    strides = [1 : i32,  1 : i32],
    padding = [0 : i32,  0 : i32,  0 : i32,  0 : i32]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv2d
// CHECK-NEXT: rock.conv2d

func.func @rock_conv2d_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) {
  rock.conv2d(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g" ,"k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1 : i32,  1 : i32],
    strides = [1 : i32,  1 : i32],
    padding = [0 : i32,  0 : i32,  0 : i32,  0 : i32]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}
// CHECK-LABEL: func.func @rock_conv2d_f16
// CHECK-NEXT: rock.conv2d

func.func @rock_conv2d_fp8_mixed(%filter : memref<?x?x?x?x?xf8E4M3FNUZ>, %input : memref<?x?x?x?x?xf8E5M2FNUZ>, %output : memref<?x?x?x?x?xf32>) {
  rock.conv2d(%filter, %input, %output) features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx940",
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1 : i32,  1 : i32],
    strides = [1 : i32,  1 : i32],
    padding = [0 : i32,  0 : i32,  0 : i32,  0 : i32]
  } : memref<?x?x?x?x?xf8E4M3FNUZ>, memref<?x?x?x?x?xf8E5M2FNUZ>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv2d_fp8_mixed
// CHECK-NEXT: rock.conv2d

func.func @rock_conv2d_bwd_data(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  rock.conv2d_bwd_data(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    kernelId = 0 : index,
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1 : i32,  1 : i32],
    strides = [1 : i32,  1 : i32],
    padding = [0 : i32,  0 : i32,  0 : i32,  0 : i32]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv2d_bwd_data
// CHECK-NEXT: rock.conv2d_bwd_data

func.func @rock_conv2d_bwd_data_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) {
  rock.conv2d_bwd_data(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    kernelId = 0 : index,
    input_layout = ["n", "gi", "c", "hi", "wi"],
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1 : i32,  1 : i32],
    strides = [1 : i32,  1 : i32],
    padding = [0 : i32,  0 : i32,  0 : i32,  0 : i32]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}
// CHECK-LABEL: func.func @rock_conv2d_bwd_data_f16
// CHECK-NEXT: rock.conv2d_bwd_data

func.func @rock_conv2d_bwd_weight(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) {
  rock.conv2d_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    numCU = 64 : i32,
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1 : i32,  1 : i32],
    strides = [1 : i32,  1 : i32],
    padding = [0 : i32,  0 : i32,  0 : i32,  0 : i32]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv2d_bwd_weight
// CHECK-NEXT: rock.conv2d_bwd_weight

func.func @rock_conv2d_bwd_weight_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) {
  rock.conv2d_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["n", "gi", "c", "hi", "wi"],
    numCU = 64 : i32,
    output_layout = ["n", "go", "k", "ho", "wo"],
    dilations = [1 : i32,  1 : i32],
    strides = [1 : i32,  1 : i32],
    padding = [0 : i32,  0 : i32,  0 : i32,  0 : i32]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv2d_bwd_weight_f16
// CHECK-NEXT: rock.conv2d_bwd_weight

func.func @rock_gemm(%a : memref<32x64xf16>, %b : memref<1x32x128xf16>, %c : memref<64x128xf32>) {
  rock.gemm %c = tr %a * %b features = none storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906"
  } : memref<64x128xf32> = memref<32x64xf16> * memref<1x32x128xf16>
  func.return
}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-NEXT: rock.gemm

// Affine maps needed when testing transform
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d2, d3 - 1, d4 - 2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 512,
  (d1 mod 512) floordiv 16, d1 mod 16)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) ->
  (d1, d0, d2, d3 + d4, d5 + d6)>

// test 1-1 dimension mappings.
func.func @rock_transform_1_to_1(%memref: memref<1x2x3x4x5xf32, 3>) {
  %transformed_memref = rock.transform %memref by
    <#map0 by [
      <PassThrough ["g"] at [0] -> ["g"] at [1]>,
      <PassThrough ["n"] at [1] -> ["n"] at [0]>,
      <PassThrough ["c"] at [2] -> ["c"] at [2]>,
      <Pad{1, 1} ["hipad"] at [3] -> ["hi"] at [3]>,
      <Pad{2, 2} ["wipad"] at [4] -> ["wi"] at [4]>
    ] bounds = [2, 1, 3, 6, 9] -> [1, 2, 3, 4, 5]>
  : memref<1x2x3x4x5xf32, 3> to memref<2x1x3x6x9xf32, #map0, 3>
  return
}
// CHECK-LABEL: func.func @rock_transform_1_to_1
//  CHECK-NEXT: rock.transform

// test multiple source dimensions map to 1 target dimension.
func.func @rock_transform_n_to_1(%memref : memref<1x128x64x32x16xf32>) {
  %transformed_memref = rock.transform %memref by
    <#map1 by [
      #rock.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>,
      #rock.transform<Merge{64, 32, 16} ["gemmK"] at [1] -> ["c", "y", "x"] at [2, 3, 4]>,
      #rock.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>
    ] bounds = [1, 32768, 128] -> [1, 128, 64, 32, 16]>
  : memref<1x128x64x32x16xf32> to memref<1x32768x128xf32, #map1>
  return
}
// CHECK-LABEL: func.func @rock_transform_n_to_1
//  CHECK-NEXT: rock.transform

// test 1 source dimension map to multiple target dimensions.
func.func @rock_transform_1_to_n(%memref : memref<?x?x?x?x?xf32>) {
  %transformed_memref = rock.transform %memref by
    <#map2 by [
      #rock.transform<PassThrough ["n", "g", "c"] at [0, 1, 2] ->
        ["n", "g", "c"] at [1, 0, 2]>,
      #rock.transform<Embed{1, 1} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>,
      #rock.transform<Embed{1, 1} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>
      // Note: fake data should work fine for now
     ] bounds = [0, 0, 0, 0, 0, 0, 0] -> [0, 0, 0, 0, 0]>
  : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32, #map2>
  return
}

// CHECK-LABEL: func.func @rock_transform_1_to_n
//  CHECK-NEXT: rock.transform

func.func @rock_gridwise_gemm(%A : memref<2x72x128xf32>, %B : memref<2x72x256xf32>, %C : memref<2x128x256xf32>) {
  rock.gridwise_gemm %C = %A * %B features = none {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.general_gemm_params<
      blockSize = 128,
      kPerBlock = 8,
      kPerThread = 1,
      kpack = 1,
      mPerBlock = 128,
      mPerThread = 4,
      nPerBlock = 128,
      nPerThread = 4>
  } : memref<2x128x256xf32> = memref<2x72x128xf32> * memref<2x72x256xf32>
  return
}

// CHECK-LABEL: func.func @rock_gridwise_gemm
//  CHECK-NEXT: rock.gridwise_gemm

func.func @rock_gridwise_gemm_accel(%A : memref<2x1024x1024xf32>, %B : memref<2x1024x2048xf32>, %C : memref<2x1024x2048xf32>) {
  rock.gridwise_gemm_accel(%A, %B, %C) storeMethod(set) features = none {
    arch = "amdgcn-amd-amdhsa:gfx908",
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      forceUnroll = true>
  } : memref<2x1024x1024xf32>, memref<2x1024x2048xf32>, memref<2x1024x2048xf32>
  return
}

// CHECK-LABEL: func.func @rock_gridwise_gemm_accel
// CHECK-NEXT: rock.gridwise_gemm_accel

func.func @rock_extract_slice(%v : vector<32xf32>) -> vector<4xf32> {
  %i = arith.constant 0 : index
  %r = rock.extract_slice %v[%i] : vector<32xf32> -> vector<4xf32>
  return %r : vector<4xf32>
}
// CHECK-LABEL: func.func @rock_extract_slice
// CHECK: rock.extract_slice

func.func @rock_insert_slice(%u: vector<4xf32>, %v: vector<32xf32>) -> vector<32xf32> {
  %i = arith.constant 0 : index
  %w = rock.insert_slice %u -> %v[%i] : vector<4xf32> -> vector<32xf32>
  return %w : vector<32xf32>
}
// CHECK-LABEL: func.func @rock_insert_slice
// CHECK: rock.insert_slice

func.func @rock_in_bounds_load(%buffer: memref<128x128xf32, 3>, %idx0: index, %idx1: index) -> vector<4xf32> {
  %ret = rock.in_bounds_load %buffer[%idx0, %idx1]
    : memref<128x128xf32, 3>, index, index -> vector<4xf32>
  return %ret : vector<4xf32>
}
// CHECK-LABEL: func.func @rock_in_bounds_load
// CHECK-NEXT: rock.in_bounds_load

func.func @rock_in_bounds_store(%buffer: memref<128x128xf32, 3>, %data: vector<4xf32>, %idx0: index, %idx1: index) {
  rock.in_bounds_store %data -> %buffer[%idx0, %idx1]
  : vector<4xf32> -> memref<128x128xf32, 3>, index, index
  return
}
// CHECK-LABEL: func.func @rock_in_bounds_store
// CHECK-NEXT: rock.in_bounds_store


func.func @rock_in_warp_transpose(%v : vector<8xf32>) -> vector<8xf32> {
  %cst4 = arith.constant 4 : index
  %l = rock.workitem_id : index
  %l2 = arith.remui %l, %cst4 : index
  %0 = rock.in_warp_transpose { size = 4 : i32,
    inGroupPerm = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
  } %v, %l2 : vector<8xf32>, index
  return %0 : vector<8xf32>
}
// CHECK-LABEL: func.func @rock_in_warp_transpose
// CHECK: rock.in_warp_transpose


func.func @init_kernel(%arg0 : memref<2x4xf32>) {
  rock.init_kernel %arg0 features = none : memref<2x4xf32>
  func.return
}
// CHECK-LABEL func.func @init_kernel
// CHECK: rock.init_kernel

func.func @converting_copy_kernel(%arg0 : memref<2x4xf32>, %arg1: memref<2x4xf16>) {
  rock.converting_copy_kernel %arg0 to %arg1 features = none : memref<2x4xf32> to memref<2x4xf16>
  func.return
}
