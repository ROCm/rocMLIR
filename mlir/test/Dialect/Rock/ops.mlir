// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s

func.func @rock_conv(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.conv(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv
// CHECK-NEXT: rock.conv

func.func @rock_conv_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.conv(%filter, %input, %output) features = none {
    filter_layout = ["g" ,"k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}
// CHECK-LABEL: func.func @rock_conv_f16
// CHECK-NEXT: rock.conv

func.func @rock_conv_fp8_mixed(%filter : memref<?x?x?x?x?xf8E4M3FNUZ>, %input : memref<?x?x?x?x?xf8E5M2FNUZ>, %output : memref<?x?x?x?x?xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx942"} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf8E4M3FNUZ>, memref<?x?x?x?x?xf8E5M2FNUZ>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv_fp8_mixed
// CHECK-NEXT: rock.conv

func.func @rock_conv_fp8_mixed_ocp(%filter : memref<?x?x?x?x?xf8E4M3FN>, %input : memref<?x?x?x?x?xf8E5M2>, %output : memref<?x?x?x?x?xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf8E4M3FN>, memref<?x?x?x?x?xf8E5M2>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv_fp8_mixed
// CHECK-NEXT: rock.conv

func.func @rock_conv_bwd_data(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.conv_bwd_data(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 0 : index,
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index],
    usesV4R1 = true
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv_bwd_data
// CHECK-NEXT: rock.conv_bwd_data

func.func @rock_conv_bwd_data_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.conv_bwd_data(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 0 : index,
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index],
    usesV4R1 = true
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}
// CHECK-LABEL: func.func @rock_conv_bwd_data_f16
// CHECK-NEXT: rock.conv_bwd_data

func.func @rock_conv_bwd_weight(%filter : memref<?x?x?x?x?xf32>, %input : memref<?x?x?x?x?xf32>, %output : memref<?x?x?x?x?xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    numCU = 64 : i32,
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index],
    usesV4R1 = true
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}
// CHECK-LABEL: func.func @rock_conv_bwd_weight
// CHECK-NEXT: rock.conv_bwd_weight

func.func @rock_conv_bwd_weight_f16(%filter : memref<?x?x?x?x?xf16>, %input : memref<?x?x?x?x?xf16>, %output : memref<?x?x?x?x?xf16>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    numCU = 64 : i32,
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight_f16
// CHECK-NEXT: rock.conv_bwd_weight

func.func @rock_gemm(%a : memref<32x64xf16>, %b : memref<1x32x128xf16>, %c : memref<64x128xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.gemm %c = tr %a * %b features = none storeMethod = set
  : memref<64x128xf32> = memref<32x64xf16> * memref<1x32x128xf16>
  func.return
}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-NEXT: rock.gemm

func.func @rock_scaled_gemm(%a : memref<32x64xf4E2M1FN>, %b : memref<1x32x128xf4E2M1FN>, %c : memref<64x128xf32>, %scaleA : memref<32x64xf8E8M0FNU>, %scaleB : memref<1x32x128xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.gemm %c = tr %a scaled by tr %scaleA * %b scaled by %scaleB features = mfma storeMethod = set
  : memref<64x128xf32> = memref<32x64xf4E2M1FN> scaled by memref<32x64xf8E8M0FNU> * memref<1x32x128xf4E2M1FN> scaled by memref<1x32x128xf8E8M0FNU>
  func.return
}
// CHECK-LABEL: func.func @rock_scaled_gemm
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
      <Pad{1, 1} ["0ipad"] at [3] -> ["0i"] at [3]>,
      <Pad{2, 2} ["1ipad"] at [4] -> ["1i"] at [4]>
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
      #rock.transform<Merge{64, 32, 16} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>,
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
      #rock.transform<Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>,
      #rock.transform<Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>
      // Note: fake data should work fine for now
     ] bounds = [0, 0, 0, 0, 0, 0, 0] -> [0, 0, 0, 0, 0]>
  : memref<?x?x?x?x?xf32> to memref<?x?x?x?x?x?x?xf32, #map2>
  return
}

// CHECK-LABEL: func.func @rock_transform_1_to_n
//  CHECK-NEXT: rock.transform

func.func @rock_gridwise_gemm(%A : memref<2x72x128xf32>, %B : memref<2x72x256xf32>, %C : memref<2x128x256xf32>) {
  rock.gridwise_gemm %C = %A * %B storeMethod(set) features = none {
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
      nPerThread = 4,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2>
  } : memref<2x128x256xf32> = memref<2x72x128xf32> * memref<2x72x256xf32>
  return
}

// CHECK-LABEL: func.func @rock_gridwise_gemm
//  CHECK-NEXT: rock.gridwise_gemm

func.func @rock_gridwise_gemm_accel(%A : memref<2x1024x1024xf32>, %B : memref<2x1024x2048xf32>, %C : memref<2x1024x2048xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx908", numCU = 64 : i32} {
  rock.gridwise_gemm_accel(%A, %B, %C) storeMethod(set) features = none {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<2x1024x1024xf32>, memref<2x1024x2048xf32>, memref<2x1024x2048xf32>
  return
}

// CHECK-LABEL: func.func @rock_gridwise_gemm_accel
// CHECK-NEXT: rock.gridwise_gemm_accel

func.func @rock_gridwise_scaled_gemm_accel(%A : memref<2x1024x1024xf4E2M1FN>, %B : memref<2x1024x2048xf4E2M1FN>, %C : memref<2x1024x2048xf32>, %scaleA : memref<2x1024x1024xf8E8M0FNU>, %scaleB : memref<2x1024x2048xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950", numCU = 256 : i32} {
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA, %scaleB) storeMethod (set) features = mfma {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1,
      scheduleVersion = 1,
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<2x1024x1024xf4E2M1FN>, memref<2x1024x2048xf4E2M1FN>, memref<2x1024x2048xf32>, memref<2x1024x1024xf8E8M0FNU>, memref<2x1024x2048xf8E8M0FNU>
  return
}

// CHECK-LABEL: func.func @rock_gridwise_scaled_gemm_accel
// CHECK-NEXT: rock.gridwise_gemm_accel

func.func @rock_blockwise_gemm_accel_scaled(%matrixA : memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>, 
                                                %matrixB : memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
                                                %matrixScaleA : memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
                                                %matrixScaleB : memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
                                                %bufferA : memref<4xf4E2M1FN, #gpu.address_space<private>>, 
                                                %bufferB : memref<4xf4E2M1FN, #gpu.address_space<private>>,
                                                %bufferScaleA : memref<4xf8E8M0FNU, #gpu.address_space<private>>,
                                                %bufferScaleB : memref<4xf8E8M0FNU, #gpu.address_space<private>>,
                                                %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA scaled by %bufferScaleA from %matrixScaleA * %bufferB from %matrixB scaled by %bufferScaleB from %matrixScaleB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx950",
    blockSize= 256 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    loadAfromLDS,
    loadBfromLDS,
    elementTypeA = f4E2M1FN,
    elementTypeB = f4E2M1FN,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1,
      scheduleVersion = 1,
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>> scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>> * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>> scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: @rock_blockwise_gemm_accel_scaled
// CHECK-NEXT: rock.blockwise_gemm_accel

// ----

func.func @rock_threadwise_gemm_accel_scaled(%matrixA : memref<1x4xvector<4xf4E2M1FN>, 5>,
                                                %matrixB : memref<1x4xvector<4xf4E2M1FN>, 5>,
                                                %matrixC : memref<1x1xvector<32xf32>, 5>, %scaleA : memref<1x4xvector<4xf8E8M0FNU>, 5>, %scaleB : memref<1x4xvector<4xf8E8M0FNU>, 5>) {
  %c0 = arith.constant 0 : index
    rock.threadwise_accel_gemm %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] features = mfma{
    arch = "amdgcn-amd-amdhsa:gfx950",
    params = #rock.xdlops_gemm_derived_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      kpack = 1,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1x1xvector<32xf32>, 5> += memref<1x4xvector<4xf4E2M1FN>, 5> scaled by memref<1x4xvector<4xf8E8M0FNU>, 5> * memref<1x4xvector<4xf4E2M1FN>, 5> scaled by memref<1x4xvector<4xf8E8M0FNU>, 5>
  return
}
// CHECK-LABEL: func.func @rock_threadwise_gemm_accel_scaled
// CHECK: rock.threadwise_accel_gemm

// ----


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

func.func @converting_copy_kernel(%arg0 : memref<2x4xf32>, %arg1: memref<2x4xf16>) {
  rock.converting_copy_kernel %arg0 to %arg1 : memref<2x4xf32> to memref<2x4xf16>
  func.return
}

// CHECK-LABEL: func.func @gridwise_attn_atomic_add
// CHECK: rock.gridwise_attention_accel
func.func @gridwise_attn_atomic_add(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    firstGemmIndices = array<i64: 0>,
    storeMethod = #rock<StoreMethod atomic_add>,
    splitKV = 1 : i32,
    enableSoftmax = false,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 1, 0>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

// CHECK-LABEL: func.func @attention
// CHECK: rock.attention
func.func @attention(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}
