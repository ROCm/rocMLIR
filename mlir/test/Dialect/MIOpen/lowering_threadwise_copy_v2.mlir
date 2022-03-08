// RUN: miopen-opt -miopen-lowering-step3 -miopen-lowering-step4 %s | FileCheck %s
#gemm_padding0 = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0, bwdPaddingInfo = "NA">

#map0 = affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 4 + d2)>
#transform_map0 = #miopen.transform_map<#map0 by [
  #miopen.transform<Embed{32, 4, 1} ["no", "ho", "wo"] at [0, 1, 2] -> ["vector"] at [0]>
] bounds = [1, 8, 4] -> [32]>

#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 4 + d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 8 + d2 * 4 + d3, d4)>
#map3 = affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, (d2 mod 196) floordiv 14, (d2 mod 196) mod 14)>

#transform_map1 = #miopen.transform_map<#map1 by [
  #miopen.transform<Embed{0, 4, 0, 1, 0} ["g", "m0", "m1", "m2", "n"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>
] bounds = [1, 4, 1, 4, 1] -> [16]>
#transform_map2 = #miopen.transform_map<#map2 by [
  #miopen.transform<PassThrough ["g"] at [0] -> ["gemmG"] at [0]>,
  #miopen.transform<Embed{8, 4, 1} ["m0", "m1", "m2"] at [1, 2, 3] -> ["gemmM"] at [1]>,
  #miopen.transform<PassThrough ["n"] at [4] -> ["gemmN"] at [2]>
] bounds = [1, 128, 2, 4, 25088] -> [1, 1024, 25088]>
#transform_map3 = #miopen.transform_map<#map3 by [
  #miopen.transform<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>,
  #miopen.transform<PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>,
  #miopen.transform<Merge{128, 14, 14} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 3, 4]>
] bounds = [1, 1024, 25088] -> [128, 1, 1024, 14, 14]>

// CHECK-LABEL: func @miopen_threadwise_copy_v2
func @miopen_threadwise_copy_v2(%source : vector<32xf32>,
                                %dest1D : memref<32xf32>,
                                %dest5D : memref<128x1x1024x14x14xf32>) {
  %c0 = arith.constant 0 : index

  // A simplified usage of threadwise_copy_v2.
  // Source vector has a transformation.
  // Source vector has no offset.
  // Source vector has a bound.
  // Dest memref has a transformation.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2 %source[%c0, %c0, %c0] ->
                            %dest1D[%c0, %c0, %c0]
  with [[#transform_map0], [#transform_map0]] {
    sourceOffset = 0 : index,
    data_per_copy = 1 : index,
    vector_read_write_dim = 0 : i32,
    upper_vector_read_dim = -1 : i32,
    bounds = [1 : index, 8 : index, 4 : index],
    storeMethod = 0 : i32,
    paddingInfo = #gemm_padding0,
    destOobDims = [false]
  } : vector<32xf32>, index, index, index ->
    memref<32xf32>, index, index, index

  // A real use case of threadwise_copy_v2.
  // Source vector has a transformation.
  // Source vector has offset and bound.
  // Dest memref has 2 transformations.
  // CHECK-NOT: scf.for
  miopen.threadwise_copy_v2 %source[%c0, %c0, %c0, %c0, %c0] ->
    %dest5D[%c0, %c0, %c0, %c0, %c0]
    with [[#transform_map1], [#transform_map2, #transform_map3]] {
      sourceOffset = 16 : index,
      bounds = [1 : index, 4 : index, 1 : index, 4 : index, 1 : index],
      storeMethod = 0 : i32,
      paddingInfo = #gemm_padding0,
      destOobDims = [false, false, false, false, false],
      upper_vector_read_dim = 0 : i32,
      data_per_copy = 1 : i32,
      dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32],
      vector_read_write_dim = 4 : i32}
      : vector<32xf32>, index, index, index, index, index ->
      memref<128x1x1024x14x14xf32>, index, index, index, index, index
  return
}

#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1 * 4 + d5)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 8 + d2 * 4 + d3, d4 * 4 + d5)>
#map6 = affine_map<(d0, d1, d2) -> (d2 floordiv 256, d0, d1, (d2 mod 256) floordiv 16, d2 mod 16)>
#transform_map4 = #miopen.transform_map<#map4 by [
  #miopen.transform<Embed{0, 4, 0, 0, 0, 1}
    ["g", "m0", "m1", "m2", "n0", "n1"] at [0, 1, 2, 3, 4, 5] -> ["raw"] at [0]>
] bounds = [1, 4, 1, 1, 1, 4] -> [16]>
#transform_map5 = #miopen.transform_map<#map5 by [
  #miopen.transform<PassThrough ["g"] at [0] -> ["gemmG"] at [0]>,
  #miopen.transform<Embed{8, 4, 1} ["m0", "m1", "m2"] at [1, 2, 3] -> ["gemmM"] at [1]>,
  #miopen.transform<Embed{4, 1} ["n0", "n1"] at [4, 5] -> ["gemmN"] at [2]>
] bounds = [1, 128, 2, 4, 8192, 4] -> [1, 1024, 32768]>
#transform_map6 = #miopen.transform_map<#map6 by [
  #miopen.transform<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>,
  #miopen.transform<PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>,
  #miopen.transform<Merge{128, 16, 16} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 3, 4]>
] bounds = [1, 1024, 32768] -> [128, 1, 1024, 16, 16]>

// CHECK-LABEL: @miopen_threadwise_copy_v2_vectorized_nchw
func @miopen_threadwise_copy_v2_vectorized_nchw(%source : vector<32xf32>,
                                %dest5D : memref<128x1x1024x16x16xf32>) {
  %c0 = arith.constant 0 : index

  // A usecase of threadwise_copy_v2 that should be vectorized
  // This threadwise_copy takes the extra n dimension split used in swizzling
  // and has dimensions that are an even multiple of 4 to prevent OOB checks
  // CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %c0_i32, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>,
  miopen.threadwise_copy_v2 %source[%c0, %c0, %c0, %c0, %c0, %c0] ->
    %dest5D[%c0, %c0, %c0, %c0, %c0, %c0]
    with [[#transform_map4], [#transform_map5, #transform_map6]] {
      sourceOffset = 0 : index,
      bounds = [1 : index, 4 : index, 1 : index, 1 : index, 1 : index, 4 : index],
      storeMethod = 0 : i32,
      paddingInfo = #gemm_padding0,
      destOobDims = [false, false, false, false, false],
      data_per_copy = 4 : i32,
      dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32],
      vector_read_write_dim = 4 : i32,
      upper_vector_read_dim = 5 : i32}
      : vector<32xf32>, index, index, index, index, index, index ->
      memref<128x1x1024x16x16xf32>, index, index, index, index, index, index

  return
}

#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 4 + d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 8 + d2 * 4 + d3, d4)>
#map9 = affine_map<(d0, d1, d2) -> (d2 floordiv 256, d0, (d2 mod 256) floordiv 16, d2 mod 16, d1)>

#transform_map7 = #miopen.transform_map<#map7 by [
  #miopen.transform<Embed{0, 4, 0, 1, 0}
    ["g", "m0", "m1", "m2", "n"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>
] bounds = [1, 4, 1, 4, 1] -> [16]>
#transform_map8 = #miopen.transform_map<#map8 by [
  #miopen.transform<PassThrough ["g"] at [0] -> ["gemmG"] at [0]>,
  #miopen.transform<Embed{8, 4, 1} ["m0", "m1", "m2"] at [1, 2, 3] -> ["gemmM"] at [1]>,
  #miopen.transform<PassThrough ["n"] at [4] -> ["gemmN"] at [2]>
] bounds = [1, 128, 2, 4, 32768] -> [1, 1024, 32768]>
#transform_map9 = #miopen.transform_map<#map9 by [
  #miopen.transform<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>,
  #miopen.transform<PassThrough ["gemmM"] at [1] -> ["ko"] at [4]>,
  #miopen.transform<Merge{128, 16, 16} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 2, 3]>
] bounds = [1, 1024, 32768] -> [128, 1, 16, 16, 1024]>

// CHECK-LABEL: @miopen_threadwise_copy_v2_vectorized_nhwc
func @miopen_threadwise_copy_v2_vectorized_nhwc(%source_offset : i32,
                                %source : vector<32xf32>,
                                %dest5D : memref<128x1x16x16x1024xf32>) {
  %c0 = arith.constant 0 : index

  // A usecase of threadwise_copy_v2 that should be vectorized
  // This threadwise_copy takes the extra n dimension split used in swizzling
  // and has dimensions that are an even multiple of 4 to prevent OOB checks
  // CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %c0_i32, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>,
  miopen.threadwise_copy_v2 %source[%c0, %c0, %c0, %c0, %c0] ->
    %dest5D[%c0, %c0, %c0, %c0, %c0]
    with [[#transform_map7], [#transform_map8, #transform_map9]] {
      sourceOffset = 0 : index,
      bounds = [1 : index, 4 : index, 1 : index, 4 : index, 1 : index],
      storeMethod = 0 : i32,
      paddingInfo = #gemm_padding0,
      destOobDims = [false, false, false, false, false],
      data_per_copy = 4 : i32,
      dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32],
      vector_read_write_dim = 4 : i32,
      upper_vector_read_dim = 3 : i32}
      : vector<32xf32>, index, index, index, index, index ->
      memref<128x1x16x16x1024xf32>, index, index, index, index, index

  return
}

