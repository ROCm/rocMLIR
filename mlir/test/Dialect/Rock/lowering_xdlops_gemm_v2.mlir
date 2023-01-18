// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s
#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 8 + d1)> by [<Embed{8, 1} ["m", "k"] at [0, 1] -> ["raw"] at [0]>] bounds = [1, 8] -> [8]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 8 + d1)> by [<Embed{8, 1} ["n", "k"] at [0, 1] -> ["raw"] at [0]>] bounds = [1, 8] -> [8]>
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 2 + d1 * 2 + d2)> by [<Embed{2, 2, 1} ["m", "n", "v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [1, 1, 2] -> [2]>

#map0 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0 * 2 + d1 * 2 + d2)>

func.func @rock_xdlops_gemm_v2_nonreduction_nokpack(%matrixA : memref<8xf32, 5>,
                                                      %matrixB : memref<8xf32, 5>,
                                                      %matrixC : memref<2xvector<32xf32>, 5>) {
  // CHECK-LABEL: func.func @rock_xdlops_gemm_v2_nonreduction_nokpack
  // CHECK: rock.in_bounds_load
  // CHECK: rock.in_bounds_load
  // CHECK: amdgpu.mfma
  %A = rock.transform %matrixA by #transform_map0 : memref<8xf32, 5> to memref<1x8xf32, #map0, 5>
  %B = rock.transform %matrixB by #transform_map1 : memref<8xf32, 5> to memref<1x8xf32, #map0, 5>
  %C = rock.transform %matrixC by #transform_map2 : memref<2xvector<32xf32>, 5> to memref<1x1x2xvector<32xf32>, #map1, 5>
  rock.xdlops_gemm_v2 %C += %A * %B {
    params = #rock.xdlops_gemm_params<
       kPerBlock = 8,
       kpack = 1,
       mPerBlock = 128,
       mPerWave = 64,
       nPerBlock = 64,
       nPerWave = 32,
       forceUnroll = true>
     } : memref<1x1x2xvector<32xf32>, #map1, 5> += memref<1x8xf32, #map0, 5> * memref<1x8xf32, #map0, 5>
  return
}

#transform_map3 = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 2 + d1)> by [<Embed{2, 1} ["m", "k"] at [0, 1] -> ["raw"] at [0]>] bounds = [2, 1] -> [2]>
#transform_map4 = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 2 + d1)> by [<Embed{2, 1} ["n", "k"] at [0, 1] -> ["raw"] at [0]>] bounds = [2, 1] -> [2]>
#transform_map5 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2)> by [<Embed{2, 2, 1} ["m", "n", "v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 2, 1] -> [2]>

#map2 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2)>

func.func @rock_xdlops_gemm_v2_reduction_kpack_f32(%matrixA : memref<2xvector<2xf32>, 5>,
                                                    %matrixB : memref<2xvector<2xf32>, 5>,
                                                    %matrixC : memref<4xvector<16xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK-LABEL: func.func @rock_xdlops_gemm_v2_reduction_kpack_f32
  // CHECK: rock.extract_slice
  // CHECK: rock.extract_slice
  // CHECK: amdgpu.mfma
  %A = rock.transform %matrixA by #transform_map3 : memref<2xvector<2xf32>, 5> to memref<2x1xvector<2xf32>, #map2, 5>
  %B = rock.transform %matrixB by #transform_map4 : memref<2xvector<2xf32>, 5> to memref<2x1xvector<2xf32>, #map2, 5>
  %C = rock.transform %matrixC by #transform_map5 : memref<4xvector<16xf32>, 5> to memref<2x2x1xvector<16xf32>, #map3, 5>
  rock.xdlops_gemm_v2 %C += %A * %B {
    params = #rock.xdlops_gemm_params<
      kPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      forceUnroll = true>
  } : memref<2x2x1xvector<16xf32>, #map3, 5> += memref<2x1xvector<2xf32>, #map2, 5> * memref<2x1xvector<2xf32>, #map2, 5>
  return
}

func.func @rock_xdlops_gemm_v2_reduction_kpack_i8(%matrixA : memref<2xvector<8xi8>, 5>,
                                                 %matrixB : memref<2xvector<8xi8>, 5>,
                                                 %matrixC : memref<1xvector<16xi32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK-LABEL: func.func @rock_xdlops_gemm_v2_reduction_kpack_i8
  // CHECK: rock.extract_slice
  // CHECK: rock.extract_slice
  // CHECK: amdgpu.mfma
  // CHECK-NOT: amdgpu.mfma
  %A = rock.transform %matrixA by #transform_map3 : memref<2xvector<8xi8>, 5> to memref<1x2xvector<8xi8>, #map2, 5>
  %B = rock.transform %matrixB by #transform_map4 : memref<2xvector<8xi8>, 5> to memref<1x2xvector<8xi8>, #map2, 5>
  %C = rock.transform %matrixC by #transform_map5 : memref<1xvector<16xi32>, 5> to memref<1x1x1xvector<16xi32>, #map3, 5>
  rock.xdlops_gemm_v2 %C += %A * %B {
    params = #rock.xdlops_gemm_params<
      kPerBlock = 4,
      kpack = 8,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      forceUnroll = true>
  } : memref<1x1x1xvector<16xi32>, #map3, 5> += memref<1x2xvector<8xi8>, #map2, 5> * memref<1x2xvector<8xi8>, #map2, 5>
  return
}
