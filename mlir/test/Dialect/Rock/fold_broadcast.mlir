// RUN: rocmlir-opt --rock-fold-broadcast %s | FileCheck %s
#map = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * 16 + d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["dim2", "dim0", "dim1"] at [0, 1, 2] -> ["dim2", "dim0", "dim1"] at [2, 0, 1]>] bounds = [1, 8, 32] -> [8, 32, 1]>
#transform_map1 = #rock.transform_map<#map1 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [4, 8, 32] -> [1, 8, 32]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1, 16} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 16, 32] -> [16, 32]>
#transform_map3 = #rock.transform_map<#map1 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [4, 16, 32] -> [1, 16, 32]>

func.func @mlir_dot_add_1(%arg0: tensor<8x32x1xf16>, %arg1: tensor<4x8x16xf16>, %arg2: tensor<16x32xf16>) -> tensor<4x8x32xf16> attributes {arch = "", kernel} {
  %0 = rock.transform %arg0 by #transform_map : tensor<8x32x1xf16> to tensor<1x8x32xf16>
  %1 = rock.transform %0 by #transform_map1 : tensor<1x8x32xf16> to tensor<4x8x32xf16>
  %2 = rock.transform %arg2 by #transform_map2 : tensor<16x32xf16> to tensor<1x16x32xf16>
  %3 = rock.transform %2 by #transform_map3 : tensor<1x16x32xf16> to tensor<4x16x32xf16>
  %4 = bufferization.alloc_tensor() : tensor<4x8x32xf16>
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  // CHECK: %[[foldA:.*]] = rock.transform %arg1 by {{.*}} : tensor<4x8x16xf16> to tensor<32x16xf16>
  // CHECK: %[[unbroadcastB:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf16> to tensor<16x32xf16>
  // CHECK: %[[foldC:.*]] = rock.transform %[[alloc]] by {{.*}} : tensor<4x8x32xf16> to tensor<32x32xf16>
  // CHECK: %[[gemmOut:.*]] = rock.gemm %[[foldC]] = %[[foldA]] * %[[unbroadcastB]] {{.*}} : tensor<32x32xf16> = tensor<32x16xf16> * tensor<16x32xf16>
  // CHECK: %[[untransform:.*]] = rock.tensor_untransform_cast %[[gemmOut]] aka %[[foldC]] : tensor<32x32xf16> to tensor<4x8x32xf16>
  %5 = rock.gemm %4 = %arg1 * %3 features =  none storeMethod =  set {arch = ""} : tensor<4x8x32xf16> = tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
  %6 = tensor.empty() : tensor<4x8x32xf16>
  // CHECK: linalg.generic {{.*}} ins(%[[untransform]], {{.*}}, {{.*}})
  %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %1 : tensor<4x8x32xf16>, tensor<4x8x32xf16>) outs(%6 : tensor<4x8x32xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.addf %in, %in_0 : f16
    linalg.yield %8 : f16
  } -> tensor<4x8x32xf16>
  return %7 : tensor<4x8x32xf16>
}

#map4 = affine_map<(d0, d1, d2) -> (d1 * 16 + d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, 0, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{1, 16} ["exp0", "exp1"] at [1, 0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [16, 1, 32] -> [16, 32]>
#transform_map5 = #rock.transform_map<#map5 by [<Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [16, 4, 32] -> [16, 1, 32]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["dim0", "dim1", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [4, 16, 32] -> [16, 4, 32]>
func.func @mlir_dot_add_2(%arg0: tensor<8x32x1xf16>, %arg1: tensor<4x8x16xf16>, %arg2: tensor<16x32xf16>) -> tensor<4x8x32xf16> attributes {arch = "", kernel} {
  %0 = rock.transform %arg0 by #transform_map : tensor<8x32x1xf16> to tensor<1x8x32xf16>
  %1 = rock.transform %0 by #transform_map1 : tensor<1x8x32xf16> to tensor<4x8x32xf16>

  %2 = rock.transform %arg2 by #transform_map4 : tensor<16x32xf16> to tensor<16x1x32xf16>
  %3 = rock.transform %2 by #transform_map5 : tensor<16x1x32xf16> to tensor<16x4x32xf16>
  %p = rock.transform %3 by #transform_map6 : tensor<16x4x32xf16> to tensor<4x16x32xf16>

  %4 = bufferization.alloc_tensor() : tensor<4x8x32xf16>
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  // CHECK: %[[foldA:.*]] = rock.transform %arg1 by {{.*}} : tensor<4x8x16xf16> to tensor<32x16xf16>
  // CHECK: %[[unbroadcastB:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf16> to tensor<16x32xf16>
  // CHECK: %[[foldC:.*]] = rock.transform %[[alloc]] by {{.*}} : tensor<4x8x32xf16> to tensor<32x32xf16>
  // CHECK: %[[gemmOut:.*]] = rock.gemm %[[foldC]] = %[[foldA]] * %[[unbroadcastB]] {{.*}} : tensor<32x32xf16> = tensor<32x16xf16> * tensor<16x32xf16>
  // CHECK: %[[untransform:.*]] = rock.tensor_untransform_cast %[[gemmOut]] aka %[[foldC]] : tensor<32x32xf16> to tensor<4x8x32xf16>
  %5 = rock.gemm %4 = %arg1 * %p features =  none storeMethod =  set {arch = ""} : tensor<4x8x32xf16> = tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
  %6 = tensor.empty() : tensor<4x8x32xf16>
  // CHECK: linalg.generic {{.*}} ins(%[[untransform]], {{.*}}, {{.*}})
  %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %1 : tensor<4x8x32xf16>, tensor<4x8x32xf16>) outs(%6 : tensor<4x8x32xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.addf %in, %in_0 : f16
    linalg.yield %8 : f16
  } -> tensor<4x8x32xf16>
  return %7 : tensor<4x8x32xf16>
}

#map7 = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 16  + d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map9 = affine_map<(d0, d1, d2) -> (d0 floordiv 4, d0 mod 4, d1, d2)>
#transform_map7 = #rock.transform_map<#map7 by [<Unmerge{1, 1, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>, <PassThrough ["exp3"] at [3] -> ["dim1"] at [1]>] bounds = [1, 1, 16, 32] -> [16, 32]>
#transform_map8 = #rock.transform_map<#map8 by [<Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 4, 16, 32] -> [1, 1, 16, 32]>
#transform_map9 = #rock.transform_map<#map9 by [<Merge{1, 4} ["dim0"] at [0] -> ["dim0", "dim1"] at [0, 1]>, <PassThrough ["dim1", "dim2"] at [1,2] -> ["dim2", "dim3"] at [3,4]>] bounds = [4, 16, 32] -> [1, 4, 16, 32]>
func.func @mlir_dot_add_3(%arg0: tensor<8x32x1xf16>, %arg1: tensor<4x8x16xf16>, %arg2: tensor<16x32xf16>) -> tensor<4x8x32xf16> attributes {arch = "", kernel} {
  %0 = rock.transform %arg0 by #transform_map : tensor<8x32x1xf16> to tensor<1x8x32xf16>
  %1 = rock.transform %0 by #transform_map1 : tensor<1x8x32xf16> to tensor<4x8x32xf16>

  %2 = rock.transform %arg2 by #transform_map7 : tensor<16x32xf16> to tensor<1x1x16x32xf16>
  %3 = rock.transform %2 by #transform_map8 : tensor<1x1x16x32xf16> to tensor<1x4x16x32xf16>
  %p = rock.transform %3 by #transform_map9 : tensor<1x4x16x32xf16> to tensor<4x16x32xf16>

  %4 = bufferization.alloc_tensor() : tensor<4x8x32xf16>
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor()
  // CHECK: %[[foldA:.*]] = rock.transform %arg1 by {{.*}} : tensor<4x8x16xf16> to tensor<32x16xf16>
  // CHECK: %[[unbroadcastB:.*]] = rock.transform {{.*}} by {{.*}} : tensor<4x16x32xf16> to tensor<16x32xf16>
  // CHECK: %[[foldC:.*]] = rock.transform %[[alloc]] by {{.*}} : tensor<4x8x32xf16> to tensor<32x32xf16>
  // CHECK: %[[gemmOut:.*]] = rock.gemm %[[foldC]] = %[[foldA]] * %[[unbroadcastB]] {{.*}} : tensor<32x32xf16> = tensor<32x16xf16> * tensor<16x32xf16>
  // CHECK: %[[untransform:.*]] = rock.tensor_untransform_cast %[[gemmOut]] aka %[[foldC]] : tensor<32x32xf16> to tensor<4x8x32xf16>
  %5 = rock.gemm %4 = %arg1 * %p features =  none storeMethod =  set {arch = ""} : tensor<4x8x32xf16> = tensor<4x8x16xf16> * tensor<4x16x32xf16> -> tensor<4x8x32xf16>
  %6 = tensor.empty() : tensor<4x8x32xf16>
  // CHECK: linalg.generic {{.*}} ins(%[[untransform]], {{.*}}, {{.*}})
  %7 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %1 : tensor<4x8x32xf16>, tensor<4x8x32xf16>) outs(%6 : tensor<4x8x32xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %8 = arith.addf %in, %in_0 : f16
    linalg.yield %8 : f16
  } -> tensor<4x8x32xf16>
  return %7 : tensor<4x8x32xf16>
}
