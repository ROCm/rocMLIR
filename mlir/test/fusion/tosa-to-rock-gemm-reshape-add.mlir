// RUN: rocmlir-driver --host-pipeline highlevel %s | rocmlir-opt --rock-fold-transpose | FileCheck %s --check-prefix=CHECK_FOLD_TP
// CHECK_FOLD_TP: #map = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK_FOLD_TP: #[[MAP1:.*]] = #rock.transform_map<#map by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <PassThrough ["dim0"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 1, 1000] -> [1, 1000]>
// CHECK_FOLD_TP: %[[ALLOC:.*]] = memref.alloc() : memref<1x1000xf32>
// CHECK_FOLD_TP: %[[TR1:.*]] = rock.transform %[[ALLOC]] by #[[MAP1]] : memref<1x1000xf32> to memref<1x1x1000xf32>
// CHECK_FOLD_TP: rock.gemm %[[TR1]] =

// RUN: rocmlir-driver --host-pipeline highlevel %s | rocmlir-opt --rock-fold-transpose --rock-affix-params --rock-conv-to-gemm --rock-gemm-to-gridwise --rock-gridwise-gemm-to-blockwise --rock-linalg-align | FileCheck %s --check-prefix=CHECK_LINALG_ALIGN

// CHECK_LINALG_ALIGN: #[[AMAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK_LINALG_ALIGN: #[[MAP1:.*]] = #rock.transform_map<#[[AMAP]] by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <PassThrough ["dim0"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 1, 1000] -> [1, 1000]>
// CHECK_LINALG_ALIGN: rock.transforming_for{{.*}}#[[MAP1]]
// CHECK_LINALG_ALIGN: rock.global_load %arg2
// CHECK_LINALG_ALIGN: linalg.generic
// CHECK_LINALG_ALIGN-SAME: outs(%[[outBuf:.*]] : memref<4xf32, #gpu.address_space<private>>)
// CHECK_LINALG_ALIGN: global_store %[[outBuf]]
// CHECK_LINALG_ALIGN-SAME: -> %arg3
// to test reshape is converted as transform and fused.

func.func @test_fusion(%arg0: tensor<1x1x512xf32> {func.read_access}, %arg1: tensor<1x512x1000xf32> {func.read_access}, %arg2: tensor<1x1000xf32> {func.read_access}) -> (tensor<1x1000xf32> {func.write_access}) attributes {kernel, arch = ""} {
    %2 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x1x512xf32>, tensor<1x512x1000xf32>) -> tensor<1x1x1000xf32>
    %3 = "tosa.reshape"(%2) {new_shape = array<i64: 1, 1000>} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %4 : tensor<1x1000xf32>
}
