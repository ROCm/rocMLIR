// RUN: rocmlir-driver --host-pipeline highlevel %s | rocmlir-opt --rock-affix-params --rock-conv-to-gemm --rock-gemm-to-gridwise -rock-regularize -rock-gridwise-gemm-to-blockwise -rock-linalg-align | FileCheck %s --check-prefix=CHECK_LINALG_ALIGN

// CHECK_LINALG_ALIGN-DAG: #[[AMAP:.*]] = affine_map<(d0, d1, d2) -> (d0 + d1, d2)>
// CHECK_LINALG_ALIGN-DAG: #[[AMAP1:.*]] = affine_map<(d0, d1) -> (d0 * 1000 + d1)>
// CHECK_LINALG_ALIGN-DAG: #[[MAP1:.*]] = #rock.transform_map<#[[AMAP]] by [<Unmerge{1, 1} ["col0", "col1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 1, 1000] -> [1, 1000]>
// CHECK_LINALG_ALIGN-DAG: #[[MAP2:.*]] = #rock.transform_map<#[[AMAP1]] by [<Unmerge{1, 1000} ["col0", "col1"] at [0, 1] -> ["dim0"] at [0]>] bounds = [1, 1000] -> [1000]>

// CHECK_LINALG_ALIGN-COUNT-2: rock.threadwise_read_into {{.*}}
// CHECK_LINALG_ALIGN: rock.threadwise_read_into {{.*}} -> [[lain:%.*]] :
// CHECK_LINALG_ALIGN: linalg.generic{{.*}} ins({{.*}}, [[lain]] :{{.*}}) outs(%[[outBuf:.*]] : memref<16xf32, #gpu.address_space<private>>)
// CHECK_LINALG_ALIGN: rock.threadwise_write_all {{.*}} %[[outBuf]] ->
// to test reshape is converted as transform and fused.

func.func @test_fusion(%arg0: tensor<1x1x512xf32> {func.read_access}, %arg1: tensor<1x512x1000xf32> {func.read_access}, %arg2: tensor<1x1000xf32> {func.read_access}) -> (tensor<1x1000xf32> {func.write_access}) attributes {kernel, arch = ""} {
    %2 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x1x512xf32>, tensor<1x512x1000xf32>) -> tensor<1x1x1000xf32>
    %3 = "tosa.reshape"(%2) {new_shape = array<i64: 1, 1000>} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<1x1000xf32>, tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %4 : tensor<1x1000xf32>
}
