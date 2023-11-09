// Unit tests for rock-lower-reduce pass

// RUN: rocmlir-opt -rock-lower-reduce %s | FileCheck %s
// CHECK-DAG: #[[AMAP:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 3 + d1) * 64 + d2)>
// CHECK-DAG: #[[AMAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[AMAP2:.*]] = affine_map<(d0) -> (d0 floordiv 144, (d0 mod 144) floordiv 12, d0 mod 12)>
// CHECK-DAG: #[[MAP0:.*]] = #rock.transform_map<#[[AMAP]] by [<Unmerge{2, 3, 64} ["bid", "iter", "tid"] at [0, 1, 2] -> ["flatDim"] at [0]>] bounds = [2, 3, 64] -> [384]>
// CHECK-DAG: #[[MAP1:.*]] = #rock.transform_map<#[[AMAP1]] by [<Pad{0, 96} ["flatDim"] at [0] -> ["flatDim"] at [0]>] bounds = [384] -> [288]>
// CHECK-DAG: #[[MAP2:.*]] = #rock.transform_map<#[[AMAP2]] by [<Merge{2, 12, 12} ["flatDim"] at [0] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [288] -> [2, 12, 12]>

// CHECK: @test_reduce_sum
func.func @test_reduce_sum(%arg0: memref<2x12x12xf32>, %arg1: memref<2x12x1xf32>) attributes {kernel, mhal.arch = ""} {
    // CHECK-DAG: %[[bid:.*]] = rock.workgroup_id : index
    // CHECK-DAG: %[[tid:.*]] = rock.workitem_id : index
    // CHECK: rock.transforming_for {{.*}} (%[[loadCoord0:.*]], %[[loadCoord1:.*]], %[[loadCoord2:.*]]) = {{.*}}#[[MAP0]], #[[MAP1]], #[[MAP2]]](%[[bid]], %c0, %[[tid]]) (%[[valid:.*]]) = validity
    // CHECK: %[[ld:.*]] = rock.global_load %arg0[%[[loadCoord0]], %[[loadCoord1]], %[[loadCoord2]]]
    // CHECK: %[[ldRed:.*]] = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
    // CHECK: rock.in_bounds_store %[[ld]] -> %[[ldRed]][%c0] : f32 -> memref<1xf32, #gpu.address_space<private>>, index
    // CHECK: rock.global_store atomic_add %[[ldRed]][%c0] -> %arg1[%[[loadCoord0]], %[[loadCoord1]], %c0] if %[[valid]]
    rock.reduce sum %arg0 into %arg1 features = mfma|dot|atomic_add {axis = 2 : index, block_size = 64 : i32, grid_size = 2 : i32} : memref<2x12x12xf32> into memref<2x12x1xf32>
    func.return
}

// CHECK: @test_reduce_max
func.func @test_reduce_max(%arg0: memref<2x12x12xf32>, %arg1: memref<2x12x1xf32>) attributes {kernel, mhal.arch = ""} {
    // CHECK-DAG: %[[bid:.*]] = rock.workgroup_id : index
    // CHECK-DAG: %[[tid:.*]] = rock.workitem_id : index
    // CHECK: rock.transforming_for {{.*}} (%[[loadCoord0:.*]], %[[loadCoord1:.*]], %[[loadCoord2:.*]]) = {{.*}}#[[MAP0]], #[[MAP1]], #[[MAP2]]](%[[bid]], %c0, %[[tid]]) (%[[valid:.*]]) = validity
    // CHECK: %[[ld:.*]] = rock.global_load %arg0[%[[loadCoord0]], %[[loadCoord1]], %[[loadCoord2]]]
    // CHECK: %[[ldRed:.*]] = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
    // CHECK: rock.in_bounds_store %[[ld]] -> %[[ldRed]][%c0] : f32 -> memref<1xf32, #gpu.address_space<private>>, index
    // CHECK: rock.global_store atomic_max %[[ldRed]][%c0] -> %arg1[%[[loadCoord0]], %[[loadCoord1]], %c0] if %[[valid]]
    rock.reduce max %arg0 into %arg1 features = mfma|dot|atomic_add {axis = 2 : index, block_size = 64 : i32, grid_size = 2 : i32} : memref<2x12x12xf32> into memref<2x12x1xf32>
    func.return
}
