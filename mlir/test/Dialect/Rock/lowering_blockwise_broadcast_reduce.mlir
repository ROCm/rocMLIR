// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise -canonicalize -split-input-file %s | FileCheck %s

// CHECK-DAG: #[[MAP:.*]] =  affine_map<(d0) -> (d0, 0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1) -> (d0 * 20 + d1)>
// CHECK-DAG: #[[MAP6:.*]] = affine_map<(d0, d1, d2) -> (d0 + d1, d2)>
// CHECK-DAG: #[[MAP7:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-DAG: #[[TMAP:.*]] = #rock.transform_map<#[[MAP]]
// CHECK-DAG: #[[TMAP1:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP2:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP3:.*]] = #rock.transform_map<#[[MAP2]]
// CHECK-DAG: #[[TMAP4:.*]] = #rock.transform_map<#[[MAP3]]
// CHECK-DAG: #[[TMAP5:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP6:.*]] = #rock.transform_map<#[[MAP4]]
// CHECK-DAG: #[[TMAP7:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP8:.*]] = #rock.transform_map<#[[MAP5]]
// CHECK-DAG: #[[TMAP9:.*]] = #rock.transform_map<#[[MAP6]]
// CHECK-DAG: #[[TMAP10:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP11:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP12:.*]] = #rock.transform_map<#[[MAP5]]
// CHECK-DAG: #[[TMAP13:.*]] = #rock.transform_map<#[[MAP7]]

// CHECK: func @rock_blockwise_reducesum_nr_threads_gt_blocksize

// CHECK-DAG: %[[NEGINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[TID0:.*]] = rock.workitem_id : index

// init the threadwise reduction accumulator
// CHECK: memref.store %[[NEGINF]], %[[TO_REDUCE_ACC_MEMREF:.*]][{{.*}}]

// CHECK: rock.transforming_for {{.*}} (%[[NRDIM_THREAD:.*]], %[[RDIM_THREAD:.*]]) = [#[[TMAP]], #[[TMAP1]]](%[[ZERO]]), (%[[ITER_ARG:.*]]) = [](%[[ZERO]]) {{.*}} bounds [20] strides [1] {
    // CHECK: %[[LOAD_VAL:.*]] = rock.in_bounds_load %arg0[%[[ITER_ARG]]]
    // CHECK: %[[LOAD_ACC:.*]] = rock.in_bounds_load %[[TO_REDUCE_ACC_MEMREF]][%[[NRDIM_THREAD]]]
    // CHECK: %[[REDUCED:.*]] = arith.maxf %[[LOAD_ACC]], %[[LOAD_VAL]]
    // CHECK: rock.in_bounds_store %[[REDUCED]] -> %[[TO_REDUCE_ACC_MEMREF]][%[[NRDIM_THREAD]]]

// CHECK-DAG: %[[TID1:.*]] = rock.workitem_id : index
// CHECK: rock.transforming_for {{.*}} (%[[ITER_ARG:.*]]) = [#[[TMAP2]], #[[TMAP3]]](%[[ZERO]], %[[ZERO]]), (%[[NRDIM_THREAD:.*]], %[[RDIM_THREAD:.*]]) = [](%[[ZERO]], %[[ZERO]]) {{.*}} bounds [20, 1] strides [1, 1] {
    // CHECK-DAG: rock.transforming_for {{.*}} (%[[NRDIM_BLK:.*]], %[[RDIM_BLK:.*]]) = [#[[TMAP4]], #[[TMAP5]]](%[[TID1]], %[[ITER_ARG]]) {{.*}} bounds [1, 1] strides [1, 1] {
        // CHECK: %[[LOAD_ACC:.*]] = rock.in_bounds_load %[[TO_REDUCE_ACC_MEMREF]][%[[NRDIM_THREAD]]]
        // CHECK: rock.transforming_for {{.*}} (%{{.*}}, %[[RDIM_BLK_SLICE:.*]]) = [#[[TMAP6]], #[[TMAP7]]](%[[TID1]]) {{.*}} bounds [1] strides [1] {
            // CHECK: rock.transforming_for {{.*}} (%[[LDS_FLAT_COORDS:.*]]) = [#[[TMAP8]]](%[[NRDIM_BLK]], %[[RDIM_BLK_SLICE]]) {{.*}} bounds [1, 1] strides [1, 1] {
                // CHECK: rock.in_bounds_store %[[LOAD_ACC]] -> %arg2[%[[LDS_FLAT_COORDS]]]
            

// CHECK: rock.lds_barrier

// init the blockwise reduction accumulator
// CHECK: memref.store %[[NEGINF]], %[[TO_REDUCE_ACC_MEMREF:.*]][{{.*}}]

// CHECK: rock.transforming_for {{.*}} (%[[LD_COORD:.*]]) = [#[[TMAP9]], #[[TMAP10]], #[[TMAP11]], #[[TMAP5]], #[[TMAP12]]](%[[TID0]], %[[ZERO]], %[[ZERO]]), {{.*}}, (%[[LDS_ST_COORD:.*]]) = [#[[TMAP9]], #[[TMAP10]], #[[TMAP11]], #[[TMAP13]], #[[TMAP12]]](%[[TID0]], %[[ZERO]], %[[ZERO]]) {{.*}} bounds [1, 1, 20] strides [1, 1, 4] {
    // CHECK: %[[TO_REDUCE_VAL:.*]] = rock.in_bounds_load {{.*}}[%[[LD_COORD]]]
    // CHECK: %[[TO_REDUCE_ACC:.*]] = rock.in_bounds_load %[[TO_REDUCE_ACC_MEMREF]][%[[ZERO]]]
    // CHECK: %[[MAX_REDUCE:.*]] = vector.reduction <maxf>, %[[TO_REDUCE_VAL]] : vector<4xf32> into f32
    // CHECK: %[[ACC_NEW:.*]] = arith.maxf %[[TO_REDUCE_ACC]], %[[MAX_REDUCE]]
    // CHECK: rock.in_bounds_store %[[ACC_NEW]] -> %arg2[%[[LDS_ST_COORD]]]

// CHECK: rock.lds_barrier
// CHECK: rock.threadwise_read_into {{.*}}(%arg2) {{.*}} -> %arg1

#inputView = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [20, 20] -> [20, 20]>
#inputView_tid = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{1, 20} ["tid"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [20] -> [1, 20]>
#inputView_iter = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{20, 1} ["iter"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [20] -> [20, 1]>
func.func @rock_blockwise_reducesum_nr_threads_gt_blocksize(%input_reg : memref<20xf32, #gpu.address_space<private>>,  %output_reg : memref<20xf32, #gpu.address_space<private>>, %ws_lds : memref<400xf32, #gpu.address_space<workgroup>>) attributes{arch = "", block_size = 20 : i32, grid_size = 8 : i32, kernel} {
    rock.blockwise_broadcast_reduce max [#inputView][#inputView_tid][#inputView_iter]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 20 : i32, nrDimPerThread = 20 : index} : memref<20xf32, #gpu.address_space<private>> using memref<400xf32, #gpu.address_space<workgroup>> into memref<20xf32, #gpu.address_space<private>>%c1 = arith.constant 1.0 : f32
    return
}

// -----

// CHECK-DAG: #[[MAP:.*]]  = affine_map<(d0) -> (d0, 0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1) -> (d0 * 20 + d1)>
// CHECK-DAG: #[[MAP6:.*]] = affine_map<(d0, d1, d2) -> (d0, d1 * 4 + d2)>
// CHECK-DAG: #[[MAP7:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-DAG: #[[TMAP:.*]] = #rock.transform_map<#[[MAP]]
// CHECK-DAG: #[[TMAP1:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP2:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP3:.*]] = #rock.transform_map<#[[MAP2]]
// CHECK-DAG: #[[TMAP4:.*]] = #rock.transform_map<#[[MAP3]]
// CHECK-DAG: #[[TMAP5:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP6:.*]] = #rock.transform_map<#[[MAP4]]
// CHECK-DAG: #[[TMAP7:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP8:.*]] = #rock.transform_map<#[[MAP5]]
// CHECK-DAG: #[[TMAP9:.*]] = #rock.transform_map<#[[MAP6]]
// CHECK-DAG: #[[TMAP10:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP11:.*]] = #rock.transform_map<#[[MAP1]]
// CHECK-DAG: #[[TMAP12:.*]] = #rock.transform_map<#[[MAP5]]
// CHECK-DAG: #[[TMAP13:.*]] = #rock.transform_map<#[[MAP7]]

// CHECK: func @rock_blockwise_reducesum_nr_threads_lt_blocksize
// CHECK: %[[TID0:.*]] = rock.workitem_id : index

// Skipping LDS workspace loading checks as they are quite same as above

// CHECK: %[[PRT_THREAD_IDX:.*]] = arith.divsi %[[TID0]], %c4
// CHECK: %[[PRT_GROUP_IDX:.*]] = arith.remsi %[[TID0]], %c4
// CHECK: rock.transforming_for {{.*}} (%[[LDS_LD_COORD:.*]]) = [#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PRT_THREAD_IDX]], %c0) {{.*}} bounds [1, 1, 4] strides [1, 1, 4] {
// CHECK: %[[TO_REDUCE_VAL:.*]] = rock.in_bounds_load {{.*}}[%[[LDS_LD_COORD]]]
// CHECK: %[[TO_REDUCE_ACC:.*]] = rock.in_bounds_load {{.*}}[%c0]
// CHECK: %[[SUM_REDUCE:.*]] = vector.reduction <add>, %[[TO_REDUCE_VAL]] : vector<4xf32> into f32
// CHECK: %[[ACC_NEW:.*]] = arith.addf %[[TO_REDUCE_ACC]], %[[SUM_REDUCE]]
// CHECK: rock.in_bounds_store %[[ACC_NEW]] -> {{.*}}[%c0] {{.*}} #gpu.address_space<private>>
// CHECK: rock.transforming_for {{.*}}[#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PRT_THREAD_IDX]], %c0) {{.*}} bounds [1, 1, 1] strides [1, 1, 1] {
// CHECK: rock.in_bounds_load {{.*}} : memref<1xf32, #gpu.address_space<private>>, index -> f32
// CHECK: rock.in_bounds_store {{.*}} : f32 -> memref<20xf32, #gpu.address_space<workgroup>>, index
// CHECK: rock.lds_barrier

// Partial threadwise reductions done now...

// CHECK: %[[PLUS_FOUR_OFFSET:.*]] = arith.addi %[[PRT_THREAD_IDX]], %c4
// CHECK: %[[PLUS_FOUR_BCHECK:.*]] = arith.cmpi slt, %[[PLUS_FOUR_OFFSET]], %c5
// CHECK: scf.if %[[PLUS_FOUR_BCHECK]] {
    // CHECK: rock.transforming_for
    // CHECK-SAME: (%[[LDS_LD_COORD1A:.*]]) = [#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PRT_THREAD_IDX]], %c0)
    // CHECK-SAME: (%[[LDS_LD_COORD1B:.*]]) = [#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PLUS_FOUR_OFFSET]], %c0)
    // CHECK: rock.in_bounds_load {{.*}}[%[[LDS_LD_COORD1A]]]
    // CHECK: rock.in_bounds_load {{.*}}[%[[LDS_LD_COORD1B]]]
    // CHECK: arith.addf
    // CHECK: rock.in_bounds_store {{.*}}[%[[LDS_LD_COORD1A]]]
// CHECK: rock.lds_barrier

// CHECK: %[[PLUS_TWO_OFFSET:.*]] = arith.addi %[[PRT_THREAD_IDX]], %c2
// CHECK: %[[PLUS_TWO_BCHECK:.*]] = arith.cmpi slt, %[[PLUS_TWO_OFFSET]], %c4
// CHECK: scf.if %[[PLUS_TWO_BCHECK]] {
    // CHECK: rock.transforming_for
    // CHECK-SAME: (%[[LDS_LD_COORD1A:.*]]) = [#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PRT_THREAD_IDX]], %c0)
    // CHECK-SAME: (%[[LDS_LD_COORD1B:.*]]) = [#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PLUS_TWO_OFFSET]], %c0)
    // CHECK: rock.in_bounds_load {{.*}}[%[[LDS_LD_COORD1A]]]
    // CHECK: rock.in_bounds_load {{.*}}[%[[LDS_LD_COORD1B]]]
    // CHECK: arith.addf
    // CHECK: rock.in_bounds_store {{.*}}[%[[LDS_LD_COORD1A]]]
// CHECK: rock.lds_barrier

// CHECK: %[[PLUS_ONE_OFFSET:.*]] = arith.addi %[[PRT_THREAD_IDX]], %c1
// CHECK: %[[PLUS_ONE_BCHECK:.*]] = arith.cmpi slt, %[[PLUS_ONE_OFFSET]], %c2
// CHECK: scf.if %[[PLUS_ONE_BCHECK]] {
    // CHECK: rock.transforming_for
    // CHECK-SAME: (%[[LDS_LD_COORD1A:.*]]) = [#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PRT_THREAD_IDX]], %c0)
    // CHECK-SAME: (%[[LDS_LD_COORD1B:.*]]) = [#[[TMAP3]], #[[TMAP4]], #[[TMAP5]], #[[TMAP1]], #[[TMAP2]]](%[[PRT_GROUP_IDX]], %[[PLUS_ONE_OFFSET]], %c0)
    // CHECK: rock.in_bounds_load {{.*}}[%[[LDS_LD_COORD1A]]]
    // CHECK: rock.in_bounds_load {{.*}}[%[[LDS_LD_COORD1B]]]
    // CHECK: arith.addf
    // CHECK: rock.in_bounds_store {{.*}}[%[[LDS_LD_COORD1A]]]
// CHECK: rock.lds_barrier

// All reductions are done and stored for each point in joint non-reduction space.
// Read-back the reduced values to regs

// CHECK: rock.transforming_for
// CHECK-SAME: (%[[LDS_LD_COORD:.*]]) = [#[[TMAP]], #[[TMAP6]], #[[TMAP2]]](%[[TID0]], %c0)
// CHECK: rock.in_bounds_load %arg2[%[[LDS_LD_COORD]]] : memref<20xf32, #gpu.address_space<workgroup>>, index -> f32
// CHECK: rock.in_bounds_store {{.*}} : f32 -> memref<4xf32, #gpu.address_space<private>>, index

#inputView = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [20, 4] -> [4, 20]>
#inputView_tid = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{1, 20} ["tid"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [20] -> [1, 20]>
#inputView_iter = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{4, 1} ["iter"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [4] -> [4, 1]>
func.func @rock_blockwise_reducesum_nr_threads_lt_blocksize(%input_reg : memref<4xf32, #gpu.address_space<private>>,  %output_reg : memref<4xf32, #gpu.address_space<private>>, %ws_lds : memref<80xf32, #gpu.address_space<workgroup>>) attributes{arch = "", block_size = 20 : i32, grid_size = 8 : i32, kernel} {
  rock.blockwise_broadcast_reduce sum [#inputView][#inputView_tid][#inputView_iter]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 20 : i32, nrDimPerThread = 4 : index} : memref<4xf32, #gpu.address_space<private>> using memref<80xf32, #gpu.address_space<workgroup>> into memref<4xf32, #gpu.address_space<private>>
  return
}
