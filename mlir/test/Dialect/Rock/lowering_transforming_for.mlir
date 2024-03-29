// RUN: rocmlir-opt --rock-sugar-to-loops %s | FileCheck %s

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + 4 * d0)>
    by [<Unmerge{16, 4} ["1", "0"] at [0, 1] -> ["r"] at [0]>]
    bounds = [16, 4] -> [64]>

#transform_map1 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)>
    by [<PassThrough ["1", "0"] at [1, 0] -> ["1", "0"] at [0, 1]>]
    bounds = [4, 16] -> [16, 4]>


#transform_map_pad = #rock.transform_map<affine_map<(d0) -> (d0)>
    by [<Pad{0, 8} ["pad"] at [0] -> ["raw"] at [0]>]
    bounds = [16] -> [8]>

module {
// CHECK-LABEL: func.func @no_transform_to_affine
func.func @no_transform_to_affine() {
    %c0 = arith.constant 0 : index
    // CHECK: %[[true:.*]] = arith.constant true
    // CHECK: affine.for %[[arg0:.*]] = {{.*}}to 2
    // CHECK: affine.for %[[arg1:.*]] = {{.*}}to 3
    // CHECK: gpu.printf "%d, %d, %d" %[[arg0]], %[[arg1]], %[[true]]
    rock.transforming_for (%arg0, %arg1) = [](%c0, %c0) (%arg2) = validity bounds [2, 3] strides [1, 1] {
        gpu.printf "%d, %d, %d" %arg0, %arg1, %arg2 : index, index, i1
    }
    return
}

// CHECK-LABEL: func.func @no_transform_to_affine_strided
func.func @no_transform_to_affine_strided() {
    %c0 = arith.constant 0 : index
    // CHECK: %[[true:.*]] = arith.constant true
    // CHECK: affine.for %[[arg0:.*]] = {{.*}}to 2 step 2
    // CHECK: affine.for %[[arg1:.*]] = {{.*}}to 3
    // CHECK: gpu.printf "%d, %d, %d" %[[arg0]], %[[arg1]], %[[true]]
    rock.transforming_for (%arg0, %arg1) = [](%c0, %c0) (%arg2) = validity bounds [2, 3] strides [2, 1] {
        gpu.printf "%d, %d, %d" %arg0, %arg1, %arg2 : index, index, i1
    }
    return
}


// CHECK-LABEL: func.func @no_transform_unrolled
func.func @no_transform_unrolled() {
    %c0 = arith.constant 0 : index
    // CHECK-NOT: affine.for
    // CHECK-COUNT-6: gpu.printf
    rock.transforming_for {forceUnroll} (%arg0, %arg1) = [](%c0, %c0) (%arg2) = validity bounds [2, 3] strides [1, 1] {
        gpu.printf "%d, %d, %d" %arg0, %arg1, %arg2 : index, index, i1
    }
    return
}

// CHECK-LABEL: func.func @no_transform_unrolled_strided
func.func @no_transform_unrolled_strided() {
    %c0 = arith.constant 0 : index
    // CHECK-NOT: affine.for
    // CHECK-COUNT-3: gpu.printf
    rock.transforming_for {forceUnroll} (%arg0, %arg1) = [](%c0, %c0) (%arg2) = validity bounds [2, 3] strides [2, 1] {
        gpu.printf "%d, %d, %d" %arg0, %arg1, %arg2 : index, index, i1
    }
    return
}


// CHECK-LABEL: func.func @one_transform
// CHECK-SAME:(%[[arg0:.*]]: index, %[[arg1:.*]]: index)
func.func @one_transform(%arg0: index, %arg1: index) {
    // CHECK: %[[true:.*]] = arith.constant true
    // CHECK: affine.for %[[d0:.*]] = 0 to 2
    // CHECK: %[[u0:.*]] = arith.addi %[[arg0]], %[[d0]]
    // CHECK: %[[cmp0:.*]] = arith.muli %[[u0]]
    // CHECK: affine.for %[[d1:.*]] = 0 to 3
    // CHECK: %[[u1:.*]] = arith.addi %[[arg1]], %[[d1]]
    // CHECK-NEXT: %[[l0:.*]] = arith.addi %[[u1]], %[[cmp0]]
    // CHECK-NEXT: gpu.printf "%d, %d" %[[l0]], %[[true]]
    rock.transforming_for (%arg2) = [#transform_map0](%arg0, %arg1) (%arg3) = validity bounds [2, 3] strides [1, 1] {
        gpu.printf "%d, %d" %arg2, %arg3 : index, i1
    }
    return
}

// CHECK-LABEL: func.func @one_transform_index_diff
// CHECK-SAME:(%[[arg0:.*]]: index, %[[arg1:.*]]: index)
func.func @one_transform_index_diff(%arg0: index, %arg1: index) {
    // CHECK: %[[true:.*]] = arith.constant true
    // CHECK: %[[linit_cmp:.*]] = arith.muli %[[arg0]]
    // CHECK: %[[linit:.*]] = arith.addi %[[arg1]], %[[linit_cmp]]
    // CHECK: affine.for %[[d0:.*]] = 0 to 2
    // CHECK-NEXT: affine.for %[[d1:.*]] = 0 to 3
    // CHECK-NEXT: %[[c0:.*]] = arith.muli %[[d0]]
    // CHECK-NEXT: %[[c1:.*]] = arith.addi %[[d1]], %[[c0]]
    // CHECK-NEXT: %[[l0:.*]] = arith.addi %[[linit]], %[[c1]]
    // CHECK-NEXT: gpu.printf "%d, %d" %[[l0]], %[[true]]
    rock.transforming_for {useIndexDiffs} (%arg2) = [#transform_map0](%arg0, %arg1) (%arg3) = validity bounds [2, 3] strides [1, 1] {
        gpu.printf "%d, %d" %arg2, %arg3 : index, i1
    }
    return
}

// CHECK-LABEL: func.func @one_transform_unroll
func.func @one_transform_unroll(%arg0: index, %arg1: index) {
    // CHECK-NOT: affine.for
    // CHECK-COUNT-2: arith.muli
    // Three printf after the second iteration of outer loop
    // CHECK-COUNT-3: gpu.printf
    rock.transforming_for {forceUnroll} (%arg2) = [#transform_map0](%arg0, %arg1) (%arg3) = validity bounds [2, 3] strides [1, 1] {
        gpu.printf "%d" %arg2 : index
    }
    return
}

// CHECK-LABEL: func.func @one_transform_index_diff_unroll
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index)
func.func @one_transform_index_diff_unroll(%arg0: index, %arg1: index) {
    // CHECK-NOT: affine.for
    // CHECK-DAG: %[[true:.*]] = arith.constant true
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4
    // CHECK-DAG: %[[c5:.*]] = arith.constant 5
    // CHECK-DAG: %[[c6:.*]] = arith.constant 6
    // CHECK: %[[l0_cmp:.*]] = arith.muli %[[arg0]]
    // CHECK: %[[l0:.*]] = arith.addi %[[arg1]], %[[l0_cmp]]
    // CHECK: gpu.printf "%d, %d" %[[l0]], %[[true]]
    // CHECK: %[[l1:.*]] = arith.addi %[[l0]], %[[c1]]
    // CHECK-NEXT: gpu.printf "%d, %d" %[[l1]], %[[true]]
    // CHECK: %[[l2:.*]] = arith.addi %[[l0]], %[[c2]]
    // CHECK-NEXT: gpu.printf "%d, %d" %[[l2]], %[[true]]
    // CHECK: %[[l3:.*]] = arith.addi %[[l0]], %[[c4]]
    // CHECK-NEXT: gpu.printf "%d, %d" %[[l3]], %[[true]]
    // CHECK: %[[l4:.*]] = arith.addi %[[l0]], %[[c5]]
    // CHECK-NEXT: gpu.printf "%d, %d" %[[l4]], %[[true]]
    // CHECK: %[[l5:.*]] = arith.addi %[[l0]], %[[c6]]
    // CHECK-NEXT: gpu.printf "%d, %d" %[[l5]], %[[true]]
    rock.transforming_for {forceUnroll, useIndexDiffs} (%arg2) = [#transform_map0](%arg0, %arg1) (%arg3) = validity bounds [2, 3] strides [1, 1] {
        gpu.printf "%d, %d" %arg2, %arg3 : index, i1
    }
    return
}

// CHECK-LABEL: func.func @deep_transforms
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index)
func.func @deep_transforms(%arg0: index, %arg1: index) {
    rock.transforming_for (%arg2) = [#transform_map1, #transform_map0](%arg0, %arg1) (%arg3) = validity bounds [2, 3] strides [1, 1] {
        // CHECK: %[[shft2:.*]] = arith.addi %[[arg1]]
        // CHECK: %[[l0_int:.*]] = arith.muli %[[shft2]]
        // CHECK-NEXT: %[[l0:.*]] = arith.addi {{.*}}, %[[l0_int]]
        // CHECK-NEXT: gpu.printf "%d" %[[l0]]
        gpu.printf "%d" %arg2 : index
    }
    return
}

// CHECK-LABEL: func.func @deep_transforms_index_diff
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index)
func.func @deep_transforms_index_diff(%arg0: index, %arg1: index) {
    // CHECK: %[[init0:.*]] = arith.muli %[[arg1]]
    // CHECK: %[[init1:.*]] = arith.addi %[[arg0]], %[[init0]]
    // CHECK: affine.for %[[d0:.*]] = 0 to 2
    // CHECK-NEXT: affine.for %[[d1:.*]] = 0 to 3
    rock.transforming_for {useIndexDiffs} (%arg2) = [#transform_map1, #transform_map0](%arg0, %arg1) (%arg3) = validity bounds [2, 3] strides [1, 1] {
        // CHECK-DAG: %[[c0:.*]] = arith.muli %[[d1]]
        // CHECK-DAG: %[[c1:.*]] = arith.addi %[[d0]], %[[c0]]
        // CHECK-DAG: %[[l0:.*]] = arith.addi %[[init1]], %[[c1]]
        // CHECK-NEXT: gpu.printf "%d" %[[l0]]
        gpu.printf "%d" %arg2 : index
    }
    return
}

// CHECK-LABEL: func.func @multi_iteration
func.func @multi_iteration() {
    %c0 = arith.constant 0 : index
    // CHECK-COUNT-6: gpu.printf
    rock.transforming_for {forceUnroll} (%arg0, %arg1) = [#transform_map1](%c0, %c0), (%arg2) = [#transform_map0](%c0, %c0) (%arg3, %arg4) = validity bounds [2, 3] strides [1, 1] {
        gpu.printf "%d, %d, %d, %d, %d" %arg0, %arg1, %arg2, %arg3, %arg4 : index, index, index, i1, i1
    }
    return
}

// CHECK-LABEL: func.func @loop_result
func.func @loop_result(%arg0: index, %arg1: index) -> index {
    // CHECK: %[[c0:.*]] = arith.constant 0
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = affine.for {{.*}} iter_args(%[[oarg:.*]] = %[[c0]]
    // CHECK: %[[inner:.*]] = affine.for {{.*}} iter_args(%[[iarg:.*]] = %[[oarg]]
    %ret = rock.transforming_for (%arg2) = [#transform_map0](%arg0, %arg1)
            (%arg3) = validity
            iter_args(%arg4 = %c0) -> (index) bounds [2, 3] strides [1, 1] {
        // CHECK: %[[iret:.*]] = arith.addi %[[iarg]]
        %i = arith.addi %arg4, %arg2 : index
        // CHECK: affine.yield %[[iret]]
        rock.yield %i : index
        // CHECK: affine.yield %[[inner]]
    }
    // CHECK: return %[[ret]]
    return %ret : index
}

// CHECK-LABEL: func.func @bounds_check_pad
// CHECK-DAG: %[[c8:.*]] = arith.constant 8
// CHECK: affine.for %[[num:.*]] = {{.*}}to 16
// CHECK: %[[valid:.*]] = arith.cmpi ult, %[[num]], %[[c8]]
// CHECK: gpu.printf "%d" %[[valid]]
func.func @bounds_check_pad() {
    %c0 = arith.constant 0 : index
    rock.transforming_for (%arg0) = [#transform_map_pad](%c0) (%arg1) = validity bounds [16] strides [1] {
        gpu.printf "%d" %arg1 : i1
    }
    return
}
}
