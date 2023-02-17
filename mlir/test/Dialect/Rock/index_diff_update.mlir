// RUN: rocmlir-opt -rock-sugar-to-loops %s | FileCheck %s

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)>
    by [#rock.transform<PassThrough ["x", "y"] at [0, 1] -> ["x", "y"] at [1, 0]>]
    bounds = [64, 128] -> [128, 64]>

#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1 - 1, d2 + 32)>
    by [#rock.transform<PassThrough ["a"] at [0] -> ["a"] at [0]>,
        #rock.transform<Pad{1, 1} ["b"] at [1] -> ["b"] at [1]>,
        #rock.transform<Slice{32, 64} ["c"] at [2] -> ["c"] at [2]>]
    bounds = [64, 66, 32] -> [64, 64, 64]>

#transform_map2 = #rock.transform_map<affine_map<(d0, d1) -> (d0 + d1)>
    by [#rock.transform<Embed{1, 1} ["x", "y"] at [0, 1] -> ["r"] at [0]>]
    bounds = [64, 3] -> [65]>

#transform_map3 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + 4 * d0)>
    by [#rock.transform<Unmerge{16, 4} ["x", "y"] at [0, 1] -> ["r"] at [0]>]
    bounds = [16, 4] -> [64]>

#transform_map4 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
    by [#rock.transform<Merge{16, 4} ["r"] at [0] -> ["x", "y"] at [0, 1]>]
    bounds = [64] -> [16, 4]>

#transform_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1)>
    by [#rock.transform<AddDim{16} ["x"] at [0] -> [] at []>,
        #rock.transform<PassThrough ["y"] at [1] -> ["y"] at [0]>]
    bounds = [16, 4] -> [4]>

#transform_map6 = #rock.transform_map<affine_map<(d0) -> (d0, 1)>
    by [<PassThrough ["x"] at [0] -> ["x"] at [0]>,
        <ConstDim{1, 8} [] at [] -> ["y"] at [1]>]
    bounds = [64] -> [64, 8]>

module {
    // CHECK-LABEL: func.func @index_diff_passthrough
    // CHECK-SAME: ({{.*}}%[[l0:.*]]: index, %[[l1:.*]]: index)
    func.func @index_diff_passthrough(%mem: memref<128x64xf32>, %v: f32,
            %l0: index, %l1: index) {
        // CHECK-DAG: %[[dx:.*]] = arith.constant 1
        // CHECK-DAG: %[[dy:.*]] = arith.constant 2
        // CHECK-DAG: %[[i0:.*]] = arith.addi %[[l0]], %[[dy]]
        // CHECK-DAG: %[[i1:.*]] = arith.addi %[[l1]], %[[dx]]
        %dx = arith.constant 1 : index
        %dy = arith.constant 2 : index
        %i0, %i1, %d0, %d1 =
            rock.index_diff_update #transform_map0(%dx, %dy) + (%l0, %l1) : index, index
        // CHECK: memref.store {{.*}}%[[i0]], %[[i1]]
        memref.store %v, %mem[%i0, %i1] : memref<128x64xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_no_dim_change
    // CHECK-SAME: ({{.*}}, %[[l0:.*]]: index, %[[l1:.*]]: index, %[[l2:.*]]: index, %[[db:.*]]: index)
    func.func @index_diff_no_dim_change(%mem: memref<64x64x64xf32>, %v: f32,
            %l0: index, %l1: index, %l2: index, %db: index) {
        // CHECK-DAG: %[[dc:.*]] = arith.constant 1
        // CHECK-DAG: %[[i1:.*]] = arith.addi %[[l1]], %[[db]]
        // CHECK-DAG: %[[i2:.*]] = arith.addi %[[l2]], %[[dc]]
        %da = arith.constant 0 : index
        %dc = arith.constant 1 : index
        %i0, %i1, %i2, %d0, %d1, %d2 = rock.index_diff_update #transform_map1(%da, %db, %dc) + (%l0, %l1, %l2) : index, index, index
        // CHECK: memref.store {{.*}}%[[l0]], %[[i1]], %[[i2]]
        memref.store %v, %mem[%i0, %i1, %i2] : memref<64x64x64xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_embed
    // CHECK-SAME: ({{.*}}%[[l0:.*]]: index)
    func.func @index_diff_embed(%mem: memref<65xf32>, %v: f32, %l0: index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        // CHECK-DAG: %[[c3:.*]] = arith.constant 3
        // CHECK-DAG: %[[i0:.*]] = arith.addi %[[l0]], %[[c3]]
        %i0, %d0 = rock.index_diff_update #transform_map2(%c1, %c2) + (%l0) : index
        // CHECK: memref.store {{.*}}%[[i0]]
        memref.store %v, %mem[%i0] : memref<65xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_unmerge
    // CHECK-SAME: ({{.*}}%[[l0:.*]]: index)
    func.func @index_diff_unmerge(%mem: memref<64xf32>, %v: f32, %l0: index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        // CHECK-DAG: %[[c6:.*]] = arith.constant 6
        // CHECK-DAG: %[[i0:.*]] = arith.addi %[[l0]], %[[c6]]
        %i0, %d0 = rock.index_diff_update #transform_map3(%c1, %c2) + (%l0) : index
        // CHECK: memref.store {{.*}}%[[i0]]
        memref.store %v, %mem[%i0] : memref<64xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_overflow_checks
    // CHECK-SAME: ({{.*}}, %[[l0:.*]]: index, %[[l1:.*]]: index)
    func.func @index_diff_overflow_checks(%mem: memref<16x4xf32>, %v: f32, %l0: index, %l1: index) {
        %c1 = arith.constant 1 : index
        // CHECK-DAG: %[[c1:.*]] = arith.constant 1
        // CHECK-DAG: %[[c4:.*]] = arith.constant 4
        // CHECK-DAG: %[[i1_raw:.*]] = arith.addi %[[l1]], %[[c1]]
        // CHECK-DAG: %[[i1:.*]] = arith.remui %[[i1_raw]], %[[c4]]
        // CHECK-DAG: %[[i1_overflow:.*]] = arith.divui %[[i1_raw]], %[[c4]]
        // CHECK-DAG: %[[i0:.*]] = arith.addi %[[l0]], %[[i1_overflow]]
        %i0, %i1, %d0, %d1 = rock.index_diff_update #transform_map4(%c1) + (%l0, %l1) : index, index
        // CHECK: memref.store {{.*}}%[[i0]], %[[i1]]
        memref.store %v, %mem[%i0, %i1] : memref<16x4xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_elide_overflow
    // CHECK-SAME: ({{.*}}, %[[l0:.*]]: index, %[[l1:.*]]: index)
    func.func @index_diff_elide_overflow(%mem: memref<16x4xf32>, %v: f32, %l0: index, %l1: index) {
        %c4 = arith.constant 4 : index
        // CHECK-DAG: %[[c1:.*]] = arith.constant 1
        // CHECK-DAG: %[[i0:.*]] = arith.addi %[[l0]], %[[c1]]
        %i0, %i1, %d0, %d1 = rock.index_diff_update #transform_map4(%c4) + (%l0, %l1) : index, index
        // CHECK: memref.store {{.*}}%[[i0]], %[[l1]]
        memref.store %v, %mem[%i0, %i1] : memref<16x4xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_add_dim
    // CHECK-SAME: ({{.*}}, %[[l0:.*]]: index, %[[dx:.*]]: index)
    func.func @index_diff_add_dim(%mem: memref<4xf32>, %v: f32, %l0: index, %dx: index) {
        // CHECK-NEXT: %[[c1:.*]] = arith.constant 1
        %c1 = arith.constant 1 : index
        // CHECK-NEXT: %[[i0:.*]] = arith.addi %[[l0]], %[[c1]]
        %i0, %d0 = rock.index_diff_update #transform_map5(%dx, %c1) + (%l0) : index
        // CHECK-NEXT: memref.store {{.*}}%[[i0]]
        memref.store %v, %mem[%i0] : memref<4xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_const_dim
    // CHECK-SAME: ({{.*}}, %[[l0:.*]]: index, %[[l1:.*]]: index, %[[dx:.*]]: index)
    func.func @index_diff_const_dim(%mem: memref<64x8xf32>, %v: f32, %l0: index, %l1: index, %dx: index) {
        // CHECK-NEXT: %[[i0:.*]] = arith.addi %[[l0]], %[[dx]]
        %i0, %i1, %d0, %d1 = rock.index_diff_update #transform_map6(%dx) + (%l0, %l1) : index, index
        // CHECK-NEXT: memref.store {{.*}}%[[i0]], %[[l1]]
        memref.store %v, %mem[%i0, %i1] : memref<64x8xf32>
        return
    }

    // CHECK-LABEL: func.func @chain_diffs
    // CHECK-SAME: ({{.*}}, %[[int0:.*]]: index, %[[int1:.*]]: index, %[[l0:.*]]: index)
    func.func @chain_diffs(%mem: memref<64xf32>, %v: f32, %int0: index, %int1: index, %l0: index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %int0_upd, %int1_upd, %dint0, %dint1 = rock.index_diff_update #transform_map0(%c1, %c2) + (%int0, %int1) : index, index
        // CHECK-DAG: %[[c9:.*]] = arith.constant 9
        // CHECK-DAG: %[[i0:.*]] = arith.addi %[[l0]], %[[c9]]
        %i0, %d0 = rock.index_diff_update #transform_map3(%dint0, %dint1) + (%l0) : index
        // CHECK: memref.store {{.*}}%[[i0]]
        memref.store %v, %mem[%i0] : memref<64xf32>
        return
    }

    // CHECK-LABEL: func.func @index_diff_broadcast
    // CHECK-SAME: ({{.*}}, %[[int0:.*]]: index, %[[int1:.*]]: index, %[[l0:.*]]: index)
    func.func @index_diff_broadcast(%mem: memref<1x64xf32>, %v: f32, %int0: index, %int1: index, %l0: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %int0_upd, %int1_upd, %dint0, %dint1 = rock.index_diff_update <affine_map<(d0, d1) -> (0, d1)> by [
          #rock.transform<PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, #rock.transform<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>
        ] bounds = [1, 64] -> [1, 64]> (%c0, %c2) + (%c1, %c3) : index, index
        // CHECK-DAG: %[[c0:.*]] = arith.constant 0
        // CHECK-DAG: %[[c5:.*]] = arith.constant 5
        // CHECK: memref.store {{.*}}, {{.*}}[%[[c0]], %[[c5]]]
        memref.store %v, %mem[%int0_upd, %int1_upd] : memref<1x64xf32>
        return
    }
}
