// RUN: rocmlir-opt --rock-threadwise-gemm-lowering %s | FileCheck %s

// CHECK-LABEL: func @write_to_lds_gated_waves
func.func @write_to_lds_gated_waves(%source: memref<32xf32, #gpu.address_space<private>>, %dest: memref<2x64x30xf32>) {
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK-DAG: [[zero1:%.+]] = arith.constant 0
  // CHECK-DAG: [[zero2:%.+]] = arith.constant 0
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ({{%.*}}, {{%.*}}) = []([[tid]], [[zero2]])
  // CHECK-SAME: , ({{%.*}}) = [{{#.*}}, {{#.*}}, {{#.*}}, {{#.*}}, {{#.*}}, {{#.*}}]([[tid]], [[zero2]])
  // CHECK-SAME: ({{%.*}}, [[valid:%.+]]) = validity
  // CHECK-SAME: bounds [1, 1]
  // CHECK-SAME: strides [1, 1]
  // CHECK-NEXT: scf.if [[valid]]
  // CHECK-NEXT: [[load:%.+]] = rock.in_bounds_load
  // CHECK-NEXT: rock.in_bounds_store [[load]] ->
  %wid = rock.workitem_id : index
  %reg = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
  
  %c0 = arith.constant 0 : index
  %0 = rock.alloc() : memref<512xi8, #gpu.address_space<workgroup>>
  %view = memref.view %0[%c0][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<512xi8, #gpu.address_space<workgroup>>
  %view_4 = memref.view %view[%c0][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<512xi8, #gpu.address_space<workgroup>>
  %view_7 = memref.view %view_4[%c0][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
  %28 = rock.transform %view_7 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 + d2)> by [<Unmerge{2, 64, 1} ["k_outer", "m", "kpack_idx"] at [0, 1, 2] -> ["raw"] at [0]>, <AddDim{1} ["kpack_vec"] at [3] -> [] at []>] bounds = [2, 64, 1, 1] -> [128]> : memref<128xf32, #gpu.address_space<workgroup>> to memref<2x64x1x1xf32, #gpu.address_space<workgroup>>
  %29 = rock.transform %28 by <affine_map<(d0, d1) -> (d0, d1, 0, 0)> by [<Merge{2, 1, 1} ["k"] at [0] -> ["k_outer", "kpack_idx", "kpack_vec"] at [0, 2, 3]>, <Merge{64} ["d"] at [1] -> ["m"] at [1]>] bounds = [2, 64] -> [2, 64, 1, 1]> : memref<2x64x1x1xf32, #gpu.address_space<workgroup>> to memref<2x64xf32, #gpu.address_space<workgroup>>
  %30 = rock.transform %29 by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m"] at [1] -> ["m"] at [1]>, <Pad{0, 0} ["k"] at [0] -> ["k"] at [0]>] bounds = [2, 64] -> [2, 64]> : memref<2x64xf32, #gpu.address_space<workgroup>> to memref<2x64xf32, #gpu.address_space<workgroup>>
  %31 = rock.transform %30 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 + d2 + d4, d3 * 64 + d1)> by [<Unmerge{2, 1, 1} ["k_thread", "kouterPerThread", "kpackPerThread"] at [0, 2, 4] -> ["k"] at [0]>, <Unmerge{1, 64} ["m_iter", "m_thread"] at [3, 1] -> ["m"] at [1]>] bounds = [2, 64, 1, 1, 1] -> [2, 64]> : memref<2x64xf32, #gpu.address_space<workgroup>> to memref<2x64x1x1x1xf32, #gpu.address_space<workgroup>>
  %32 = rock.transform %31 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by [<PassThrough ["m_thread", "kouterPerThread", "m_iter", "kpackPerThread"] at [1, 2, 3, 4] -> ["m_thread", "kouterPerThread", "m_iter", "kpackPerThread"] at [1, 2, 3, 4]>, <Pad{0, 2} ["k_thread"] at [0] -> ["k_thread"] at [0]>] bounds = [4, 64, 1, 1, 1] -> [2, 64, 1, 1, 1]> : memref<2x64x1x1x1xf32, #gpu.address_space<workgroup>> to memref<4x64x1x1x1xf32, #gpu.address_space<workgroup>>
  %33 = rock.transform %32 by <affine_map<(d0, d1) -> (d0 floordiv 64, d0 mod 64, 0, 0, 0)> by [<Merge{4, 64} ["tid"] at [0] -> ["k_thread", "m_thread"] at [0, 1]>, <Merge{1, 1, 1} ["iter"] at [1] -> ["kouterPerThread", "m_iter", "kpackPerThread"] at [2, 3, 4]>] bounds = [256, 1] -> [4, 64, 1, 1, 1]> : memref<4x64x1x1x1xf32, #gpu.address_space<workgroup>> to memref<256x1xf32, #gpu.address_space<workgroup>>

  rock.threadwise_write_all features =  mfma|dot|atomic_add {forceUnroll, useIndexDiffs} %reg -> [](%33) [%wid] by  set : memref<1xf32, #gpu.address_space<private>> -> memref<256x1xf32, #gpu.address_space<workgroup>>

  func.return
}