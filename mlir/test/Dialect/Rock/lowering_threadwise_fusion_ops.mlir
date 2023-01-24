// Note: this should be in a post-fusion pass
// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise %s | FileCheck --enable-var-scope %s

// CHECK-DAG: #[[$ON_OP:transform_map.+]] = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$IN_FUNC:transform_map.+]] = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2 - 2)>
#transform_map0 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  by [<PassThrough ["x", "y", "z"] at [0, 1, 2] -> ["x", "y", "z"] at [0, 1, 2]>]
  bounds = [2, 64, 32] -> [2, 64, 32]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2 - 2)>
  by [<PassThrough ["x", "y"] at [0, 1]  -> ["x", "y"] at [0, 1]>,
    <Pad{2, 0} ["z"] at [2] -> ["z"] at [2]>]
  bounds = [2, 64, 32] -> [2, 64, 30]>

// CHECK-LABEL: func @threadwise_read_into
// CHECK-SAME: [[source:%.+]]: memref<2x64x30xf32>, [[dest:%.+]]: memref<32xf32, 5>
func.func @threadwise_read_into( %source: memref<2x64x30xf32>, %dest: memref<32xf32, 5>) {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ({{%.*}}, {{%.*}}, [[i:%.+]]) = []([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 2]
  // CHECK: [[tmp:%.+]] = rock.buffer_load [[source]][[[args]]]
  // CHECK-SAME: leftOobDims = [2 : i32], rightOobDims = []
  // CHECK: [[tmp1:%.+]] = rock.insert_slice [[tmp]]
  // CHECK: rock.in_bounds_store [[tmp1]] -> [[dest]][[[i]]]

  %view = rock.transform %source by #transform_map1 : memref<2x64x30xf32> to memref<2x64x32xf32>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map0](%view) -> %dest
    : memref<2x64x32xf32> -> memref<32xf32, 5>
  func.return
}


// CHECK-LABEL: func @threadwise_write_all
// CHECK-SAME: [[source:%.+]]: memref<32xf32, 5>, [[dest:%.+]]: memref<2x64x30xf32>
func.func @threadwise_write_all(%source: memref<32xf32, 5>, %dest: memref<2x64x30xf32>) {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ({{%.*}}, {{%.*}}, [[i:%.+]]) = []([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 2]
  // CHECK: %[[ldval:.*]] = rock.in_bounds_load [[source]][[[i]]]
  // CHECK: rock.buffer_store  set %[[ldval]] -> [[dest]][[[args]]]
  // CHECK-SAME: leftOobDims = [2 : i32], rightOobDims = []

  %view = rock.transform %dest by #transform_map1 : memref<2x64x30xf32> to memref<2x64x32xf32>
  rock.threadwise_write_all {forceUnroll, useIndexDiffs}
    %source -> [#transform_map0](%view) by set
    : memref<32xf32, 5> -> memref<2x64x32xf32>
  func.return
}
