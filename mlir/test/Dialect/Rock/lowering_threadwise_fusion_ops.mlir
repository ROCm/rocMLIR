// Note: this should be in a post-fusion pass
// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise --canonicalize %s | FileCheck --enable-var-scope %s

// CHECK-DAG: #[[$ON_OP:transform_map]] = #rock.transform_map
// CHECK-DAG-SAME: PassThrough
// CHECK-DAG-SAME: [0, 1, 2]
// CHECK-DAG-SAME: [0, 1, 2]
#transform_map0 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  by [<PassThrough ["x", "y", "z"] at [0, 1, 2] -> ["x", "y", "z"] at [0, 1, 2]>]
  bounds = [2, 64, 32] -> [2, 64, 32]>
// CHECK-DAG: #[[$IN_FUNC:transform_map.+]] = #rock.transform_map
// CHECK-DAG-SAME: PassThrough
// CHECK-DAG-SAME: [0, 1]
// CHECK-DAG-SAME: [0, 1]
// CHECK-DAG-SAME: Pad{2, 0}
#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2 - 2)>
  by [<PassThrough ["x", "y"] at [0, 1]  -> ["x", "y"] at [0, 1]>,
    <Pad{2, 0} ["z"] at [2] -> ["z"] at [2]>]
  bounds = [2, 64, 32] -> [2, 64, 30]>

// CHECK-DAG: #[[$ON_OP_IDX:transform_map.+]] = #rock.transform_map
// CHECK-DAG-SAME: PassThrough
// CHECK-DAG-SAME: [0, 1, 2, 3]
// CHECK-DAG-SAME: [0, 1, 2, 3]
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  by [<PassThrough ["w", "x", "y", "z"] at [0, 1, 2, 3] -> ["w", "x", "y", "z"] at [0, 1, 2, 3]>]
  bounds = [3, 2, 64, 32] -> [3, 2, 64, 32]>
// CHECK-DAG: #[[$IN_FUNC_IDX:transform_map.+]] = #rock.transform_map
// CHECK-DAG-SAME: PassThrough
// CHECK-DAG-SAME: [0, 1, 2]
// CHECK-DAG-SAME: [0, 1, 2]
// CHECK-DAG-SAME: Pad{2, 0}
#transform_map3 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 - 2)>
  by [<PassThrough ["w", "x", "y"] at [0, 1, 2]  -> ["w", "x", "y"] at [0, 1, 2]>,
    <Pad{2, 0} ["z"] at [3] -> ["z"] at [3]>]
  bounds = [3, 2, 64, 32] -> [3, 2, 64, 30]>


// CHECK-LABEL: func @threadwise_read_into
// CHECK-SAME: [[source:%.+]]: memref<2x64x30xf32>, [[dest:%.+]]: memref<32xf32, #gpu.address_space<private>>
func.func @threadwise_read_into( %source: memref<2x64x30xf32>, %dest: memref<32xf32, #gpu.address_space<private>>) {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ({{%.*}}, {{%.*}}, [[i:%.+]]) = []([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ([[valid:%.+]], {{%.*}}) = validity
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 2]
  // CHECK-NEXT: [[tmp:%.+]] = rock.buffer_load [[source]][[[args]]] if [[valid]]
  // CHECK-NEXT: rock.in_bounds_store [[tmp]] -> [[dest]][[[i]]]

  %view = rock.transform %source by #transform_map1 : memref<2x64x30xf32> to memref<2x64x32xf32>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map0](%view)[%bid, %tid] -> %dest
    : memref<2x64x32xf32> -> memref<32xf32, #gpu.address_space<private>>
  func.return
}

// CHECK-LABEL: func @threadwise_read_into_extra_idx
// CHECK-SAME: [[source:%.+]]: memref<3x2x64x30xf32>, [[dest:%.+]]: memref<32xf32, #gpu.address_space<private>>
func.func @threadwise_read_into_extra_idx( %source: memref<3x2x64x30xf32>, %dest: memref<32xf32, #gpu.address_space<private>>) {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[extra_idx:%.+]] = arith.constant 1
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP_IDX]], #[[$IN_FUNC_IDX]]]([[extra_idx]], [[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ({{%.*}}, {{%.*}}, {{%.*}}, [[i:%.+]]) = []([[extra_idx]], [[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ([[valid:%.+]], {{%.*}}) = validity
  // CHECK-SAME: bounds [1, 1, 1, 32]
  // CHECK-SAME: strides [1, 1, 1, 2]
  // CHECK-NEXT: [[tmp:%.+]] = rock.buffer_load [[source]][[[args]]] if [[valid]]
  // CHECK-NEXT: rock.in_bounds_store [[tmp]] -> [[dest]][[[i]]]

  %view = rock.transform %source by #transform_map3 : memref<3x2x64x30xf32> to memref<3x2x64x32xf32>
  %extra_idx = arith.constant 1 : index
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map2](%view)[%extra_idx, %bid, %tid] -> %dest
    : memref<3x2x64x32xf32> -> memref<32xf32, #gpu.address_space<private>>
  func.return
}


// CHECK-LABEL: func @threadwise_write_all
// CHECK-SAME: [[source:%.+]]: memref<32xf32, #gpu.address_space<private>>, [[dest:%.+]]: memref<2x64x30xf32>
func.func @threadwise_write_all(%source: memref<32xf32, #gpu.address_space<private>>, %dest: memref<2x64x30xf32>) {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ({{%.*}}, {{%.*}}, [[i:%.+]]) = []([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ({{%.*}}, [[valid:%.+]]) = validity
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 2]
  // CHECK-NEXT: [[tmp:%.+]] = rock.in_bounds_load [[source]][[[i]]]
  // CHECK-NEXT: rock.buffer_store set [[tmp]] -> [[dest]][[[args]]] if [[valid]]

  %view = rock.transform %dest by #transform_map1 : memref<2x64x30xf32> to memref<2x64x32xf32>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_write_all features = dot {forceUnroll, useIndexDiffs}
    %source -> [#transform_map0](%view)[%bid, %tid] by set
    : memref<32xf32, #gpu.address_space<private>> -> memref<2x64x32xf32>
  func.return
}

// CHECK-LABEL: func @threadwise_write_all_extra_idx
// CHECK-SAME: [[source:%.+]]: memref<32xf32, #gpu.address_space<private>>, [[dest:%.+]]: memref<3x2x64x30xf32>
func.func @threadwise_write_all_extra_idx(%source: memref<32xf32, #gpu.address_space<private>>, %dest: memref<3x2x64x30xf32>) {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[extra_idx:%.+]] = arith.constant 2
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ({{%.*}}, {{%.*}}, {{%.*}}, [[i:%.+]]) = []([[extra_idx]], [[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP_IDX]], #[[$IN_FUNC_IDX]]]([[extra_idx]], [[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ({{%.*}}, [[valid:%.+]]) = validity
  // CHECK-SAME: bounds [1, 1, 1, 32]
  // CHECK-SAME: strides [1, 1, 1, 2]
  // CHECK-NEXT: [[tmp:%.+]] = rock.in_bounds_load [[source]][[[i]]]
  // CHECK-NEXT: rock.buffer_store set [[tmp]] -> [[dest]][[[args]]] if [[valid]]

  %view = rock.transform %dest by #transform_map3 : memref<3x2x64x30xf32> to memref<3x2x64x32xf32>
  %extra_idx = arith.constant 2 : index
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_write_all features = dot {forceUnroll, useIndexDiffs}
    %source -> [#transform_map2](%view)[%extra_idx, %bid, %tid] by set
    : memref<32xf32, #gpu.address_space<private>> -> memref<3x2x64x32xf32>
  func.return
}
