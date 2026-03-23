// Note: this should be in a post-fusion pass
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering --canonicalize | FileCheck --enable-var-scope %s

// CHECK-DAG: #[[$ON_OP:transform_map[0-9]*]] = #rock.transform_map{{.*}}PassThrough{{.*}}[0, 1, 2]{{.*}}[0, 1, 2]
#transform_map0 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  by [<PassThrough ["1", "0", "z"] at [0, 1, 2] -> ["1", "0", "z"] at [0, 1, 2]>]
  bounds = [2, 64, 32] -> [2, 64, 32]>
// CHECK-DAG: #[[$IN_FUNC:transform_map[0-9]*]] = #rock.transform_map{{.*}}PassThrough{{.*}}[0, 1]{{.*}}[0, 1]{{.*}}Pad{2, 0}
#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2 - 2)>
  by [<PassThrough ["1", "0"] at [0, 1]  -> ["1", "0"] at [0, 1]>,
    <Pad{2, 0} ["z"] at [2] -> ["z"] at [2]>]
  bounds = [2, 64, 32] -> [2, 64, 30]>

// CHECK-DAG: #[[$ON_OP_IDX:transform_map[0-9]*]] = #rock.transform_map{{.*}}PassThrough{{.*}}[0, 1, 2, 3]{{.*}}[0, 1, 2, 3]
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  by [<PassThrough ["1", "1", "0", "z"] at [0, 1, 2, 3] -> ["1", "1", "0", "z"] at [0, 1, 2, 3]>]
  bounds = [3, 2, 64, 32] -> [3, 2, 64, 32]>
// CHECK-DAG: #[[$IN_FUNC_IDX:transform_map[0-9]*]] = #rock.transform_map{{.*}}PassThrough{{.*}}[0, 1, 2]{{.*}}[0, 1, 2]{{.*}}Pad{2, 0}
#transform_map3 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 - 2)>
  by [<PassThrough ["1", "1", "0"] at [0, 1, 2]  -> ["1", "1", "0"] at [0, 1, 2]>,
    <Pad{2, 0} ["z"] at [3] -> ["z"] at [3]>]
  bounds = [3, 2, 64, 32] -> [3, 2, 64, 30]>

// CHECK-LABEL: func @threadwise_prefetch
// CHECK-SAME: [[source:%.+]]: memref<2x64x30xf32>
func.func @threadwise_prefetch(%source: memref<2x64x30xf32>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 1]
  // CHECK-NEXT: rock.global_prefetch [[source]][[[args]]]

  %view = rock.transform %source by #transform_map1 : memref<2x64x30xf32> to memref<2x64x32xf32>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_prefetch {forceUnroll, useIndexDiffs}
    [#transform_map0](%view)[%bid, %tid] : memref<2x64x32xf32>
  func.return
}

// CHECK-LABEL: func @threadwise_prefetch_scalar
// CHECK-SAME: [[source:%.+]]: memref<f32>
func.func @threadwise_prefetch_scalar(%source: memref<f32>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: () = [#transform_map{{[0-9]*}}]([[zero]])
  // CHECK-SAME: bounds [1]
  // CHECK-SAME: strides [1]
  // CHECK-NEXT: rock.global_prefetch [[source]][]
  rock.threadwise_prefetch {forceUnroll, useIndexDiffs}
    [](%source)[]
    : memref<f32>
  func.return
}


// CHECK-LABEL: func @threadwise_prefetch_extra_idx
// CHECK-SAME: [[source:%.+]]: memref<3x2x64x30xf32>
func.func @threadwise_prefetch_extra_idx(%source: memref<3x2x64x30xf32>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[extra_idx:%.+]] = arith.constant 1
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP_IDX]], #[[$IN_FUNC_IDX]]]([[extra_idx]], [[bid]], [[tid]], [[zero]])
  // CHECK-SAME: bounds [1, 1, 1, 32]
  // CHECK-SAME: strides [1, 1, 1, 1]
  // CHECK-NEXT: rock.global_prefetch [[source]][[[args]]]

  %view = rock.transform %source by #transform_map3 : memref<3x2x64x30xf32> to memref<3x2x64x32xf32>
  %extra_idx = arith.constant 1 : index
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_prefetch {forceUnroll, useIndexDiffs}
    [#transform_map2](%view)[%extra_idx, %bid, %tid]
    : memref<3x2x64x32xf32>
  func.return
}
