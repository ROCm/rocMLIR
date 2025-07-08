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

#transform_map4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2 - 2)>
by [<PassThrough ["1", "0"] at [0, 1]  -> ["1", "0"] at [0, 1]>,
  <Pad{2, 0} ["z"] at [2] -> ["z"] at [2]>]
bounds = [1, 128, 32] -> [1, 128, 30]>


// CHECK-LABEL: func @threadwise_read_into
// CHECK-SAME: [[source:%.+]]: memref<2x64x30xf32>, [[dest:%.+]]: memref<32xf32, #gpu.address_space<private>>
func.func @threadwise_read_into( %source: memref<2x64x30xf32>, %dest: memref<32xf32, #gpu.address_space<private>>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ({{%.*}}, {{%.*}}, [[i:%.+]]) = []([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ([[valid:%.+]], {{%.*}}) = validity
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 2]
  // CHECK-NEXT: [[tmp:%.+]] = rock.global_load [[source]][[[args]]] if [[valid]]
  // CHECK-NEXT: rock.in_bounds_store [[tmp]] -> [[dest]][[[i]]]

  %view = rock.transform %source by #transform_map1 : memref<2x64x30xf32> to memref<2x64x32xf32>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map0](%view)[%bid, %tid] -> %dest
    : memref<2x64x32xf32> -> memref<32xf32, #gpu.address_space<private>>
  func.return
}

// CHECK-LABEL: func @threadwise_read_into_scalar
// CHECK-SAME: [[source:%.+]]: memref<f32>, [[dest:%.+]]: memref<1xf32, #gpu.address_space<private>>
func.func @threadwise_read_into_scalar(%source: memref<f32>, %dest: memref<1xf32, #gpu.address_space<private>>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: () = [#transform_map{{[0-9]*}}]([[zero]])
  // CHECK-SAME: ([[i:%.+]]) = []([[zero]])
  // CHECK-SAME: ([[valid:%.+]], {{%.*}}) = validity
  // CHECK-SAME: bounds [1]
  // CHECK-SAME: strides [1]
  // CHECK-NEXT: [[tmp:%.+]] = rock.global_load [[source]][] if [[valid]]
  // CHECK-NEXT: rock.in_bounds_store [[tmp]] -> [[dest]][[[i]]]
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [](%source)[] -> %dest
    : memref<f32> -> memref<1xf32, #gpu.address_space<private>>
  func.return
}


// CHECK-LABEL: func @threadwise_read_into_extra_idx
// CHECK-SAME: [[source:%.+]]: memref<3x2x64x30xf32>, [[dest:%.+]]: memref<32xf32, #gpu.address_space<private>>
func.func @threadwise_read_into_extra_idx( %source: memref<3x2x64x30xf32>, %dest: memref<32xf32, #gpu.address_space<private>>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
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
  // CHECK-NEXT: [[tmp:%.+]] = rock.global_load [[source]][[[args]]] if [[valid]]
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
func.func @threadwise_write_all(%source: memref<32xf32, #gpu.address_space<private>>, %dest: memref<2x64x30xf32>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ({{%.*}}, {{%.*}}, [[i:%.+]]) = []([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], [[zero]])
  // CHECK-SAME: ({{%.*}}, [[valid:%.+]]) = validity
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 2]
  // CHECK-NEXT: rock.global_store set [[source]][[[i]]] -> [[dest]][[[args]]] if [[valid]]

  %view = rock.transform %dest by #transform_map1 : memref<2x64x30xf32> to memref<2x64x32xf32>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_write_all features = dot {forceUnroll, useIndexDiffs}
    %source -> [#transform_map0](%view)[%bid, %tid] by set
    : memref<32xf32, #gpu.address_space<private>> -> memref<2x64x32xf32>
  func.return
}

// CHECK-LABEL: func @threadwise_write_all_scalar
// CHECK-SAME: [[source:%.+]]: memref<1xf32, #gpu.address_space<private>>, [[dest:%.+]]: memref<f32>
func.func @threadwise_write_all_scalar(%source: memref<1xf32, #gpu.address_space<private>>, %dest: memref<f32>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[i:%.+]]) = []([[zero]])
  // CHECK-SAME: () = [#{{.*}}]([[zero]])
  // CHECK-SAME: ({{%.*}}, [[valid:%.+]]) = validity
  // CHECK-SAME: bounds [1]
  // CHECK-SAME: strides [1]
  // CHECK-NEXT: rock.global_store set [[source]][[[i]]] -> [[dest]][] if [[valid]]

  rock.threadwise_write_all features = dot {forceUnroll, useIndexDiffs}
    %source -> [](%dest)[] by set
    : memref<1xf32, #gpu.address_space<private>> -> memref<f32>
  func.return
}

// CHECK-LABEL: func @threadwise_write_all_extra_idx
// CHECK-SAME: [[source:%.+]]: memref<32xf32, #gpu.address_space<private>>, [[dest:%.+]]: memref<3x2x64x30xf32>
func.func @threadwise_write_all_extra_idx(%source: memref<32xf32, #gpu.address_space<private>>, %dest: memref<3x2x64x30xf32>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
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
  // CHECK-NEXT: rock.global_store set [[source]][[[i]]] -> [[dest]][[[args]]] if [[valid]]

  %view = rock.transform %dest by #transform_map3 : memref<3x2x64x30xf32> to memref<3x2x64x32xf32>
  %extra_idx = arith.constant 2 : index
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_write_all features = dot {forceUnroll, useIndexDiffs}
    %source -> [#transform_map2](%view)[%extra_idx, %bid, %tid] by set
    : memref<32xf32, #gpu.address_space<private>> -> memref<3x2x64x32xf32>
  func.return
}

// CHECK-LABEL: func @threadwise_read_into_big_to_small_vec
// CHECK-SAME: %[[source:.+]]: memref<4xvector<8xf16>, #gpu.address_space<workgroup>>, %[[dest:.+]]: memref<8xvector<4xf16>, #gpu.address_space<private>>
func.func @threadwise_read_into_big_to_small_vec(%source: memref<4xvector<8xf16>, #gpu.address_space<workgroup>>, %dest: memref<8xvector<4xf16>, #gpu.address_space<private>>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4] strides [1] {
    // CHECK-DAG: %[[ldval:.+]] = scf.if
    // CHECK-DAG: %[[ldvalSlice:.+]] = rock.extract_slice %[[ldval]][%c0] : vector<8xf16> -> vector<4xf16>
    // CHECK-DAG: %[[baseElOffset:.+]] = arith.muli %{{.*}}, %c8 : index
    // CHECK-DAG: %[[destElOffset:.+]] = arith.divui %[[baseElOffset]], %c4 : index
    // CHECK-DAG: memref.store %[[ldvalSlice]], %[[dest]][%[[destElOffset]]] : memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK-DAG: %[[ldvalSlice2:.+]] = rock.extract_slice %[[ldval]][%c4] : vector<8xf16> -> vector<4xf16>
    // CHECK-DAG: %[[baseElOffset2:.+]] = arith.muli %{{.*}}, %c8 : index
    // CHECK-DAG: %[[baseElOffset2SliceOffset:.+]] = arith.addi %[[baseElOffset2]], %c4 : index
    // CHECK-DAG: %[[destElOffset2:.+]] = arith.divui %[[baseElOffset2SliceOffset]], %c4 : index
    // CHECK-DAG: memref.store %[[ldvalSlice2]], %[[dest]][%[[destElOffset2]]] : memref<8xvector<4xf16>, #gpu.address_space<private>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%source)[] -> %dest : memref<4xvector<8xf16>, #gpu.address_space<workgroup>> -> memref<8xvector<4xf16>, #gpu.address_space<private>>
  func.return
}

// CHECK-LABEL: func @threadwise_read_into_scalar_to_vec
// CHECK-SAME: %[[source:.+]]: memref<32xf16, #gpu.address_space<workgroup>>, %[[dest:.+]]: memref<8xvector<4xf16>, #gpu.address_space<private>>
func.func @threadwise_read_into_scalar_to_vec(%source: memref<32xf16, #gpu.address_space<workgroup>>, %dest: memref<8xvector<4xf16>, #gpu.address_space<private>>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [32] strides [8] {
    // CHECK-DAG: %[[ldval:.+]] = scf.if
    // CHECK-DAG: %[[ldvalSlice:.+]] = rock.extract_slice %[[ldval]][%c0] : vector<8xf16> -> vector<4xf16>
    // CHECK-DAG: %[[baseElOffset:.+]] = arith.muli %{{.*}}, %c8 : index
    // CHECK-DAG: %[[destElOffset:.+]] = arith.divui %[[baseElOffset]], %c4 : index
    // CHECK-DAG: memref.store %[[ldvalSlice]], %[[dest]][%[[destElOffset]]] : memref<8xvector<4xf16>, #gpu.address_space<private>>
    // CHECK-DAG: %[[ldvalSlice2:.+]] = rock.extract_slice %[[ldval]][%c4] : vector<8xf16> -> vector<4xf16>
    // CHECK-DAG: %[[baseElOffset2:.+]] = arith.muli %{{.*}}, %c8 : index
    // CHECK-DAG: %[[baseElOffset2SliceOffset:.+]] = arith.addi %[[baseElOffset2]], %c4 : index
    // CHECK-DAG: %[[destElOffset2:.+]] = arith.divui %[[baseElOffset2SliceOffset]], %c4 : index
    // CHECK-DAG: memref.store %[[ldvalSlice2]], %[[dest]][%[[destElOffset2]]] : memref<8xvector<4xf16>, #gpu.address_space<private>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%source)[] -> %dest : memref<32xf16, #gpu.address_space<workgroup>> -> memref<8xvector<4xf16>, #gpu.address_space<private>>
  func.return
}

// CHECK-LABEL: func @threadwise_read_into_small_to_big_vec
// CHECK-SAME: %[[source:.+]]: memref<8xvector<4xf16>, #gpu.address_space<workgroup>>, %[[dest:.+]]: memref<4xvector<8xf16>, #gpu.address_space<private>>
func.func @threadwise_read_into_small_to_big_vec(%source: memref<8xvector<4xf16>, #gpu.address_space<workgroup>>, %dest: memref<4xvector<8xf16>, #gpu.address_space<private>>) attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [8] strides [1] {
    // CHECK-DAG: %[[ldval:.+]] = scf.if
    // CHECK-DAG: %[[baseElOffset:.+]] = arith.muli {{.*}}, %c4 : index
    // CHECK-DAG: %[[destElOffset:.+]] = arith.divui %[[baseElOffset]], %c8 : index
    // CHECK-DAG: %[[destVec:.+]] = memref.load %[[dest]][%[[destElOffset]]] : memref<4xvector<8xf16>, #gpu.address_space<private>>
    // CHECK-DAG: %[[destVecSliceOffset:.+]] = arith.remui %[[baseElOffset]], %c8 : index
    // CHECK-DAG: %[[newDestVec:.+]] = rock.insert_slice %[[ldval]] -> %[[destVec]][%[[destVecSliceOffset]]] : vector<4xf16> -> vector<8xf16>
    // CHECK-DAG: memref.store %[[newDestVec]], %[[dest]][%[[destElOffset]]] : memref<4xvector<8xf16>, #gpu.address_space<private>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%source)[] -> %dest : memref<8xvector<4xf16>, #gpu.address_space<workgroup>> -> memref<4xvector<8xf16>, #gpu.address_space<private>>
  func.return
}

// Note: The source is given as private memory to comply with the validity check
// that got added during review without reeding to rewrite the test. If we start
// supporting dynamic validities for reads from global memory, take the <private>
// off the source.
// CHECK-LABEL: func @threadwise_read_into_validities
// CHECK-SAME: [[source:%.+]]: memref<2x64x30xf32, #gpu.address_space<private>>, [[dest:%.+]]: memref<32xf32, #gpu.address_space<private>>, [[validIn:%.+]]: vector<32xi1>
func.func @threadwise_read_into_validities(%source: memref<2x64x30xf32, #gpu.address_space<private>>,
    %dest: memref<32xf32, #gpu.address_space<private>>,
    %validIn: vector<32xi1>) -> vector<32xi1> attributes {block_size = 128 : i32, arch = "##TOKEN_ARCH##"} {
  // CHECK-DAG: [[trues:%.+]] = arith.constant dense<true>
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id
  // CHECK: [[validOut:%.+]] = rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: ([[args:%.+, %.+, %.+]]) = [#[[$ON_OP]], #[[$IN_FUNC]]]([[bid]], [[tid]], %{{[^)]+}})
  // CHECK-SAME: ({{%.*}}, {{%.*}}, [[i:%.+]]) = []([[bid]], [[tid]], %{{[^)]+}})
  // CHECK-SAME: ([[valid:%.+]], {{%.*}}) = validity
  // CHECK-SAME: iter_args ([[validOutIter:%.+]] = [[trues]]) -> (vector<32xi1>)
  // CHECK-SAME: bounds [1, 1, 32]
  // COM: Note that the strides are 1 here despite a potential vectorization.
  // CHECK-SAME: strides [1, 1, 1]
  // CHECK-NEXT: [[vInElem:%.+]] = vector.extract [[validIn]][[[i]]]
  // CHECK-NEXT: [[validCombo:%.+]] = arith.andi [[valid]], [[vInElem]] : i1
  // CHECK-NEXT: [[validOutIterNext:%.+]] = rock.insert_slice [[validCombo]] -> [[validOutIter]][[[i]]]
  // CHECK-NEXT: [[tmp:%.+]] = scf.if [[validCombo]]
  // CHECK: rock.in_bounds_store [[tmp]] -> [[dest]][[[i]]]
  // CHECK-NEXT: yield [[validOutIterNext]]
  // CHECK: return [[validOut]]

  %view = rock.transform %source by #transform_map1 : memref<2x64x30xf32, #gpu.address_space<private>> to memref<2x64x32xf32, #gpu.address_space<private>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  %validOut = rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map0](%view)[%bid, %tid] -> %dest
    if [%validIn]
    : memref<2x64x32xf32, #gpu.address_space<private>> -> memref<32xf32, #gpu.address_space<private>>, vector<32xi1>
  func.return %validOut : vector<32xi1>
}

// Note: We hardcode gfx950 arch to make sure we can use rock.global_load_to_lds
// CHECK-LABEL: func @threadwise_read_into_lds
// CHECK-SAME: [[source:%.+]]: memref<1x128x30xf16>, [[dest:%.+]]: memref<8192xi8, #gpu.address_space<workgroup>>
func.func @threadwise_read_into_lds(%source: memref<1x128x30xf16>, %lds: memref<8192xi8, #gpu.address_space<workgroup>>) attributes {block_size = 128 : i32, arch = "gfx950:sramecc+:xnack-"} {
  // CHECK-DAG: [[c128:%.+]] = arith.constant 128 : index
  // CHECK-DAG: [[zero:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[bid:%.+]] = rock.workgroup_id : index
  // CHECK-DAG: [[tid:%.+]] = rock.workitem_id : index
  // CHECK-DAG: [[ldsView:%.+]] = memref.view {{.*}} memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xf16, #gpu.address_space<workgroup>>
  // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
  // CHECK-SAME: bounds [1, 1, 32]
  // CHECK-SAME: strides [1, 1, 2]
  // CHECK-NEXT: [[ldsIndex0:%.+]] = arith.muli %{{.*}}, [[c128]] : index
  // CHECK-NEXT: [[ldsIndex1:%.+]] = arith.muli %{{.*}}, [[c128]] : index
  // CHECK-NEXT: [[ldsIndex:%.+]] = arith.addi [[ldsIndex0]], [[ldsIndex1]] : index
  // CHECK-NEXT: rock.global_load_to_lds [[source]]{{.*}} -> [[ldsView]][[[ldsIndex]]] if {{.*}} {transferType = f32} : memref<1x128x30xf16> -> memref<4096xf16, #gpu.address_space<workgroup>>

  %view = rock.transform %source by #transform_map4 : memref<1x128x30xf16> to memref<1x128x32xf16>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  %c0 = arith.constant 0 : index
  %view_lds = memref.view %lds[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xf16, #gpu.address_space<workgroup>>
  %55 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} 
    [](%view) [%bid, %tid] -> %view_lds 
    : memref<1x128x32xf16> -> memref<4096xf16, #gpu.address_space<workgroup>>, vector<4096xi1>
  func.return
}
