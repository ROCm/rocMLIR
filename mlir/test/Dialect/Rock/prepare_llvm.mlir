// RUN: rocmlir-opt %s --rock-prepare-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: @access_ptr
// CHECK-SAME: (%[[BASE:.+]]: !llvm.ptr, %[[IDX:.+]]: i64)
llvm.func @access_ptr(%base: !llvm.ptr, %idx: i64) -> (!llvm.ptr) attributes {rocdl.kernel} {
  %p0 = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %c1 = llvm.mlir.constant(1) : i64
  // CHECK: %[[NEXT:.+]] = llvm.add %[[IDX]]
  %next = llvm.add %idx, %c1 : i64
  // CHECK: = llvm.getelementptr inbounds %[[BASE]][%[[NEXT]]]
  %p2 = llvm.getelementptr %base[%next] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.return %p2 : !llvm.ptr
}

// -----

// CHECK-LABEL: @access_ptr_p7
// CHECK-SAME: (%[[BASE:.+]]: !llvm.ptr<7>, %[[IDX:.+]]: i32)
llvm.func @access_ptr_p7(%base: !llvm.ptr<7>, %idx: i32) -> (!llvm.ptr<7>) attributes {rocdl.kernel} {
  // CHECK: = llvm.getelementptr %[[BASE]][%[[IDX]]]
  %p0 = llvm.getelementptr %base[%idx] : (!llvm.ptr<7>, i32) -> !llvm.ptr<7>, f32
  llvm.return %p0 : !llvm.ptr<7>
}

// -----

llvm.mlir.global internal @lds() {addr_space = 3 : i32, alignment = 64 : i64} : !llvm.array<2048 x i8>

// CHECK-LABEL: @fix_align
llvm.func @fix_align(%arg0: !llvm.ptr {llvm.noalias}) -> vector<8xf16> attributes {rocdl.kernel} {
  // CHECK: llvm.load
  // CHECK-SAME: alignment = 16 : i64
  %v = llvm.load %arg0 {alignment = 2 : i64} : !llvm.ptr -> vector<8xf16>
  %lds = llvm.mlir.addressof @lds : !llvm.ptr<3>
  // CHECK: llvm.load
  // CHECK-SAME: alignment = 16 : i64
  %w = llvm.load %lds {alignment = 2 : i64} : !llvm.ptr<3> -> vector<8xf16>
  %ret = llvm.fadd %v, %w : vector<8xf16>
  llvm.return %ret : vector<8xf16>
}

// -----

// CHECK-LABEL: @invariant_load
llvm.func @invariant_load(%arg0: f32, %arg1: !llvm.ptr {llvm.noalias, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.writeonly}) attributes {rocdl.kernel} {
  // CHECK: llvm.load
  // CHECK-SAME: invariant
  %v = llvm.load %arg1 : !llvm.ptr -> f32
  %w = llvm.fadd %v, %arg0 : f32
  llvm.store %w, %arg2 {nontemporal} : f32, !llvm.ptr
  llvm.return
}

// -----

llvm.func @atomic_clean(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32) attributes {rocdl.kernel} {
  // CHECK: llvm.atomicrmw
  // CHECK-SAME: syncscope("agent-one-as") monotonic
  %v1 = llvm.atomicrmw add %arg0, %arg1 seq_cst : !llvm.ptr<1>, i32
  // CHECK: llvm.cmpxchg
  // CHECK: syncscope("agent-one-as") monotonic monotonic
  %v2 = llvm.cmpxchg %arg0, %v1, %arg2 seq_cst seq_cst : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @select_clean(%arg0: !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>,
                        %arg1: !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>,
                        %arg2: i32) -> !llvm.ptr<3> attributes {rocdl.kernel}{

  %c0 = llvm.mlir.constant(0) : i32
  %cond = llvm.icmp "ugt" %arg2, %c0: i32
  //CHECK: %[[cond:.*]] = llvm.icmp
  //CHECK: %[[ptr0:.*]] = llvm.extractvalue
  //CHECK: %[[ptr1:.*]] = llvm.extractvalue
  //CHECK: %[[ptr:.*]] = llvm.select %[[cond]], %[[ptr0]], %[[ptr1]]
  //CHECK  llvm.return %[[ptr]]
  %s = llvm.select %cond, %arg0, %arg1: i1, !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
  %ptr = llvm.extractvalue %s[0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
  llvm.return %ptr : !llvm.ptr<3>
}
