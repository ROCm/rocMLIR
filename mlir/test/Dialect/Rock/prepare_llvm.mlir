// RUN: rocmlir-opt %s --rock-prepare-llvm | FileCheck %s

// CHECK-LABEL: @access_ptr
// CHECK-SAME: (%[[BASE:.+]]: !llvm.ptr, %[[IDX:.+]]: i64)
llvm.func @access_ptr(%base: !llvm.ptr, %idx: i64) -> () {
  // CHECK: = llvm.getelementptr inbounds %[[BASE]][%[[IDX]]]
  %p0 = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %c1 = llvm.mlir.constant(1) : i64
  // CHECK: %[[NEXT:.+]] = llvm.add %[[IDX]]
  %next = llvm.add %idx, %c1 : i64
  // CHECK: = llvm.getelementptr inbounds %[[BASE]][%[[NEXT]]]
  %p2 = llvm.getelementptr %base[%next] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.return
}
