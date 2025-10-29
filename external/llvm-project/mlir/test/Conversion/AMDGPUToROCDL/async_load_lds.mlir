// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1250 | FileCheck %s

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3

// CHECK-LABEL: func @global_load_to_rocdl_f32
// CHECK-SAME: (%[[ARG0:.*]]: memref<128x72xf32, 1>)
func.func @global_load_to_rocdl_f32(%global : memref<128x72xf32, #gpu_global_addrspace>) {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c32 = arith.constant 32 : index
  %alloc = memref.alloc() : memref<64x64xf32, #gpu_lds_addrspace>
  // CHECK: %[[GLOBAL_DESC:.*]] = builtin.unrealized_conversion_cast %[[ARG0]]

  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[IC0:.*]] = builtin.unrealized_conversion_cast %c0 : index to i64
  // CHECK: %[[C12:.*]] = arith.constant 12 : index
  // CHECK: %[[IC12:.*]] = builtin.unrealized_conversion_cast %[[C12]]
  // CHECK: %[[C32:.*]] = arith.constant 32 : index
  // CHECK: %[[IC32:.*]] = builtin.unrealized_conversion_cast %[[C32]]

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK: %[[LDS_DESC:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[GLOBAL_BASE:.*]] = llvm.extractvalue %[[GLOBAL_DESC]][1]

  // CHECK: %[[C72:.*]] = llvm.mlir.constant(72 : index) : i64
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IC12]], %[[C72]] : i64
  // CHECK: %[[SRC_OFFSET:.*]] = llvm.add %[[MUL]], %[[IC0]] : i64

  // CHECK: %[[GLOBAL_PTR:.*]] = llvm.getelementptr %[[GLOBAL_BASE]][%[[SRC_OFFSET]]]
  // CHECK: %[[LDS_BASE:.*]] = llvm.extractvalue %[[LDS_DESC]][1]

  // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : index) : i64
  // CHECK: %[[MUL_2:.*]] = llvm.mul %[[IC32]], %[[C64]] : i64
  // CHECK: %[[DST_OFFSET:.*]] = llvm.add %[[MUL_2]], %[[IC0]] : i64

  // CHECK: %[[LDS_PTR:.*]] = llvm.getelementptr %[[LDS_BASE]][%[[DST_OFFSET]]]
  // CHECK: rocdl.global.load.async.to.lds.b32 %[[GLOBAL_PTR]], %[[LDS_PTR]]
  amdgpu.async_load_to_lds %global[%c12, %c0], %alloc[%c32, %c0]
    : f32, memref<128x72xf32, #gpu_global_addrspace>, memref<64x64xf32, #gpu_lds_addrspace>
  func.return
}
