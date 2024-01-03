// RUN: rocmlir-driver -kernel-pipeline=gpu,rocdl %s | FileCheck %s

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1030"} {
// CHECK-LABEL: llvm.func @add_scalar
// CHECK-SAME: (%[[val:.*]]: f32,
// CHECK: %[[init:.+]] = rocdl.raw.ptr.buffer.load
// CHECK: llvm.br ^[[bb:.+]](%[[init]] : f32)
// CHECK: ^[[bb]](%[[prev:.+]]: f32)
// CHECK-DAG: %[[add:.+]] = llvm.fadd %[[val]], %[[prev]]
// CHECK-DAG: %[[prevInt:.+]] = llvm.bitcast %[[prev]] : f32 to i32
// CHECK-DAG: %[[addInt:.+]] = llvm.bitcast %[[add]] : f32 to i32
// CHECK: %[[resInt:.+]] = rocdl.raw.ptr.buffer.atomic.cmpswap %[[addInt]], %[[prevInt]]
// CHECK: %[[res:.+]] = llvm.bitcast %[[resInt]] : i32 to f32
// CHECK: %[[cond:.+]] = llvm.icmp "eq" %[[resInt]], %[[prevInt]]
// CHECK: llvm.cond_br %[[cond]], ^{{.*}}, ^[[bb]](%[[res]] : f32)
func.func @add_scalar(%val: f32, %mem: memref<4xf32>) attributes {kernel} {
  %c0 = arith.constant 0 : i32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = false} %val -> %mem[%c0] : f32 -> memref<4xf32>, i32
  return
}
}
