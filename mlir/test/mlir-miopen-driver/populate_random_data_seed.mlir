// RUN: mlir-miopen-driver -ph -p | FileCheck %s 

// CHECK:  [[MIN:%.*]] = constant 1 : i16
// CHECK-NEXT:   [[MAX:%.*]] = constant 1 : i16
// CHECK-NEXT:   [[SEED:%.*]] = constant 1 : i32
// CHECK-NEXT:    call @mcpuMemset4DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?xf32>, i16, i16, i32) -> ()
// CHECK-NEXT:    call @mcpuMemset4DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1| FileCheck %s --check-prefix=RAND1

// RAND1:  [[MIN:%.*]] = constant -5 : i16
// RAND1-NEXT:   [[MAX:%.*]] = constant 5 : i16
// RAND1-NEXT:   [[SEED:%.*]] = constant 1 : i32
// RAND1-NEXT:    call @mcpuMemset4DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?xf32>, i16, i16, i32) -> ()
// RAND1-NEXT:    call @mcpuMemset4DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 2| FileCheck %s --check-prefix=RAND2

// RAND2:  [[MIN:%.*]] = constant -5 : i16
// RAND2-NEXT:   [[MAX:%.*]] = constant 5 : i16
// RAND2-NEXT:   [[SEED:%.*]] = constant 2 : i32
// RAND2-NEXT:    call @mcpuMemset4DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?xf32>, i16, i16, i32) -> ()
// RAND2-NEXT:    call @mcpuMemset4DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?xf32>, i16, i16, i32) -> ()
