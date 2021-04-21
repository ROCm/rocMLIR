// RUN: mlir-miopen-driver -ph -p | FileCheck %s 

// CHECK:  [[MIN:%.*]] = constant 1 : i16
// CHECK-NEXT:   [[MAX:%.*]] = constant 1 : i16
// CHECK-NEXT:   [[SEED:%.*]] = constant 1 : i32
// CHECK-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// CHECK-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1| FileCheck %s --check-prefix=RAND1

// RAND1:  [[MIN:%.*]] = constant -5 : i16
// RAND1-NEXT:   [[MAX:%.*]] = constant 5 : i16
// RAND1-NEXT:   [[SEED:%.*]] = constant 1 : i32
// RAND1-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// RAND1-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 2| FileCheck %s --check-prefix=RAND2

// RAND2:  [[MIN:%.*]] = constant -5 : i16
// RAND2-NEXT:   [[MAX:%.*]] = constant 5 : i16
// RAND2-NEXT:   [[SEED:%.*]] = constant 2 : i32
// RAND2-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// RAND2-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1 -rand_side filter | FileCheck %s --check-prefix=FWD_FILTER

// FWD_FILTER:    [[ONE:%.*]] = constant 1 : i16
// FWD_FILTER:    [[ZERO:%.*]] = constant 0 : i16
// FWD_FILTER-NEXT:  [[MIN:%.*]] = constant -5 : i16
// FWD_FILTER-NEXT:   [[MAX:%.*]] = constant 5 : i16
// FWD_FILTER-NEXT:   [[SEED:%.*]] = constant 1 : i32
// FWD_FILTER-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// FWD_FILTER-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[ONE]], [[ONE]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1 -rand_side input | FileCheck %s --check-prefix=FWD_INPUT

// FWD_INPUT:    [[ONE:%.*]] = constant 1 : i16
// FWD_INPUT:    [[ZERO:%.*]] = constant 0 : i16
// FWD_INPUT-NEXT:  [[MIN:%.*]] = constant -5 : i16
// FWD_INPUT-NEXT:   [[MAX:%.*]] = constant 5 : i16
// FWD_INPUT-NEXT:   [[SEED:%.*]] = constant 1 : i32
// FWD_INPUT-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[ONE]], [[ONE]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// FWD_INPUT-NEXT:    call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1 -rand_side filter -operation conv2d_bwd_data | FileCheck %s --check-prefix=BWD_DATA_FILTER

// BWD_DATA_FILTER:  [[ONE:%.*]] = constant 1 : i16
// BWD_DATA_FILTER:  [[ZERO:%.*]] = constant 0 : i16
// BWD_DATA_FILTER:  [[MIN:%.*]] = constant -5 : i16
// BWD_DATA_FILTER:  [[MAX:%.*]] = constant 5 : i16
// BWD_DATA_FILTER:  [[SEED:%.*]] = constant 1 : i32
// BWD_DATA_FILTER:   call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_DATA_FILTER:   call @mcpuMemset5DFloatRand({{.*}}, [[ZERO]], [[ZERO]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_DATA_FILTER:   call @mcpuMemset5DFloatRand({{.*}}, [[ONE]], [[ONE]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1 -rand_side output  -operation conv2d_bwd_data | FileCheck %s --check-prefix=BWD_DATA_OUTPUT

// BWD_DATA_OUTPUT:  [[ONE:%.*]] = constant 1 : i16
// BWD_DATA_OUTPUT:  [[ZERO:%.*]] = constant 0 : i16
// BWD_DATA_OUTPUT:  [[MIN:%.*]] = constant -5 : i16
// BWD_DATA_OUTPUT:  [[MAX:%.*]] = constant 5 : i16
// BWD_DATA_OUTPUT:  [[SEED:%.*]] = constant 1 : i32
// BWD_DATA_OUTPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[ONE]], [[ONE]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_DATA_OUTPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[ZERO]], [[ZERO]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_DATA_OUTPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1 -rand_side input -operation conv2d_bwd_weight | FileCheck %s --check-prefix=BWD_WEIGHT_INPUT

// BWD_WEIGHT_INPUT:  [[ONE:%.*]] = constant 1 : i16
// BWD_WEIGHT_INPUT:  [[ZERO:%.*]] = constant 0 : i16
// BWD_WEIGHT_INPUT:  [[MIN:%.*]] = constant -5 : i16
// BWD_WEIGHT_INPUT:  [[MAX:%.*]] = constant 5 : i16
// BWD_WEIGHT_INPUT:  [[SEED:%.*]] = constant 1 : i32
// BWD_WEIGHT_INPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[ZERO]], [[ZERO]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_WEIGHT_INPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_WEIGHT_INPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[ONE]], [[ONE]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

// RUN: mlir-miopen-driver -ph -p -rand 1 -rand_side output  -operation conv2d_bwd_weight | FileCheck %s --check-prefix=BWD_WEIGHT_OUTPUT

// BWD_WEIGHT_OUTPUT:  [[ONE:%.*]] = constant 1 : i16
// BWD_WEIGHT_OUTPUT:  [[ZERO:%.*]] = constant 0 : i16
// BWD_WEIGHT_OUTPUT:  [[MIN:%.*]] = constant -5 : i16
// BWD_WEIGHT_OUTPUT:  [[MAX:%.*]] = constant 5 : i16
// BWD_WEIGHT_OUTPUT:  [[SEED:%.*]] = constant 1 : i32
// BWD_WEIGHT_OUTPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[ZERO]], [[ZERO]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_WEIGHT_OUTPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[ONE]], [[ONE]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
// BWD_WEIGHT_OUTPUT:   call @mcpuMemset5DFloatRand({{.*}}, [[MIN]], [[MAX]], [[SEED]]) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
