// RUN: rocmlir-gen --arch %arch -ph -p -rand=none | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=NONE

// NONE-NOT: call @seedRandomValues

// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,RAND1,RAND2,RAND3
// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 -rand_side filter | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,RAND1,FIXED2,FIXED3
// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 -rand_side input | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,RAND2,FIXED3
// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 -rand_side filter -operation conv_bwd_data | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,RAND1,FIXED2,FIXED3
// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 -rand_side output  -operation conv_bwd_data | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,FIXED2,RAND3
// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 -rand_side input -operation conv_bwd_weight | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,RAND2,FIXED3
// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 -rand_side output  -operation conv_bwd_weight | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,FIXED2,RAND3

// CHECK-DAG: %[[min:.*]] = arith.constant -5 : i16
// CHECK-DAG: %[[max:.*]] = arith.constant 5 : i16
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : i32
// HASFIXED-DAG: %[[one_i16:.*]] = arith.constant 1 : i16
// CHECK: call @seedRandomValues(%[[one]])

// CHECK: memref.collapse_shape
// CHECK-NEXT: affine.for
// RAND1-NEXT: %[[val1:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED1-NEXT: %[[val1:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// CHECK-NEXT: memref.store %[[val1]]
// CHECK: memref.collapse_shape
// CHECK-NEXT: affine.for
// RAND2-NEXT: %[[val2:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED2-NEXT: %[[val2:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// CHECK-NEXT: memref.store %[[val2]]
// CHECK: memref.collapse_shape
// CHECK-NEXT: affine.for
// RAND3-NEXT: %[[val3:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED3-NEXT: %[[val3:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// CHECK-NEXT: memref.store %[[val1]]

// RUN: rocmlir-gen --arch %arch -ph -p -rand 2 | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=SEED2
// SEED2: %[[two:.*]] = arith.constant 2 : i32
// SEED2: call @seedRandomValues(%[[two]])

// RUN: rocmlir-gen --arch %arch -ph -p -rand 1 -rand_type float -rand_min 1 -rand_max 3 | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=RAND_FLOAT

// RAND_FLOAT-DAG: %[[min:.*]] = arith.constant 1 : i16
// RAND_FLOAT-DAG: %[[max:.*]] = arith.constant 3 : i16
// RAND_FLOAT-DAG: %[[one:.*]] = arith.constant 1 : i32
// RAND_FLOAT: call @seedRandomValues(%[[one]])

// RAND_FLOAT: memref.collapse_shape
// RAND_FLOAT-NEXT: affine.for
// RAND_FLOAT-NEXT: %[[val1:.*]] = func.call @randomFloatValue(%[[min]], %[[max]])
// RAND_FLOAT-NEXT: memref.store %[[val1]]
// RAND_FLOAT: memref.collapse_shape
// RAND_FLOAT-NEXT: affine.for
// RAND_FLOAT-NEXT: %[[val2:.*]] = func.call @randomFloatValue(%[[min]], %[[max]])
// RAND_FLOAT-NEXT: memref.store %[[val2]]
// RAND_FLOAT: memref.collapse_shape
// RAND_FLOAT-NEXT: affine.for
// RAND_FLOAT-NEXT: %[[val3:.*]] = func.call @randomFloatValue(%[[min]], %[[max]])
// RAND_FLOAT-NEXT: memref.store %[[val1]]
