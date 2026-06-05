// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand=none | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=NONE

// NONE-NOT: call @seedRandomValues

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,RAND1,RAND2,RAND3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side filter | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,RAND1,FIXED2,FIXED3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side input | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,RAND2,FIXED3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side filter -operation conv_bwd_data -v4r1 0 | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,RAND1,FIXED2,FIXED3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side filter -operation conv_bwd_data -v4r1 1 | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,RAND1,FIXED2,FIXED3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side output  -operation conv_bwd_data -v4r1 0 | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,FIXED2,RAND3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side output  -operation conv_bwd_data -v4r1 1 | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,FIXED2,RAND3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side input -operation conv_bwd_weight | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,RAND2,FIXED3
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_side output  -operation conv_bwd_weight | rocmlir-opt -canonicalize | FileCheck %s --check-prefixes=CHECK,HASFIXED,FIXED1,FIXED2,RAND3

// CHECK-LABEL: @main
// CHECK-DAG: %[[min:.*]] = arith.constant -5 : i16
// CHECK-DAG: %[[max:.*]] = arith.constant 5 : i16
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : i32
// HASFIXED-DAG: %[[one_i16:.*]] = arith.constant 1 : i16
// CHECK: call @seedRandomValues(%[[one]])

// CHECK: memref.alloc
// CHECK: affine.for
// RAND1-NEXT: %[[val1:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED1-NEXT: %[[val1:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// CHECK-NEXT: memref.store %[[val1]]
// CHECK: memref.alloc
// CHECK-NEXT: affine.for
// RAND2-NEXT: %[[val2:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED2-NEXT: %[[val2:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// CHECK-NEXT: memref.store %[[val2]]
// CHECK: memref.alloc
// CHECK-NEXT: affine.for
// RAND3-NEXT: %[[val3:.*]] = func.call @randomIntegerValue(%[[min]], %[[max]])
// FIXED3-NEXT: %[[val3:.*]] = func.call @randomIntegerValue(%[[one_i16]], %[[one_i16]])
// CHECK-NEXT: memref.store %[[val1]]

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 2 | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=SEED2
// SEED2: %[[two:.*]] = arith.constant 2 : i32
// SEED2: call @seedRandomValues(%[[two]])

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -ph -p -rand 1 -rand_type float -rand_min 1 -rand_max 3 | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=RAND_FLOAT

// RAND_FLOAT-LABEL: @main
// RAND_FLOAT-DAG: %[[min:.*]] = arith.constant 1 : i16
// RAND_FLOAT-DAG: %[[max:.*]] = arith.constant 3 : i16
// RAND_FLOAT-DAG: %[[one:.*]] = arith.constant 1 : i32
// RAND_FLOAT: call @seedRandomValues(%[[one]])

// RAND_FLOAT: memref.alloc
// RAND_FLOAT-NEXT: affine.for
// RAND_FLOAT-NEXT: %[[val1:.*]] = func.call @randomFloatValue(%[[min]], %[[max]])
// RAND_FLOAT-NEXT: memref.store %[[val1]]
// RAND_FLOAT: memref.alloc
// RAND_FLOAT-NEXT: affine.for
// RAND_FLOAT-NEXT: %[[val2:.*]] = func.call @randomFloatValue(%[[min]], %[[max]])
// RAND_FLOAT-NEXT: memref.store %[[val2]]
// RAND_FLOAT: memref.alloc
// RAND_FLOAT-NEXT: affine.for
// RAND_FLOAT-NEXT: %[[val3:.*]] = func.call @randomFloatValue(%[[min]], %[[max]])
// RAND_FLOAT-NEXT: memref.store %[[val1]]

// `-rand_min_int` / `-rand_max_int` override the [-5, 5) default range for
// integer randomness. All three GEMM args (A, B, C) should pick up the new
// bounds: exactly three randomIntegerValue calls, all using the overridden
// constants, and no float randomization helper anywhere.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -out_datatype i32 -g 1 -m 32 -n 32 -k 32 -ph -rand 1 -rand_type int -rand_min_int -3 -rand_max_int 7 | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=RAND_INT_BOUNDS
// RAND_INT_BOUNDS-DAG: %[[min:.*]] = arith.constant -3 : i16
// RAND_INT_BOUNDS-DAG: %[[max:.*]] = arith.constant 7 : i16
// RAND_INT_BOUNDS-COUNT-3: func.call @randomIntegerValue(%[[min]], %[[max]])
// RAND_INT_BOUNDS-NOT:     func.call @randomIntegerValue
// RAND_INT_BOUNDS-NOT: func.func private @randomFloatValue

// `-rand_type_int_for_inputs` selectively forces integer randomness on
// specific argument indices even when `-rand_type float` is in effect.
// Listing only index 0 leaves the other two args (B, C) as float random.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph -rand 1 -rand_type float -rand_type_int_for_inputs 0 | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=RAND_MIXED_ONE
// RAND_MIXED_ONE-COUNT-1: func.call @randomIntegerValue
// RAND_MIXED_ONE-NOT:     func.call @randomIntegerValue
// RAND_MIXED_ONE-COUNT-2: func.call @randomFloatValue
// RAND_MIXED_ONE-NOT:     func.call @randomFloatValue

// Repeating the flag accumulates indices; with both 0 and 1 forced to int,
// only the output remains a float-random tensor.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t f32 -g 1 -m 32 -n 32 -k 32 -ph -rand 1 -rand_type float -rand_type_int_for_inputs 0 -rand_type_int_for_inputs 1 | rocmlir-opt -canonicalize | FileCheck %s --check-prefix=RAND_MIXED_TWO
// RAND_MIXED_TWO-COUNT-2: func.call @randomIntegerValue
// RAND_MIXED_TWO-NOT:     func.call @randomIntegerValue
// RAND_MIXED_TWO-COUNT-1: func.call @randomFloatValue
// RAND_MIXED_TWO-NOT:     func.call @randomFloatValue
