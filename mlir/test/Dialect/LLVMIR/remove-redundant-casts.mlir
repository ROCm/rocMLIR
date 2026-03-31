// RUN: rocmlir-opt --rock-remove-redundant-casts %s | FileCheck %s

// Parallel wide buffer already exists, so the pass should redirect the load to
// the wide buffer, eliminating fpext
// CHECK-LABEL: llvm.func @test_parallel_buffer_exists
llvm.func @test_parallel_buffer_exists() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(4 : i64) : i64
  %2 = llvm.mlir.constant(0 : i32) : i32
  %3 = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  %4 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %5 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %6 = llvm.fptrunc %3 : vector<4xf32> to vector<4xf16>
  llvm.store %6, %4 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %3, %5 : vector<4xf32>, !llvm.ptr<5>
  %7 = llvm.getelementptr %4[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %8 = llvm.getelementptr %5[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %9 = llvm.fptrunc %3 : vector<4xf32> to vector<4xf16>
  llvm.store %9, %7 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %3, %8 : vector<4xf32>, !llvm.ptr<5>
  %10 = llvm.getelementptr %4[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %11 = llvm.getelementptr %5[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %12 = llvm.fptrunc %3 : vector<4xf32> to vector<4xf16>
  llvm.store %12, %10 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %3, %11 : vector<4xf32>, !llvm.ptr<5>
  %13 = llvm.getelementptr %4[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %14 = llvm.getelementptr %5[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %15 = llvm.fptrunc %3 : vector<4xf32> to vector<4xf16>
  llvm.store %15, %13 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %3, %14 : vector<4xf32>, !llvm.ptr<5>
  %16 = llvm.load %4 : !llvm.ptr<5> -> vector<4xf16>
  %17 = llvm.fpext %16 : vector<4xf16> to vector<4xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  %18 = llvm.fadd %17, %3 : vector<4xf32>
  
  llvm.return
}

// No parallel buffer so the pass should create one
// CHECK-LABEL: llvm.func @test_create_wide_buffer
llvm.func @test_create_wide_buffer() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<2.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f16
  // CHECK: llvm.alloca {{.*}} x f32
  %3 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %3, %2 : vector<4xf16>, !llvm.ptr<5>
  %4 = llvm.getelementptr %2[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %5 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %5, %4 : vector<4xf16>, !llvm.ptr<5>
  %6 = llvm.getelementptr %2[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %7 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %7, %6 : vector<4xf16>, !llvm.ptr<5>
  %8 = llvm.getelementptr %2[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %9 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %9, %8 : vector<4xf16>, !llvm.ptr<5>
  %10 = llvm.load %2 : !llvm.ptr<5> -> vector<4xf16>
  // CHECK-NOT: llvm.fpext
  %11 = llvm.fpext %10 : vector<4xf16> to vector<4xf32>
  %12 = llvm.fadd %11, %1 : vector<4xf32>
  llvm.return
}

// The load reads subset of what was stored
// CHECK-LABEL: llvm.func @test_subset_load
llvm.func @test_subset_load() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<3.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %3 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %4 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %4, %2 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %3 : vector<4xf32>, !llvm.ptr<5>
  %5 = llvm.getelementptr %2[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %6 = llvm.getelementptr %3[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %7 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %7, %5 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %6 : vector<4xf32>, !llvm.ptr<5>
  %8 = llvm.getelementptr %2[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %9 = llvm.getelementptr %3[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %10 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %10, %8 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %9 : vector<4xf32>, !llvm.ptr<5>
  %11 = llvm.getelementptr %2[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %12 = llvm.getelementptr %3[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %13 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %13, %11 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %12 : vector<4xf32>, !llvm.ptr<5>
  %14 = llvm.load %2 : !llvm.ptr<5> -> vector<2xf16>
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  %15 = llvm.fpext %14 : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  %16 = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %17 = llvm.fadd %15, %16 : vector<2xf32>
  llvm.return
}

// Safe case: load through a GEP at a non-zero offset (with parallel buffer)
// CHECK-LABEL: llvm.func @test_load_with_gep
llvm.func @test_load_with_gep() {
  %sz = llvm.mlir.constant(16 : i64) : i64
  %val = llvm.mlir.constant(dense<2.500000e+00> : vector<4xf32>) : vector<4xf32>
  %narrow = llvm.alloca %sz x f16 : (i64) -> !llvm.ptr<5>
  %wide = llvm.alloca %sz x f32 : (i64) -> !llvm.ptr<5>
  %t0 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t0, %narrow : vector<4xf16>, !llvm.ptr<5>
  llvm.store %val, %wide : vector<4xf32>, !llvm.ptr<5>
  %gn4 = llvm.getelementptr %narrow[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %gw4 = llvm.getelementptr %wide[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %t1 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t1, %gn4 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %val, %gw4 : vector<4xf32>, !llvm.ptr<5>
  %gn8 = llvm.getelementptr %narrow[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %gw8 = llvm.getelementptr %wide[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %t2 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t2, %gn8 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %val, %gw8 : vector<4xf32>, !llvm.ptr<5>
  %gn12 = llvm.getelementptr %narrow[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %gw12 = llvm.getelementptr %wide[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %t3 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t3, %gn12 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %val, %gw12 : vector<4xf32>, !llvm.ptr<5>
  // Load at offset 4 through GEP — the pass should rewrite this to a GEP
  // from the wide buffer with f32 element type.
  %load_gep = llvm.getelementptr %narrow[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_gep : !llvm.ptr<5> -> vector<4xf16>
  %ext = llvm.fpext %loaded : vector<4xf16> to vector<4xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  %res = llvm.fadd %ext, %val : vector<4xf32>
  llvm.return
}

// Safe case: multiple loads at different GEP offsets (no parallel buffer)
// CHECK-LABEL: llvm.func @test_multiple_gep_loads
llvm.func @test_multiple_gep_loads() {
  %sz = llvm.mlir.constant(16 : i64) : i64
  %val = llvm.mlir.constant(dense<3.500000e+00> : vector<4xf32>) : vector<4xf32>
  %buf = llvm.alloca %sz x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  %t0 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t0, %buf : vector<4xf16>, !llvm.ptr<5>
  %g4 = llvm.getelementptr %buf[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t1 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t1, %g4 : vector<4xf16>, !llvm.ptr<5>
  %g8 = llvm.getelementptr %buf[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t2 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t2, %g8 : vector<4xf16>, !llvm.ptr<5>
  %g12 = llvm.getelementptr %buf[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t3 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t3, %g12 : vector<4xf16>, !llvm.ptr<5>
  // Two loads at different GEP offsets
  %lg0 = llvm.getelementptr %buf[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %l0 = llvm.load %lg0 : !llvm.ptr<5> -> vector<4xf16>
  %e0 = llvm.fpext %l0 : vector<4xf16> to vector<4xf32>
  %lg8 = llvm.getelementptr %buf[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %l1 = llvm.load %lg8 : !llvm.ptr<5> -> vector<4xf16>
  %e1 = llvm.fpext %l1 : vector<4xf16> to vector<4xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  %res0 = llvm.fadd %e0, %val : vector<4xf32>
  %res1 = llvm.fadd %e1, %val : vector<4xf32>
  llvm.return
}

// Unsafe case: intervening non-fptrunc store
// CHECK-LABEL: llvm.func @test_unsafe_intervening_store
llvm.func @test_unsafe_intervening_store() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<4.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.mlir.constant(dense<9.000000e+00> : vector<4xf16>) : vector<4xf16>
  %3 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %4 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %5 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %5, %3 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %4 : vector<4xf32>, !llvm.ptr<5>
  llvm.store %2, %3 : vector<4xf16>, !llvm.ptr<5>
  %6 = llvm.load %3 : !llvm.ptr<5> -> vector<4xf16>
  %7 = llvm.fpext %6 : vector<4xf16> to vector<4xf32>
  // CHECK: llvm.fpext
  %8 = llvm.fadd %7, %1 : vector<4xf32>
  llvm.return
}

// Unsafe case: fpext to different type than original
// CHECK-LABEL: llvm.func @test_type_mismatch
llvm.func @test_type_mismatch() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<5.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %3 = llvm.alloca %0 x f64 : (i64) -> !llvm.ptr<5>
  %4 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %4, %2 : vector<4xf16>, !llvm.ptr<5>
  %5 = llvm.load %2 : !llvm.ptr<5> -> vector<4xf16>
  %6 = llvm.fpext %5 : vector<4xf16> to vector<4xf64>
  // CHECK: llvm.fpext
  %7 = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf64>) : vector<4xf64>
  %8 = llvm.fadd %6, %7 : vector<4xf64>
  llvm.return
}

// Unsafe case: narrow buffer is not an alloca (function argument)
// CHECK-LABEL: llvm.func @test_not_alloca
llvm.func @test_not_alloca(%narrow_buf: !llvm.ptr<5>) {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<6.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %3 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %3, %narrow_buf : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %2 : vector<4xf32>, !llvm.ptr<5>
  %4 = llvm.load %narrow_buf : !llvm.ptr<5> -> vector<4xf16>
  %5 = llvm.fpext %4 : vector<4xf16> to vector<4xf32>
  // CHECK: llvm.fpext
  %6 = llvm.fadd %5, %1 : vector<4xf32>
  llvm.return
}

// Unsafe case: buffer not fully covered by fptrunc stores (partial coverage)
// CHECK-LABEL: llvm.func @test_partial_coverage
llvm.func @test_partial_coverage() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<7.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %3 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %4 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %4, %2 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %3 : vector<4xf32>, !llvm.ptr<5>
  %5 = llvm.load %2 : !llvm.ptr<5> -> vector<4xf16>
  %6 = llvm.fpext %5 : vector<4xf16> to vector<4xf32>
  // CHECK: llvm.fpext
  %7 = llvm.fadd %6, %1 : vector<4xf32>
  llvm.return
}

// The wide (f32) value is also stored into the SAME buffer as the narrow
// (f16) value at a non-overlapping offset. The pass must not treat this as a
// parallel wide store — doing so would redirect the load to reinterpret the
// f16 bytes as f32. Instead it should create a separate wide buffer.
// CHECK-LABEL: llvm.func @test_wide_store_same_buffer
llvm.func @test_wide_store_same_buffer() {
  %sz = llvm.mlir.constant(16 : i64) : i64
  %val = llvm.mlir.constant(dense<1.100000e+01> : vector<4xf32>) : vector<4xf32>
  %buf = llvm.alloca %sz x f16 : (i64) -> !llvm.ptr<5>
  %t0 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t0, %buf : vector<4xf16>, !llvm.ptr<5>
  %g4 = llvm.getelementptr %buf[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t1 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t1, %g4 : vector<4xf16>, !llvm.ptr<5>
  %g8 = llvm.getelementptr %buf[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t2 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t2, %g8 : vector<4xf16>, !llvm.ptr<5>
  %g12 = llvm.getelementptr %buf[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t3 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t3, %g12 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %val, %g8 : vector<4xf32>, !llvm.ptr<5>
  %loaded = llvm.load %buf : !llvm.ptr<5> -> vector<4xf16>
  %ext = llvm.fpext %loaded : vector<4xf16> to vector<4xf32>
  // CHECK-NOT: llvm.fpext
  %res = llvm.fadd %ext, %val : vector<4xf32>
  llvm.return
}

// Safe case: non-overlapping intervening store (store to different indices)
// CHECK-LABEL: llvm.func @test_non_overlapping_store
llvm.func @test_non_overlapping_store() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<8.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.mlir.constant(dense<9.000000e+00> : vector<4xf16>) : vector<4xf16>
  %3 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %4 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %5 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %5, %3 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %4 : vector<4xf32>, !llvm.ptr<5>
  %6 = llvm.getelementptr %3[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %7 = llvm.getelementptr %4[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %8 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %8, %6 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %7 : vector<4xf32>, !llvm.ptr<5>
  %9 = llvm.getelementptr %3[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %10 = llvm.getelementptr %4[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %11 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %11, %9 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %10 : vector<4xf32>, !llvm.ptr<5>
  %12 = llvm.getelementptr %3[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %13 = llvm.getelementptr %4[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %14 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %14, %12 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %13 : vector<4xf32>, !llvm.ptr<5>
  llvm.store %2, %9 : vector<4xf16>, !llvm.ptr<5>
  %15 = llvm.load %3 : !llvm.ptr<5> -> vector<4xf16>
  %16 = llvm.fpext %15 : vector<4xf16> to vector<4xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  %17 = llvm.fadd %16, %1 : vector<4xf32>
  llvm.return
}

// Unsafe case: fptrunc store does NOT dominate load (store after load)
// CHECK-LABEL: llvm.func @test_store_after_load
llvm.func @test_store_after_load() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<10.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %3 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %4 = llvm.load %2 : !llvm.ptr<5> -> vector<4xf16>
  %5 = llvm.fpext %4 : vector<4xf16> to vector<4xf32>
  // CHECK: llvm.fpext
  %6 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %6, %2 : vector<4xf16>, !llvm.ptr<5>
  llvm.store %1, %3 : vector<4xf32>, !llvm.ptr<5>
  %7 = llvm.fadd %5, %1 : vector<4xf32>
  llvm.return
}

// Unsafe case: fptrunc stores only execute on one branch of a conditional,
// so they do not dominate the post-merge load.
// CHECK-LABEL: llvm.func @test_store_in_conditional
llvm.func @test_store_in_conditional(%cond: i1) {
  %sz = llvm.mlir.constant(16 : i64) : i64
  %val = llvm.mlir.constant(dense<10.500000e+00> : vector<4xf32>) : vector<4xf32>
  %buf = llvm.alloca %sz x f16 : (i64) -> !llvm.ptr<5>
  llvm.cond_br %cond, ^then, ^merge
^then:
  %t0 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t0, %buf : vector<4xf16>, !llvm.ptr<5>
  %g1 = llvm.getelementptr %buf[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t1 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t1, %g1 : vector<4xf16>, !llvm.ptr<5>
  %g2 = llvm.getelementptr %buf[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t2 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t2, %g2 : vector<4xf16>, !llvm.ptr<5>
  %g3 = llvm.getelementptr %buf[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t3 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t3, %g3 : vector<4xf16>, !llvm.ptr<5>
  llvm.br ^merge
^merge:
  %loaded = llvm.load %buf : !llvm.ptr<5> -> vector<4xf16>
  %ext = llvm.fpext %loaded : vector<4xf16> to vector<4xf32>
  // CHECK: llvm.fpext
  %res = llvm.fadd %ext, %val : vector<4xf32>
  llvm.return
}

// Unsafe case: fptrunc stores inside a loop body do not dominate the
// post-loop load (^exit is reached from ^header, not ^body).
// CHECK-LABEL: llvm.func @test_store_in_loop
llvm.func @test_store_in_loop(%n: i32) {
  %sz = llvm.mlir.constant(16 : i64) : i64
  %val = llvm.mlir.constant(dense<10.750000e+00> : vector<4xf32>) : vector<4xf32>
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %buf = llvm.alloca %sz x f16 : (i64) -> !llvm.ptr<5>
  llvm.br ^header(%zero : i32)
^header(%i: i32):
  %cond = llvm.icmp "slt" %i, %n : i32
  llvm.cond_br %cond, ^body, ^exit
^body:
  %t0 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t0, %buf : vector<4xf16>, !llvm.ptr<5>
  %g1 = llvm.getelementptr %buf[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t1 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t1, %g1 : vector<4xf16>, !llvm.ptr<5>
  %g2 = llvm.getelementptr %buf[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t2 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t2, %g2 : vector<4xf16>, !llvm.ptr<5>
  %g3 = llvm.getelementptr %buf[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %t3 = llvm.fptrunc %val : vector<4xf32> to vector<4xf16>
  llvm.store %t3, %g3 : vector<4xf16>, !llvm.ptr<5>
  %next = llvm.add %i, %one : i32
  llvm.br ^header(%next : i32)
^exit:
  %loaded = llvm.load %buf : !llvm.ptr<5> -> vector<4xf16>
  %ext = llvm.fpext %loaded : vector<4xf16> to vector<4xf32>
  // CHECK: llvm.fpext
  %res = llvm.fadd %ext, %val : vector<4xf32>
  llvm.return
}

// Cleanup: when the narrow buffer has no remaining uses after transformation,
// the fptrunc ops, narrow stores, and narrow alloca should all be erased
// CHECK-LABEL: llvm.func @test_cleanup_erases_narrow_ops
llvm.func @test_cleanup_erases_narrow_ops() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<12.000000e+00> : vector<4xf32>) : vector<4xf32>
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f16
  // CHECK: llvm.alloca {{.*}} x f32
  %3 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %3, %2 : vector<4xf16>, !llvm.ptr<5>
  %4 = llvm.getelementptr %2[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %5 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %5, %4 : vector<4xf16>, !llvm.ptr<5>
  %6 = llvm.getelementptr %2[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %7 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %7, %6 : vector<4xf16>, !llvm.ptr<5>
  %8 = llvm.getelementptr %2[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %9 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %9, %8 : vector<4xf16>, !llvm.ptr<5>
  %10 = llvm.load %2 : !llvm.ptr<5> -> vector<4xf16>
  %11 = llvm.fpext %10 : vector<4xf16> to vector<4xf32>
  // CHECK-NOT: llvm.fptrunc
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  %12 = llvm.fadd %11, %1 : vector<4xf32>
  llvm.return
}

// Cleanup: when the narrow buffer still has other uses (here, ptrtoint) after
// transformation, the fptrunc stores and narrow alloca must be KEPT
// CHECK-LABEL: llvm.func @test_cleanup_keeps_narrow_ops_when_buffer_in_use
llvm.func @test_cleanup_keeps_narrow_ops_when_buffer_in_use() {
  %0 = llvm.mlir.constant(16 : i64) : i64
  %1 = llvm.mlir.constant(dense<13.000000e+00> : vector<4xf32>) : vector<4xf32>
  // CHECK: llvm.alloca {{.*}} x f16
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  %3 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  // CHECK: llvm.fptrunc
  llvm.store %3, %2 : vector<4xf16>, !llvm.ptr<5>
  %4 = llvm.getelementptr %2[4] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %5 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %5, %4 : vector<4xf16>, !llvm.ptr<5>
  %6 = llvm.getelementptr %2[8] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %7 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %7, %6 : vector<4xf16>, !llvm.ptr<5>
  %8 = llvm.getelementptr %2[12] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %9 = llvm.fptrunc %1 : vector<4xf32> to vector<4xf16>
  llvm.store %9, %8 : vector<4xf16>, !llvm.ptr<5>
  // This ptrtoint keeps the narrow buffer "observable", preventing cleanup
  // of the fptrunc stores and narrow alloca
  %ptr_as_int = llvm.ptrtoint %2 : !llvm.ptr<5> to i64
  %10 = llvm.load %2 : !llvm.ptr<5> -> vector<4xf16>
  %11 = llvm.fpext %10 : vector<4xf16> to vector<4xf32>
  // The load→fpext transformation should still succeed:
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<4xf32>
  %12 = llvm.fadd %11, %1 : vector<4xf32>
  llvm.return
}

// Safe case: scalar types (not vectors)
// CHECK-LABEL: llvm.func @test_scalar_types
llvm.func @test_scalar_types() {
  %0 = llvm.mlir.constant(4 : i64) : i64
  %1 = llvm.mlir.constant(11.000000e+00 : f32) : f32
  %2 = llvm.alloca %0 x f16 : (i64) -> !llvm.ptr<5>
  %3 = llvm.alloca %0 x f32 : (i64) -> !llvm.ptr<5>
  %4 = llvm.fptrunc %1 : f32 to f16
  llvm.store %4, %2 : f16, !llvm.ptr<5>
  llvm.store %1, %3 : f32, !llvm.ptr<5>
  %5 = llvm.getelementptr %2[1] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %6 = llvm.getelementptr %3[1] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %7 = llvm.fptrunc %1 : f32 to f16
  llvm.store %7, %5 : f16, !llvm.ptr<5>
  llvm.store %1, %6 : f32, !llvm.ptr<5>
  %8 = llvm.getelementptr %2[2] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %9 = llvm.getelementptr %3[2] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %10 = llvm.fptrunc %1 : f32 to f16
  llvm.store %10, %8 : f16, !llvm.ptr<5>
  llvm.store %1, %9 : f32, !llvm.ptr<5>
  %11 = llvm.getelementptr %2[3] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %12 = llvm.getelementptr %3[3] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %13 = llvm.fptrunc %1 : f32 to f16
  llvm.store %13, %11 : f16, !llvm.ptr<5>
  llvm.store %1, %12 : f32, !llvm.ptr<5>
  %14 = llvm.load %2 : !llvm.ptr<5> -> f16
  %15 = llvm.fpext %14 : f16 to f32
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> f32
  %16 = llvm.fadd %15, %1 : f32
  llvm.return
}

