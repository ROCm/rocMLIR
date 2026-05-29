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

// Safe case: dynamic GEPs from canonical loops cover the full buffer.
// CHECK-LABEL: llvm.func @test_dynamic_full_coverage_loop
llvm.func @test_dynamic_full_coverage_loop() {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):  // 2 preds: ^bb0, ^store_body
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^load_header(%zero : i32)

^store_body:  // pred: ^store_header
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^load_header(%j: i32):  // 2 preds: ^store_header, ^load_body
  %keep_loading = llvm.icmp "slt" %j, %sixteen : i32
  llvm.cond_br %keep_loading, ^load_body, ^done

^load_body:  // pred: ^load_header
  %load_ptr = llvm.getelementptr %narrow[%j] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  %sum = llvm.fadd %extended, %wide : vector<2xf32>
  %next_j = llvm.add %j, %two : i32
  llvm.br ^load_header(%next_j : i32)

^done:  // pred: ^load_header
  llvm.return
}

// The f16 buffer is populated by a zero-based counted loop, then read by a later
// counted loop and immediately fpext'd before f32 add/relu.
// CHECK-LABEL: llvm.func @test_reduced_convolution_epilogue_from_test_mlir
llvm.func @test_reduced_convolution_epilogue_from_test_mlir() {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %relu_zero = llvm.mlir.constant(dense<0.000000e+00> : vector<2xf32>) : vector<2xf32>
  %bias = llvm.alloca %size x f32 : (i64) -> !llvm.ptr<5>
  %out = llvm.alloca %size x f32 : (i64) -> !llvm.ptr<5>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  %wmma_acc = llvm.alloca %size x f32 : (i64) -> !llvm.ptr<5>
  llvm.br ^bb38(%zero : i32)

^bb38(%iv_store: i32):  // 2 preds: ^bb0, ^bb39
  %keep_storing = llvm.icmp "slt" %iv_store, %sixteen : i32
  llvm.cond_br %keep_storing, ^bb39, ^bb40

^bb39:  // pred: ^bb38
  %acc_ptr = llvm.getelementptr %wmma_acc[%iv_store] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
  %acc = llvm.load %acc_ptr {alignment = 4 : i64} : !llvm.ptr<5> -> vector<2xf32>
  %narrow_value = llvm.fptrunc %acc : vector<2xf32> to vector<2xf16>
  %narrow_store_ptr = llvm.getelementptr %narrow[%iv_store] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %narrow_store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_store = llvm.add %iv_store, %two : i32
  llvm.br ^bb38(%next_store : i32)

^bb40:  // pred: ^bb38
  llvm.br ^bb41(%zero : i32)

^bb41(%iv_load: i32):  // 2 preds: ^bb40, ^bb42
  %keep_loading = llvm.icmp "slt" %iv_load, %sixteen : i32
  llvm.cond_br %keep_loading, ^bb42, ^bb43

^bb42:  // pred: ^bb41
  %narrow_load_ptr = llvm.getelementptr %narrow[%iv_load] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %narrow_load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %bias_ptr = llvm.getelementptr %bias[%iv_load] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
  %bias_value = llvm.load %bias_ptr {alignment = 4 : i64} : !llvm.ptr<5> -> vector<2xf32>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  %sum = llvm.fadd %extended, %bias_value : vector<2xf32>
  %relu = llvm.intr.maximum(%sum, %relu_zero) : (vector<2xf32>, vector<2xf32>) -> vector<2xf32>
  %out_ptr = llvm.getelementptr %out[%iv_load] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
  llvm.store %relu, %out_ptr {alignment = 4 : i64} : vector<2xf32>, !llvm.ptr<5>
  %next_load = llvm.add %iv_load, %two : i32
  llvm.br ^bb41(%next_load : i32)

^bb43:  // pred: ^bb41
  llvm.return
}

// Safe case: the loop guard uses the swapped form `bound > iv` (equivalent to
// `iv < bound`). The counted-loop recognizer accepts both operand orderings.
// CHECK-LABEL: llvm.func @test_dynamic_loop_swapped_operand_predicate
llvm.func @test_dynamic_loop_swapped_operand_predicate() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "sgt" %sixteen, %i : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: a bypass edge reaches the load block without going through
// the store loop, so the loop header does not dominate the load. The pass
// must not redirect the load to a wide buffer that may never have been
// written.
// CHECK-LABEL: llvm.func @test_dynamic_loop_bypass
llvm.func @test_dynamic_loop_bypass(%cond: i1) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.cond_br %cond, ^store_header(%zero : i32), ^bypass

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^load_block(%zero : i32)

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^bypass:
  llvm.br ^load_block(%zero : i32)

^load_block(%j: i32):
  %load_ptr = llvm.getelementptr %narrow[%j] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the load is inside the store loop body, so at load time the
// loop has only completed the current iteration -- partial, not full
// coverage. The pass must not fire.
// CHECK-LABEL: llvm.func @test_dynamic_loop_load_in_body
llvm.func @test_dynamic_loop_load_in_body() {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^header(%zero : i32)

^header(%i: i32):
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^body, ^done

^body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %load_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  %sum = llvm.fadd %extended, %wide : vector<2xf32>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^header(%next_i : i32)

^done:
  llvm.return
}

// Unsafe case: the loop body has a second predecessor (a block reached after
// the loop branches back into it), so the body can execute with an `iv`
// outside [lowerBound, upperBound). Full coverage can no longer be proven, so
// the pass must not fire.
// CHECK-LABEL: llvm.func @test_dynamic_loop_body_extra_predecessor
llvm.func @test_dynamic_loop_body_extra_predecessor(%cond: i1) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.cond_br %cond, ^reenter, ^ret

^reenter:
  llvm.br ^store_body

^ret:
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: loop starts at 8 rather than 0, so the lower half of the
// buffer is never written -- not full coverage.
// CHECK-LABEL: llvm.func @test_dynamic_loop_nonzero_lower_bound
llvm.func @test_dynamic_loop_nonzero_lower_bound() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %eight = llvm.mlir.constant(8 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%eight : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the loop step (4) is larger than the vector lane count (2),
// leaving holes between successive store tiles.
// CHECK-LABEL: llvm.func @test_dynamic_loop_step_exceeds_vector
llvm.func @test_dynamic_loop_step_exceeds_vector() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %four = llvm.mlir.constant(4 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %four : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: loop upper bound is 12 while the buffer holds 16 elements;
// the last 4 elements are never written.
// CHECK-LABEL: llvm.func @test_dynamic_loop_upper_bound_short
llvm.func @test_dynamic_loop_upper_bound_short() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %twelve = llvm.mlir.constant(12 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %twelve : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: GEP is indexed by `iv + 1` rather than `iv` directly. The
// recognizer requires the GEP index to be the loop's induction-variable
// block argument so the per-iteration access tile is known.
// CHECK-LABEL: llvm.func @test_dynamic_loop_indirect_gep_index
llvm.func @test_dynamic_loop_indirect_gep_index() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %offset_idx = llvm.add %i, %one : i32
  %store_ptr = llvm.getelementptr %narrow[%offset_idx] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the loop guard uses `sle`, which the recognizer intentionally
// does not match. (Although `iv <= 14` with step 2 from 0 happens to cover
// the same elements as `iv < 16`, keeping the recognizer to strict `<` /
// `>` avoids reasoning about boundary conditions.)
// CHECK-LABEL: llvm.func @test_dynamic_loop_sle_predicate
llvm.func @test_dynamic_loop_sle_predicate() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %fourteen = llvm.mlir.constant(14 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "sle" %i, %fourteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the store loop iterates with a non-constant upper bound, so
// the recognizer cannot determine that the loop covers the buffer.
// CHECK-LABEL: llvm.func @test_dynamic_loop_non_constant_upper_bound
llvm.func @test_dynamic_loop_non_constant_upper_bound(%bound: i32) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %bound : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: loop step is non-constant. The recognizer requires a known
// constant step to compute coverage per iteration.
// CHECK-LABEL: llvm.func @test_dynamic_loop_non_constant_step
llvm.func @test_dynamic_loop_non_constant_step(%step: i32) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %step : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Safe case: unsigned-less-than (`ult`) loop guard. The recognizer accepts
// both signed and unsigned strict less-than predicates.
// CHECK-LABEL: llvm.func @test_dynamic_loop_ult_predicate
llvm.func @test_dynamic_loop_ult_predicate() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "ult" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// Safe case: unsigned-greater-than (`ugt`) loop guard with swapped operands.
// Equivalent to `iv ult bound`; the recognizer accepts both operand orderings
// for the unsigned predicate.
// CHECK-LABEL: llvm.func @test_dynamic_loop_ugt_swapped_predicate
llvm.func @test_dynamic_loop_ugt_swapped_predicate() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "ugt" %sixteen, %i : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the load lives inside the loop's header block (before the
// cond_br terminator). At load time the loop may not yet have completed any
// iteration that writes the index being loaded, so the buffer is only
// partially covered. The pass must not fire.
// CHECK-LABEL: llvm.func @test_dynamic_loop_load_in_header
llvm.func @test_dynamic_loop_load_in_header() {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^header(%zero : i32)

^header(%i: i32):
  %load_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  %sum = llvm.fadd %extended, %wide : vector<2xf32>
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^body, ^done

^body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^header(%next_i : i32)

^done:
  llvm.return
}

// Unsafe case: the alloca holds 15 f16 elements but the store loop's vector
// tile is 2 lanes wide, so the per-iteration access tile (2 elements) does
// not divide the buffer size. The recognizer rejects this to avoid reasoning
// about partial tiles at the buffer tail.
// CHECK-LABEL: llvm.func @test_dynamic_loop_buffer_not_multiple_of_tile
llvm.func @test_dynamic_loop_buffer_not_multiple_of_tile() -> vector<2xf32> {
  %size = llvm.mlir.constant(15 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %fifteen = llvm.mlir.constant(15 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %fifteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Safe case: the loop header has two non-body predecessors that pass the same
// constant initial value. The recognizer accepts this since the lower bound is
// consistent across all entry paths.
// CHECK-LABEL: llvm.func @test_dynamic_loop_multiple_entries_same_lower_bound
llvm.func @test_dynamic_loop_multiple_entries_same_lower_bound(%cond: i1) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.cond_br %cond, ^path1, ^path2

^path1:
  llvm.br ^store_header(%zero : i32)

^path2:
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the loop header has two non-body predecessors that pass
// different constant initial values (0 vs 2). The recognizer requires a
// consistent lower bound and rejects this case.
// CHECK-LABEL: llvm.func @test_dynamic_loop_multiple_entries_conflicting_lower_bound
llvm.func @test_dynamic_loop_multiple_entries_conflicting_lower_bound(%cond: i1) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.cond_br %cond, ^path1, ^path2

^path1:
  llvm.br ^store_header(%zero : i32)

^path2:
  llvm.br ^store_header(%two : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Safe case (known pessimism): the writer loop body contains both the narrow
// fptrunc store and a pre-existing parallel wide store of the same wide value.
// `selectConsistentWideBuffers` uses op-level dominance, which rejects the
// pre-existing in-loop wide store as a reuse candidate. The pass still
// transforms the load correctly by allocating a fresh wide buffer next to the
// narrow one, leaving the original pre-existing wide alloca/store in the IR
// as dead state. Documented in `selectConsistentWideBuffers`.
// CHECK-LABEL: llvm.func @test_dynamic_loop_with_preexisting_wide_store
llvm.func @test_dynamic_loop_with_preexisting_wide_store() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // The pre-existing parallel wide alloca; pessimistically left in the IR.
  %wide_buf = llvm.alloca %size x f32 : (i64) -> !llvm.ptr<5>
  // The pass allocates a fresh wide buffer rather than reusing %wide_buf.
  // CHECK: llvm.alloca {{.*}} x f32
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %narrow_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %narrow_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %wide_ptr = llvm.getelementptr %wide_buf[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f32
  llvm.store %wide, %wide_ptr {alignment = 4 : i64} : vector<2xf32>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the per-iteration access tile is wider than the loop step, so
// successive iterations overlap rather than tile the buffer. Although the
// union of all writes still covers [0, 16), the recognizer intentionally
// rejects mismatched step/tile because the "last write wins" semantics that
// keep the wide buffer in sync depend on a clean tiling.
// CHECK-LABEL: llvm.func @test_dynamic_loop_step_smaller_than_vector
llvm.func @test_dynamic_loop_step_smaller_than_vector() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %one = llvm.mlir.constant(1 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %one : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Safe case: the post-loop load has no GEP at all (loads directly from the
// alloca base pointer). The dynamic-loop coverage proof comes from the
// writer-loop's GEP, not the load's, so the load shape is irrelevant to
// recognizing coverage. The pass must still rewrite the load to read from a
// fresh wide buffer.
// CHECK-LABEL: llvm.func @test_dynamic_loop_load_without_gep
llvm.func @test_dynamic_loop_load_without_gep() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %loaded = llvm.load %narrow {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_step_first_operand
llvm.func @test_dynamic_loop_step_first_operand() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %two, %i : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_body_cond_br_terminator
llvm.func @test_dynamic_loop_body_cond_br_terminator() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep_storing = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep_storing, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  %recheck = llvm.icmp "slt" %next_i, %sixteen : i32
  llvm.cond_br %recheck, ^store_header(%next_i : i32), ^after_loop

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// Unsafe case: the dynamic index into the narrow buffer is the function-
// argument induction variable of the entry block. The recognizer's "header"
// is then the entry block, whose terminator is `llvm.return` rather than a
// `cond_br`, so the counted-loop match fails immediately.
// CHECK-LABEL: llvm.func @test_dynamic_loop_iv_is_function_arg
llvm.func @test_dynamic_loop_iv_is_function_arg(%iv: i32) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%iv] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_cond_not_icmp
llvm.func @test_dynamic_loop_cond_not_icmp(%dyn_cond: i1) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  llvm.cond_br %dyn_cond, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_zero_step
llvm.func @test_dynamic_loop_zero_step() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %zero : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_latch_not_add
llvm.func @test_dynamic_loop_latch_not_add() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %neg_two = llvm.mlir.constant(-2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.sub %i, %neg_two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_non_constant_initial
llvm.func @test_dynamic_loop_non_constant_initial(%init: i32) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%init : i32)

^store_header(%i: i32):
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_dynamic_alloca_size
llvm.func @test_dynamic_loop_dynamic_alloca_size(%dyn_size: i64) -> vector<2xf32> {
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %dyn_size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_two_buffers_shared_iv
llvm.func @test_dynamic_loop_two_buffers_shared_iv() {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow1 = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  %narrow2 = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // Both narrow allocas are replaced by fresh wide buffers.
  // CHECK: llvm.alloca {{.*}} x f32
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^store_body, ^after_loop

^store_body:
  %narrow_value1 = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %ptr1 = llvm.getelementptr %narrow1[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value1, %ptr1 {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %narrow_value2 = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %ptr2 = llvm.getelementptr %narrow2[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value2, %ptr2 {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr1 = llvm.getelementptr %narrow1[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded1 = llvm.load %load_ptr1 {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %ext1 = llvm.fpext %loaded1 : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  %load_ptr2 = llvm.getelementptr %narrow2[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded2 = llvm.load %load_ptr2 {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %ext2 = llvm.fpext %loaded2 : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  %sum = llvm.fadd %ext1, %ext2 : vector<2xf32>
  llvm.return
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_header_as_second_successor
llvm.func @test_dynamic_loop_header_as_second_successor(%cond: i1) -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %zero = llvm.mlir.constant(0 : i32) : i32
  %two = llvm.mlir.constant(2 : i32) : i32
  %sixteen = llvm.mlir.constant(16 : i32) : i32
  %wide = llvm.mlir.constant(dense<1.000000e+00> : vector<2xf32>) : vector<2xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK: llvm.alloca {{.*}} x f32
  llvm.cond_br %cond, ^preheader, ^store_header(%zero : i32)

^preheader:
  llvm.br ^store_header(%zero : i32)

^store_header(%i: i32):
  %keep = llvm.icmp "slt" %i, %sixteen : i32
  llvm.cond_br %keep, ^store_body, ^after_loop

^store_body:
  %narrow_value = llvm.fptrunc %wide : vector<2xf32> to vector<2xf16>
  %store_ptr = llvm.getelementptr %narrow[%i] : (!llvm.ptr<5>, i32) -> !llvm.ptr<5>, f16
  llvm.store %narrow_value, %store_ptr {alignment = 2 : i64} : vector<2xf16>, !llvm.ptr<5>
  %next_i = llvm.add %i, %two : i32
  llvm.br ^store_header(%next_i : i32)

^after_loop:
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK-NOT: llvm.fpext
  // CHECK: llvm.load {{.*}} -> vector<2xf32>
  llvm.return %extended : vector<2xf32>
}

// CHECK-LABEL: llvm.func @test_dynamic_loop_gep_elem_bitwidth_mismatch
llvm.func @test_dynamic_loop_gep_elem_bitwidth_mismatch() -> vector<2xf32> {
  %size = llvm.mlir.constant(16 : i64) : i64
  %wide3 = llvm.mlir.constant(dense<1.000000e+00> : vector<3xf32>) : vector<3xf32>
  %narrow = llvm.alloca %size x f16 : (i64) -> !llvm.ptr<5>
  // CHECK-NOT: llvm.alloca {{.*}} x f32
  // GEP elem type intentionally `f32` so that vector<3xf16> (48 bits) does
  // not tile the 32-bit element cleanly.
  %store_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f32
  %v3 = llvm.fptrunc %wide3 : vector<3xf32> to vector<3xf16>
  llvm.store %v3, %store_ptr {alignment = 2 : i64} : vector<3xf16>, !llvm.ptr<5>
  %load_ptr = llvm.getelementptr %narrow[0] : (!llvm.ptr<5>) -> !llvm.ptr<5>, f16
  %loaded = llvm.load %load_ptr {alignment = 2 : i64} : !llvm.ptr<5> -> vector<2xf16>
  %extended = llvm.fpext %loaded : vector<2xf16> to vector<2xf32>
  // CHECK: llvm.fpext
  llvm.return %extended : vector<2xf32>
}
