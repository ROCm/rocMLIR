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
