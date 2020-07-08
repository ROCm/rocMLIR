// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file | FileCheck %s --dump-input-on-failure

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index) {
    // CHECK: rocdl.workitem.id.x : !llvm.i32
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workitem.id.y : !llvm.i32
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workitem.id.z : !llvm.i32
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.dim.x : !llvm.i32
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.y : !llvm.i32
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.z : !llvm.i32
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.id.x : !llvm.i32
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.id.y : !llvm.i32
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.id.z : !llvm.i32
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.grid.dim.x : !llvm.i32
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.grid.dim.y : !llvm.i32
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.grid.dim.z : !llvm.i32
    %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)

    std.return %tIdX, %tIdY, %tIdZ, %bDimX, %bDimY, %bDimZ,
               %bIdX, %bIdY, %bIdZ, %gDimX, %gDimY, %gDimZ
        : index, index, index, index, index, index,
          index, index, index, index, index, index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_sync()
  func @gpu_sync() {
    // CHECK: rocdl.barrier
    gpu.barrier
    std.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_fabs_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_fabs_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_fabs
  func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.absf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_fabs_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.absf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_fabs_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_ceil_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_ceil_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_ceil
  func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.ceilf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_ceil_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.ceilf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_ceil_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_cos_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_cos_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_cos
  func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.cos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cos_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.cos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cos_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----
gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_exp_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_exp
  func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %exp_f32 = std.exp %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result32 = std.exp %exp_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}


// -----

// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module {
  "test.symbol_scope"() ({
    // CHECK: test.symbol_scope
    // CHECK: llvm.func @__ocml_exp_f32(!llvm.float) -> !llvm.float
    // CHECK: llvm.func @__ocml_exp_f64(!llvm.double) -> !llvm.double
    // CHECK-LABEL: func @gpu_exp
    func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %exp_f32 = std.exp %arg_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
      %result32 = std.exp %exp_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
      %result64 = std.exp %arg_f64 : f64
      // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
      std.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_log_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log
  func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log10_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_log10_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log10
  func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log10 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log10_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log10 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log10_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log2_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_log2_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log2
  func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log2_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log2_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_tanh_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_tanh_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_tanh
  func @gpu_tanh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.tanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tanh_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.tanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tanh_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module_mfma_f32 {
  // CHECK-LABEL: func @mfma_f32
  func @mfma_f32(%arg0: f32, %arg1: f32, %arg2: memref<64xf32>) {
    %0 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0 = constant 0 : index
    %1 = load %0[%c0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %2, %0[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = load %0[%c1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %4, %0[%c1] : memref<2xvector<32xf32>>

    // ----

    %5 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_0 = constant 0 : index
    %6 = load %5[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %7, %5[%c0_0] : memref<2xvector<32xf32>>

    // ----

    %8 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_1 = constant 0 : index
    %9 = load %8[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %10, %8[%c0_1] : memref<2xvector<32xf32>>

    // ----

    %11 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_2 = constant 0 : index
    %12 = load %11[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2f32"} : f32, vector<16xf32>
    store %13, %11[%c0_2] : memref<4xvector<16xf32>>

    // ----

    %14 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_3 = constant 0 : index
    %15 = load %14[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f32"} : f32, vector<4xf32>
    store %16, %14[%c0_3] : memref<16xvector<4xf32>>

    // ----

    %17 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_4 = constant 0 : index
    %18 = load %17[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    store %19, %17[%c0_4] : memref<4xvector<16xf32>>

    // ----

    %20 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_5 = constant 0 : index
    %21 = load %20[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    store %22, %20[%c0_5] : memref<4xvector<16xf32>>

    // ----

    %23 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_6 = constant 0 : index
    %24 = load %23[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    store %25, %23[%c0_6] : memref<16xvector<4xf32>>

    // ----

    %26 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_7 = constant 0 : index
    %27 = load %26[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    store %28, %26[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = load %26[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm.float, !llvm.float, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    store %30, %26[%c1_8] : memref<16xvector<4xf32>>
    return
  }
}

// -----

gpu.module @test_module_mfma_f16 {
  // CHECK-LABEL: func @mfma_f16
  func @mfma_f16(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<64xf32>) {
    %0 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0 = constant 0 : index
    %1 = load %0[%c0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %2, %0[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = load %0[%c1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %4, %0[%c1] : memref<2xvector<32xf32>>

    // ----

    %5 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_0 = constant 0 : index
    %6 = load %5[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %7, %5[%c0_0] : memref<2xvector<32xf32>>

    // ----

    %8 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_1 = constant 0 : index
    %9 = load %8[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %10, %8[%c0_1] : memref<2xvector<32xf32>>

    // ----

    %11 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_2 = constant 0 : index
    %12 = load %11[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x8f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x8f16"} : vector<4xf16>, vector<16xf32>
    store %13, %11[%c0_2] : memref<4xvector<16xf32>>

    // ----

    %14 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_3 = constant 0 : index
    %15 = load %14[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x16f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x16f16"} : vector<4xf16>, vector<4xf32>
    store %16, %14[%c0_3] : memref<16xvector<4xf32>>

    // ----

    %17 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_4 = constant 0 : index
    %18 = load %17[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    store %19, %17[%c0_4] : memref<4xvector<16xf32>>

    // ----

    %20 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_5 = constant 0 : index
    %21 = load %20[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    store %22, %20[%c0_5] : memref<4xvector<16xf32>>

    // ----

    %23 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_6 = constant 0 : index
    %24 = load %23[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    store %25, %23[%c0_6] : memref<16xvector<4xf32>>

    // ----

    %26 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_7 = constant 0 : index
    %27 = load %26[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    store %28, %26[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = load %26[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<4 x half>">, !llvm<"<4 x half>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    store %30, %26[%c1_8] : memref<16xvector<4xf32>>
    return
  }
}

// ----

gpu.module @test_module_mfma_bf16 {
  // CHECK-LABEL: func @mfma_bf16
  func @mfma_bf16(%arg0: vector<2xbf16>, %arg1: vector<2xbf16>, %arg2: memref<64xf32>) {
    %0 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0 = constant 0 : index
    %1 = load %0[%c0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>
    store %2, %0[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = load %0[%c1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>
    store %4, %0[%c1] : memref<2xvector<32xf32>>

    // ----

    %5 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_0 = constant 0 : index
    %6 = load %5[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>
    store %7, %5[%c0_0] : memref<2xvector<32xf32>>

    // ----

    %8 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_1 = constant 0 : index
    %9 = load %8[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<32 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<32 x float>">
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>
    store %10, %8[%c0_1] : memref<2xvector<32xf32>>

    // ----

    %11 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_2 = constant 0 : index
    %12 = load %11[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4bf16"} : vector<2xbf16>, vector<16xf32>
    store %13, %11[%c0_2] : memref<4xvector<16xf32>>

    // ----

    %14 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_3 = constant 0 : index
    %15 = load %14[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x8bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x8bf16"} : vector<2xbf16>, vector<4xf32>
    store %16, %14[%c0_3] : memref<16xvector<4xf32>>

    // ----

    %17 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_4 = constant 0 : index
    %18 = load %17[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x2bf16"} : vector<2xbf16>, vector<16xf32>
    store %19, %17[%c0_4] : memref<4xvector<16xf32>>

    // ----

    %20 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_5 = constant 0 : index
    %21 = load %20[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<16 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<16 x float>">
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x2bf16"} : vector<2xbf16>, vector<16xf32>
    store %22, %20[%c0_5] : memref<4xvector<16xf32>>

    // ----

    %23 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_6 = constant 0 : index
    %24 = load %23[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xbf16>, vector<4xf32>
    store %25, %23[%c0_6] : memref<16xvector<4xf32>>

    // ----

    %26 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_7 = constant 0 : index
    %27 = load %26[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xbf16>, vector<4xf32>
    store %28, %26[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = load %26[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (!llvm<"<2 x bfloat>">, !llvm<"<2 x bfloat>">, !llvm<"<4 x float>">, !llvm.i32, !llvm.i32, !llvm.i32) -> !llvm<"<4 x float>">
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xbf16>, vector<4xf32>
    store %30, %26[%c1_8] : memref<16xvector<4xf32>>
    return
  }
} 
