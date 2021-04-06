// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-rocdl='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  // CHECK32-LABEL: func @gpu_index_ops()
  func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index) {
    // CHECK32-NOT: = llvm.sext %{{.*}} : i32 to i64

    // CHECK: rocdl.workitem.id.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workitem.id.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workitem.id.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.dim.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.id.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.id.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.id.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.grid.dim.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.grid.dim.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.grid.dim.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)

    std.return %tIdX, %tIdY, %tIdZ, %bDimX, %bDimY, %bDimZ,
               %bIdX, %bIdY, %bIdZ, %gDimX, %gDimY, %gDimZ
        : index, index, index, index, index, index,
          index, index, index, index, index, index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_comp
  // CHECK32-LABEL: func @gpu_index_comp
  func @gpu_index_comp(%idx : index) -> index {
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : i64
    // CHECK32: = llvm.add %{{.*}}, %{{.*}} : i32
    %0 = addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : i64
    // CHECK32: llvm.return %{{.*}} : i32
    std.return %0 : index
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
  // CHECK-LABEL: func @gpu_lds_sync()
  func @gpu_lds_sync() {
    // CHECK: rocdl.lds_barrier
    gpu.lds_barrier
    std.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_fabs_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_fabs_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_fabs
  func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.absf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_fabs_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.absf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_fabs_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_ceil_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_ceil_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_ceil
  func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.ceilf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_ceil_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.ceilf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_ceil_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_floor_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_floor_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_floor
  func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.floorf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_floor_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.floorf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_floor_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_cos_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cos_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_cos
  func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.cos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cos_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.cos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cos_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----
gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_exp
  func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %exp_f32 = std.exp %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result32 = std.exp %exp_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}


// -----

// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module {
  "test.symbol_scope"() ({
    // CHECK: test.symbol_scope
    // CHECK: llvm.func @__ocml_exp_f32(f32) -> f32
    // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
    // CHECK-LABEL: func @gpu_exp
    func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %exp_f32 = std.exp %arg_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
      %result32 = std.exp %exp_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
      %result64 = std.exp %arg_f64 : f64
      // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64
      std.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log
  func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.log %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log1p_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log1p_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log1p
  func @gpu_log1p(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log1p %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log1p_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.log1p %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log1p_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log10_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log10_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log10
  func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log10 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log10_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.log10 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log10_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log2_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log2
  func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log2_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.log2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log2_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_rsqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_rsqrt_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_rsqrt
  func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = std.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = std.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_rsqrt_f64(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_sqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_sqrt_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_sqrt
  func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = std.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = std.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_sqrt_f64(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_tanh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_tanh_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_tanh
  func @gpu_tanh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.tanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tanh_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.tanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tanh_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_atan_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_atan
  func @gpu_atan(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.atan %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.atan %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan2_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_atan2_f64(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_atan2
  func @gpu_atan2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.atan2 %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan2_f32(%{{.*}}) : (f32, f32) -> f32
    %result64 = std.atan2 %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan2_f64(%{{.*}}) : (f64, f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_pow_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_pow_f64(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_pow
  func @gpu_pow(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_pow_f32(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = std.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_pow_f64(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
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
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %2, %0[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = load %0[%c1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %4, %0[%c1] : memref<2xvector<32xf32>>

    // ----

    %5 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_0 = constant 0 : index
    %6 = load %5[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %7, %5[%c0_0] : memref<2xvector<32xf32>>

    // ----

    %8 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_1 = constant 0 : index
    %9 = load %8[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    store %10, %8[%c0_1] : memref<2xvector<32xf32>>

    // ----

    %11 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_2 = constant 0 : index
    %12 = load %11[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2f32"} : f32, vector<16xf32>
    store %13, %11[%c0_2] : memref<4xvector<16xf32>>

    // ----

    %14 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_3 = constant 0 : index
    %15 = load %14[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f32"} : f32, vector<4xf32>
    store %16, %14[%c0_3] : memref<16xvector<4xf32>>

    // ----

    %17 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_4 = constant 0 : index
    %18 = load %17[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    store %19, %17[%c0_4] : memref<4xvector<16xf32>>

    // ----

    %20 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_5 = constant 0 : index
    %21 = load %20[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    store %22, %20[%c0_5] : memref<4xvector<16xf32>>

    // ----

    %23 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_6 = constant 0 : index
    %24 = load %23[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    store %25, %23[%c0_6] : memref<16xvector<4xf32>>

    // ----

    %26 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_7 = constant 0 : index
    %27 = load %26[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    store %28, %26[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = load %26[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
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
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %2, %0[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = load %0[%c1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %4, %0[%c1] : memref<2xvector<32xf32>>

    // ----

    %5 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_0 = constant 0 : index
    %6 = load %5[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %7, %5[%c0_0] : memref<2xvector<32xf32>>

    // ----

    %8 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_1 = constant 0 : index
    %9 = load %8[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    store %10, %8[%c0_1] : memref<2xvector<32xf32>>

    // ----

    %11 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_2 = constant 0 : index
    %12 = load %11[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x8f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x8f16"} : vector<4xf16>, vector<16xf32>
    store %13, %11[%c0_2] : memref<4xvector<16xf32>>

    // ----

    %14 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_3 = constant 0 : index
    %15 = load %14[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x16f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x16f16"} : vector<4xf16>, vector<4xf32>
    store %16, %14[%c0_3] : memref<16xvector<4xf32>>

    // ----

    %17 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_4 = constant 0 : index
    %18 = load %17[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    store %19, %17[%c0_4] : memref<4xvector<16xf32>>

    // ----

    %20 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_5 = constant 0 : index
    %21 = load %20[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    store %22, %20[%c0_5] : memref<4xvector<16xf32>>

    // ----

    %23 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_6 = constant 0 : index
    %24 = load %23[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    store %25, %23[%c0_6] : memref<16xvector<4xf32>>

    // ----

    %26 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_7 = constant 0 : index
    %27 = load %26[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    store %28, %26[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = load %26[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    store %30, %26[%c1_8] : memref<16xvector<4xf32>>
    return
  }
}

// ----

gpu.module @test_module_mfma_bf16 {
  // CHECK-LABEL: func @mfma_bf16
  func @mfma_bf16(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<64xf32>) {
    %0 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0 = constant 0 : index
    %1 = load %0[%c0] : memref<2xvector<32xf32>>
    // CHECK:           %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:      %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    store %2, %0[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = load %0[%c1] : memref<2xvector<32xf32>>

    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    store %4, %0[%c1] : memref<2xvector<32xf32>>

    // ----

    %5 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_0 = constant 0 : index
    %6 = load %5[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    store %7, %5[%c0_0] : memref<2xvector<32xf32>>

    // ----

    %8 = vector.type_cast %arg2 : memref<64xf32> to memref<2xvector<32xf32>>
    %c0_1 = constant 0 : index
    %9 = load %8[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    store %10, %8[%c0_1] : memref<2xvector<32xf32>>

    %11 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_2 = constant 0 : index
    %12 = load %11[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4bf16"} : vector<2xi16>, vector<16xf32>
    store %13, %11[%c0_2] : memref<4xvector<16xf32>>

    // ----

    %14 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_3 = constant 0 : index
    %15 = load %14[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x8bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x8bf16"} : vector<2xi16>, vector<4xf32>
    store %16, %14[%c0_3] : memref<16xvector<4xf32>>

    // ----

    %17 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_4 = constant 0 : index
    %18 = load %17[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x2bf16"} : vector<2xi16>, vector<16xf32>
    store %19, %17[%c0_4] : memref<4xvector<16xf32>>

    // ----

    %20 = vector.type_cast %arg2 : memref<64xf32> to memref<4xvector<16xf32>>
    %c0_5 = constant 0 : index
    %21 = load %20[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x2bf16"} : vector<2xi16>, vector<16xf32>
    store %22, %20[%c0_5] : memref<4xvector<16xf32>>

    // ----

    %23 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_6 = constant 0 : index
    %24 = load %23[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xi16>, vector<4xf32>
    store %25, %23[%c0_6] : memref<16xvector<4xf32>>

    // ----

    %26 = vector.type_cast %arg2 : memref<64xf32> to memref<16xvector<4xf32>>
    %c0_7 = constant 0 : index
    %27 = load %26[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xi16>, vector<4xf32>
    store %28, %26[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = load %26[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xi16>, vector<4xf32>
    store %30, %26[%c1_8] : memref<16xvector<4xf32>>
    return
  }
} 
