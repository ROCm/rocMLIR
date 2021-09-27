// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-rocdl='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: llvm.func @gpu_index_ops
  // CHECK32-LABEL: llvm.func @gpu_index_ops
  gpu.func @gpu_index_ops(%out : memref<12xindex>)
      -> () {
    // CHECK32-NOT: = llvm.sext %{{.*}} : i32 to i64

    // CHECK: rocdl.workitem.id.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    %c0 = constant 0 : index
    memref.store %tIdX, %out[%c0] : memref<12xindex>
    // CHECK: rocdl.workitem.id.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    %c1 = constant 1 : index
    memref.store %tIdY, %out[%c1] : memref<12xindex>
    // CHECK: rocdl.workitem.id.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)
    %c2 = constant 2 : index
    memref.store %tIdZ, %out[%c2] : memref<12xindex>

    // CHECK: rocdl.workgroup.dim.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    %c3 = constant 3 : index
    memref.store %bDimX, %out[%c3] : memref<12xindex>
    // CHECK: rocdl.workgroup.dim.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    %c4 = constant 4 : index
    memref.store %bDimY, %out[%c4] : memref<12xindex>
    // CHECK: rocdl.workgroup.dim.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)
    %c5 = constant 5 : index
    memref.store %bDimZ, %out[%c5] : memref<12xindex>

    // CHECK: rocdl.workgroup.id.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    %c6 = constant 6 : index
    memref.store %bIdX, %out[%c6] : memref<12xindex>
    // CHECK: rocdl.workgroup.id.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    %c7 = constant 7 : index
    memref.store %bIdY, %out[%c7] : memref<12xindex>
    // CHECK: rocdl.workgroup.id.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)
    %c8 = constant 8 : index
    memref.store %bIdZ, %out[%c8] : memref<12xindex>

    // CHECK: rocdl.grid.dim.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    %c9 = constant 9 : index
    memref.store %gDimX, %out[%c9] : memref<12xindex>
    // CHECK: rocdl.grid.dim.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    %c10 = constant 10 : index
    memref.store %gDimY, %out[%c10] : memref<12xindex>
    // CHECK: rocdl.grid.dim.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)
    %c11 = constant 11 : index
    memref.store %gDimZ, %out[%c11] : memref<12xindex>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: llvm.func @gpu_index_comp
  // CHECK32-LABEL: llvm.func @gpu_index_comp
  gpu.func @gpu_index_comp(%idx : index) -> index {
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : i64
    // CHECK32: = llvm.add %{{.*}}, %{{.*}} : i32
    %0 = addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : i64
    // CHECK32: llvm.return %{{.*}} : i32
    gpu.return %0 : index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: llvm.func @gpu_sync()
  gpu.func @gpu_sync() {
    // CHECK: rocdl.barrier
    gpu.barrier
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: llvm.func @gpu_lds_sync()
  gpu.func @gpu_lds_sync() {
    // CHECK: rocdl.lds_barrier
    gpu.lds_barrier
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_fabs_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_fabs_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_fabs
  gpu.func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = std.absf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_fabs_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.absf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_fabs_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_ceil_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_ceil_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_ceil
  gpu.func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = std.ceilf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_ceil_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.ceilf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_ceil_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_floor_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_floor_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_floor
  gpu.func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = std.floorf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_floor_f32(%{{.*}}) : (f32) -> f32
    %result64 = std.floorf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_floor_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_cos_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cos_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_cos
  gpu.func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.cos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cos_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.cos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cos_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_exp
  gpu.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %exp_f32 = math.exp %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.exp %exp_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp2_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_exp2
  gpu.func @gpu_exp2(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %exp2_f32 = math.exp2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp2_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.exp2 %exp2_f32 : f32
    // CHECK: llvm.call @__ocml_exp2_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp2_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_exp
  gpu.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %exp_f32 = math.exp %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.exp %exp_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_expm1_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_expm1_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_expm1
  gpu.func @gpu_expm1(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %expm1_f32 = math.expm1 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_expm1_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.expm1 %expm1_f32 : f32
    // CHECK: llvm.call @__ocml_expm1_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_expm1_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_log
  gpu.func @gpu_log(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.log %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log1p_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log1p_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_log1p
  gpu.func @gpu_log1p(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.log1p %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log1p_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log1p %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log1p_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log10_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log10_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_log10
  gpu.func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.log10 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log10_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log10 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log10_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log2_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_log2
  gpu.func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.log2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log2_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log2_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_rsqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_rsqrt_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_rsqrt
  gpu.func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf16>, %out1 : memref<?xf32>, %out2 : memref<?xf64>) -> () {
    %result16 = math.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_rsqrt_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result16, %out0[%c0] : memref<?xf16>
    memref.store %result32, %out1[%c0] : memref<?xf32>
    memref.store %result64, %out2[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_sqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_sqrt_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_sqrt
  gpu.func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf16>, %out1 : memref<?xf32>, %out2 : memref<?xf64>) -> () {
    %result16 = math.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_sqrt_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result16, %out0[%c0] : memref<?xf16>
    memref.store %result32, %out1[%c0] : memref<?xf32>
    memref.store %result64, %out2[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_tanh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_tanh_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_tanh
  gpu.func @gpu_tanh(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.tanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tanh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.tanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tanh_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_atan_f64(f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_atan
  gpu.func @gpu_atan(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.atan %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.atan %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan_f64(%{{.*}}) : (f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan2_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_atan2_f64(f64, f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_atan2
  gpu.func @gpu_atan2(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.atan2 %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan2_f32(%{{.*}}) : (f32, f32) -> f32
    %result64 = math.atan2 %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan2_f64(%{{.*}}) : (f64, f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_pow_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_pow_f64(f64, f64) -> f64
  // CHECK-LABEL: llvm.func @gpu_pow
  gpu.func @gpu_pow(%arg_f32 : f32, %arg_f64 : f64, %out0 : memref<?xf32>, %out1 : memref<?xf64>) -> () {
    %result32 = math.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_pow_f32(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_pow_f64(%{{.*}}, %{{.*}}) : (f64, f64) -> f64

    %c0 = constant 0 : index
    memref.store %result32, %out0[%c0] : memref<?xf32>
    memref.store %result64, %out1[%c0] : memref<?xf64>

    gpu.return
  }
}

// -----

gpu.module @test_module_mfma_f32 {
  // CHECK-LABEL: llvm.func @mfma_f32_0
  gpu.func @mfma_f32_0(%arg0: f32, %arg1: f32, %arg2: memref<2xvector<32xf32>>) {
    %c0 = constant 0 : index
    %1 = memref.load %arg2[%c0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    memref.store %2, %arg2[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = memref.load %arg2[%c1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    memref.store %4, %arg2[%c1] : memref<2xvector<32xf32>>
    gpu.return
  }

  // ----

  // CHECK-LABEL: llvm.func @mfma_f32_1
  gpu.func @mfma_f32_1(%arg0: f32, %arg1: f32, %arg2: memref<2xvector<32xf32>>) {
    %c0_0 = constant 0 : index
    %6 = memref.load %arg2[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    memref.store %7, %arg2[%c0_0] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f32_2
  gpu.func @mfma_f32_2(%arg0: f32, %arg1: f32, %arg2: memref<2xvector<32xf32>>) {
    %c0_1 = constant 0 : index
    %9 = memref.load %arg2[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    memref.store %10, %arg2[%c0_1] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f32_3
  gpu.func @mfma_f32_3(%arg0: f32, %arg1: f32, %arg2: memref<4xvector<16xf32>>) {
    %c0_2 = constant 0 : index
    %12 = memref.load %arg2[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2f32"} : f32, vector<16xf32>
    memref.store %13, %arg2[%c0_2] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f32_4
  gpu.func @mfma_f32_4(%arg0: f32, %arg1: f32, %arg2: memref<16xvector<4xf32>>) {
    %c0_3 = constant 0 : index
    %15 = memref.load %arg2[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f32"} : f32, vector<4xf32>
    memref.store %16, %arg2[%c0_3] : memref<16xvector<4xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f32_5
  gpu.func @mfma_f32_5(%arg0: f32, %arg1: f32, %arg2: memref<4xvector<16xf32>>) {
    %c0_4 = constant 0 : index
    %18 = memref.load %arg2[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    memref.store %19, %arg2[%c0_4] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f32_6
  gpu.func @mfma_f32_6(%arg0: f32, %arg1: f32, %arg2: memref<4xvector<16xf32>>) {
    %c0_5 = constant 0 : index
    %21 = memref.load %arg2[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    memref.store %22, %arg2[%c0_5] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f32_7
  gpu.func @mfma_f32_7(%arg0: f32, %arg1: f32, %arg2: memref<16xvector<4xf32>>) {
    %c0_6 = constant 0 : index
    %24 = memref.load %arg2[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    memref.store %25, %arg2[%c0_6] : memref<16xvector<4xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f32_8
  gpu.func @mfma_f32_8(%arg0: f32, %arg1: f32, %arg2: memref<16xvector<4xf32>>) {
    %c0_7 = constant 0 : index
    %27 = memref.load %arg2[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    memref.store %28, %arg2[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = memref.load %arg2[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x1f32 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    memref.store %30, %arg2[%c1_8] : memref<16xvector<4xf32>>
    gpu.return
  }
}

// -----

gpu.module @test_module_mfma_f16 {
  // CHECK-LABEL: llvm.func @mfma_f16_0
  gpu.func @mfma_f16_0(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<2xvector<32xf32>>) {
    %c0 = constant 0 : index
    %1 = memref.load %arg2[%c0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    memref.store %2, %arg2[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = memref.load %arg2[%c1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    memref.store %4, %arg2[%c1] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_1
  gpu.func @mfma_f16_1(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<2xvector<32xf32>>) {
    %c0_0 = constant 0 : index
    %6 = memref.load %arg2[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    memref.store %7, %arg2[%c0_0] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_2
  gpu.func @mfma_f16_2(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<2xvector<32xf32>>) {
    %c0_1 = constant 0 : index
    %9 = memref.load %arg2[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    memref.store %10, %arg2[%c0_1] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_3
  gpu.func @mfma_f16_3(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<4xvector<16xf32>>) {
    %c0_2 = constant 0 : index
    %12 = memref.load %arg2[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x8f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x8f16"} : vector<4xf16>, vector<16xf32>
    memref.store %13, %arg2[%c0_2] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_4
  gpu.func @mfma_f16_4(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<16xvector<4xf32>>) {
    %c0_3 = constant 0 : index
    %15 = memref.load %arg2[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x16f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x16f16"} : vector<4xf16>, vector<4xf32>
    memref.store %16, %arg2[%c0_3] : memref<16xvector<4xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_5
  gpu.func @mfma_f16_5(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<4xvector<16xf32>>) {
    %c0_4 = constant 0 : index
    %18 = memref.load %arg2[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    memref.store %19, %arg2[%c0_4] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_6
  gpu.func @mfma_f16_6(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<4xvector<16xf32>>) {
    %c0_5 = constant 0 : index
    %21 = memref.load %arg2[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    memref.store %22, %arg2[%c0_5] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_7
  gpu.func @mfma_f16_7(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<16xvector<4xf32>>) {
    %c0_6 = constant 0 : index
    %24 = memref.load %arg2[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    memref.store %25, %arg2[%c0_6] : memref<16xvector<4xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_f16_8
  gpu.func @mfma_f16_8(%arg0: vector<4xf16>, %arg1: vector<4xf16>, %arg2: memref<16xvector<4xf32>>) {
    %c0_7 = constant 0 : index
    %27 = memref.load %arg2[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    memref.store %28, %arg2[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = memref.load %arg2[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x4f16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    memref.store %30, %arg2[%c1_8] : memref<16xvector<4xf32>>
    gpu.return
  }
}

// ----

gpu.module @test_module_mfma_bf16 {
  // CHECK-LABEL: llvm.func @mfma_bf16_0
  gpu.func @mfma_bf16_0(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<2xvector<32xf32>>) {
    %c0 = constant 0 : index
    %1 = memref.load %arg2[%c0] : memref<2xvector<32xf32>>
    // CHECK:           %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:      %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:      %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %2 = gpu.mfma(%arg0, %arg1, %1) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    memref.store %2, %arg2[%c0] : memref<2xvector<32xf32>>
    %c1 = constant 1 : index
    %3 = memref.load %arg2[%c1] : memref<2xvector<32xf32>>

    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %4 = gpu.mfma(%arg0, %arg1, %3) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    memref.store %4, %arg2[%c1] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_1
  gpu.func @mfma_bf16_1(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<2xvector<32xf32>>) {
    %c0_0 = constant 0 : index
    %6 = memref.load %arg2[%c0_0] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %7 = gpu.mfma(%arg0, %arg1, %6) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    memref.store %7, %arg2[%c0_0] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_2
  gpu.func @mfma_bf16_2(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<2xvector<32xf32>>) {
    %c0_1 = constant 0 : index
    %9 = memref.load %arg2[%c0_1] : memref<2xvector<32xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<32xf32>, i32, i32, i32) -> vector<32xf32>
    %10 = gpu.mfma(%arg0, %arg1, %9) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x2bf16"} : vector<2xi16>, vector<32xf32>
    memref.store %10, %arg2[%c0_1] : memref<2xvector<32xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_3
  gpu.func @mfma_bf16_3(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<4xvector<16xf32>>) {
    %c0_2 = constant 0 : index
    %12 = memref.load %arg2[%c0_2] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.32x32x4bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %13 = gpu.mfma(%arg0, %arg1, %12) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4bf16"} : vector<2xi16>, vector<16xf32>
    memref.store %13, %arg2[%c0_2] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_4
  gpu.func @mfma_bf16_4(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<16xvector<4xf32>>) {
    %c0_3 = constant 0 : index
    %15 = memref.load %arg2[%c0_3] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x8bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %16 = gpu.mfma(%arg0, %arg1, %15) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x8bf16"} : vector<2xi16>, vector<4xf32>
    memref.store %16, %arg2[%c0_3] : memref<16xvector<4xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_5
  gpu.func @mfma_bf16_5(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<4xvector<16xf32>>) {
    %c0_4 = constant 0 : index
    %18 = memref.load %arg2[%c0_4] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %19 = gpu.mfma(%arg0, %arg1, %18) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x2bf16"} : vector<2xi16>, vector<16xf32>
    memref.store %19, %arg2[%c0_4] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_6
  gpu.func @mfma_bf16_6(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<4xvector<16xf32>>) {
    %c0_5 = constant 0 : index
    %21 = memref.load %arg2[%c0_5] : memref<4xvector<16xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.16x16x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<16xf32>, i32, i32, i32) -> vector<16xf32>
    %22 = gpu.mfma(%arg0, %arg1, %21) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x2bf16"} : vector<2xi16>, vector<16xf32>
    memref.store %22, %arg2[%c0_5] : memref<4xvector<16xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_7
  gpu.func @mfma_bf16_7(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<16xvector<4xf32>>) {
    %c0_6 = constant 0 : index
    %24 = memref.load %arg2[%c0_6] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %25 = gpu.mfma(%arg0, %arg1, %24) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xi16>, vector<4xf32>
    memref.store %25, %arg2[%c0_6] : memref<16xvector<4xf32>>
    gpu.return
  }

    // ----

  // CHECK-LABEL: llvm.func @mfma_bf16_8
  gpu.func @mfma_bf16_8(%arg0: vector<2xi16>, %arg1: vector<2xi16>, %arg2: memref<16xvector<4xf32>>) {
    %c0_7 = constant 0 : index
    %27 = memref.load %arg2[%c0_7] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %28 = gpu.mfma(%arg0, %arg1, %27) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xi16>, vector<4xf32>
    memref.store %28, %arg2[%c0_7] : memref<16xvector<4xf32>>
    %c1_8 = constant 1 : index
    %29 = memref.load %arg2[%c1_8] : memref<16xvector<4xf32>>
    // CHECK:      %[[IMM0:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-NEXT: %[[IMM1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: %[[IMM2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: %{{.*}} = rocdl.mfma.f32.4x4x2bf16 %{{.*}}, %{{.*}}, %{{.*}}, %[[IMM0]], %[[IMM1]], %[[IMM2]] : (vector<2xi16>, vector<2xi16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
    %30 = gpu.mfma(%arg0, %arg1, %29) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x2bf16"} : vector<2xi16>, vector<4xf32>
    memref.store %30, %arg2[%c1_8] : memref<16xvector<4xf32>>
    gpu.return
  }
} 
gpu.module @test_module {
  // CHECK-LABEL: @kernel_func
  // CHECK: attributes
  // CHECK: gpu.kernel
  // CHECK: rocdl.kernel
  gpu.func @kernel_func() kernel {
    gpu.return
  }
}
