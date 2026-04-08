// RUN: rocmlir-opt --rock-add-sched-group-barriers %s | FileCheck %s

// Unit tests for the rock-add-sched-group-barriers pass operating on
// pre-lowered IR. These target edge cases that are difficult to produce
// through the full kernel pipeline.

// Negative test: conditional code (scf.if) in the loop body -- skipped.
// CHECK-LABEL: func @has_scf_if
// CHECK-NOT: rocdl.iglp.opt
func.func @has_scf_if() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.if %true {
      memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    }
    scf.yield
  }
  return
}

// Positive test: no global loads but has MFMA + double-buffered LDS.
// iglp_opt only requires DS+MFMA pattern.
// CHECK-LABEL: func @no_global_loads
// CHECK: rocdl.iglp.opt 0
func.func @no_global_loads() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.yield
  }
  return
}

// Negative test: no MFMA ops -- skipped.
// CHECK-LABEL: func @no_mfma
// CHECK-NOT: rocdl.iglp.opt
func.func @no_mfma() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    scf.yield
  }
  return
}

// Positive test: single-buffered (no arith.select) -- iglp_opt works for both.
// CHECK-LABEL: func @not_double_buffered
// CHECK: rocdl.iglp.opt 0
func.func @not_double_buffered() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %lds = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %lds[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %lds[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.yield
  }
  return
}

// Positive test: multi-wave GEMM (default, no block_size attr -> variant 0).
// CHECK-LABEL: func @multi_wave_gemm
// CHECK: rocdl.iglp.opt 0
func.func @multi_wave_gemm() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.yield
  }
  return
}

// Positive test: explicit multi-wave (block_size=256 > waveSize=64 on gfx942).
// CHECK-LABEL: func @multi_wave_explicit
// CHECK: rocdl.iglp.opt 0
func.func @multi_wave_explicit() attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx942", block_size = 256 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.yield
  }
  return
}

// Positive test: single-wave GEMM (block_size=64 <= waveSize=64 on gfx942).
// Uses variant 0 (variant 1 triggers LLVM backend assertions).
// CHECK-LABEL: func @single_wave_gemm
// CHECK: rocdl.iglp.opt 0
func.func @single_wave_gemm() attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx942", block_size = 64 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.yield
  }
  return
}

// Positive test: single-wave on RDNA/Navi (block_size=32 <= waveSize=32 on gfx1100).
// Uses variant 0 (variant 1 triggers LLVM backend assertions).
// CHECK-LABEL: func @single_wave_navi
// CHECK: rocdl.iglp.opt 0
func.func @single_wave_navi() attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1100", block_size = 32 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.yield
  }
  return
}

// Positive test: many MFMAs (>25) -- no longer skipped with iglp_opt.
// CHECK-LABEL: func @many_mfma
// CHECK: rocdl.iglp.opt 0
func.func @many_mfma() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    affine.for %j = 0 to 26 {
      %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    }
    scf.yield
  }
  return
}

// Negative test: multiple scf.for loops where one has nesting -- all skipped.
// CHECK-LABEL: func @nested_scf_for
// CHECK-NOT: rocdl.iglp.opt
func.func @nested_scf_for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<0.0> : vector<4xf32>
  %cstf16 = arith.constant dense<0.0> : vector<4xf16>
  %zero = arith.constant 0.0 : f16
  %true = arith.constant true
  %lds0 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %lds1 = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
  %global = memref.alloc() : memref<4096xf16, #gpu.address_space<global>>
  scf.for %iv = %c0 to %c10 step %c1 {
    %sel = arith.select %true, %lds0, %lds1 : memref<1024xf16, #gpu.address_space<workgroup>>
    %g0 = vector.load %global[%c0] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.for %jv = %c0 to %c10 step %c1 {
      scf.yield
    }
    scf.yield
  }
  return
}
