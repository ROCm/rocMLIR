// RUN: rocmlir-opt --rock-add-sched-group-barriers %s | FileCheck %s

// Unit tests for the rock-add-sched-group-barriers pass operating on
// pre-lowered IR. These target edge cases that are difficult to produce
// through the full kernel pipeline.

// Negative test: >25 matrix multiply ops per iteration -- skipped.
// 26 MFMAs in affine.for 0..26 exceeds the threshold.
// CHECK-LABEL: func @many_mfma
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
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

// Negative test: conditional code (scf.if) in the loop body -- skipped.
// CHECK-LABEL: func @has_scf_if
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
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

// Negative test: no global loads -- skipped.
// CHECK-LABEL: func @no_global_loads
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
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
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
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

// Negative test: not double-buffered (no arith.select on LDS base) -- skipped.
// CHECK-LABEL: func @not_double_buffered
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
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

// Positive test: exactly 1 MFMA with multiple DS ops.
// CHECK-LABEL: func @single_mfma
// CHECK: amdgpu.sched_barrier allow = <none>
// CHECK: rocdl.sched.group.barrier 8, 1, 0
// CHECK: rocdl.sched.group.barrier 512, 3, 0
// CHECK: rocdl.sched.group.barrier 32, 2, 0
// CHECK: rocdl.sched.group.barrier 256, 4, 0
// CHECK: amdgpu.sched_barrier allow = <none>
func.func @single_mfma() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
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
    %g1 = vector.load %global[%c1] : memref<4096xf16, #gpu.address_space<global>>, vector<4xf16>
    memref.store %zero, %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    memref.store %zero, %sel[%c1] : memref<1024xf16, #gpu.address_space<workgroup>>
    memref.store %zero, %sel[%c2] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r0 = memref.load %sel[%c0] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r1 = memref.load %sel[%c1] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r2 = memref.load %sel[%c2] : memref<1024xf16, #gpu.address_space<workgroup>>
    %r3 = memref.load %sel[%c3] : memref<1024xf16, #gpu.address_space<workgroup>>
    %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    scf.yield
  }
  return
}

// Positive test: exactly 25 MFMAs (at the threshold boundary).
// CHECK-LABEL: func @boundary_25_mfma
// CHECK: amdgpu.sched_barrier allow = <none>
// CHECK: rocdl.sched.group.barrier 8, 1, 0
// CHECK: amdgpu.sched_barrier allow = <none>
func.func @boundary_25_mfma() {
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
    affine.for %j = 0 to 25 {
      %m = amdgpu.mfma 16x16x16 %cstf16 * %cstf16 + %cst blgp = none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    }
    scf.yield
  }
  return
}

// Negative test: multiple scf.for loops where one has nesting -- all skipped.
// The nested scf.for causes the entire function to be skipped.
// CHECK-LABEL: func @nested_scf_for
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
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
