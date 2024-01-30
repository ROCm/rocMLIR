// RUN: mlir-opt -convert-amdgpu-to-rocdl=chipset=gfx908 %s | FileCheck %s
// RUN: mlir-opt -convert-amdgpu-to-rocdl=chipset=gfx90a %s | FileCheck %s
// RUN: mlir-opt -convert-amdgpu-to-rocdl=chipset=gfx942 %s | FileCheck %s
  
func.func @test_dpp(%arg0: i32) -> i32 {
  // CHECK-LABEL: func @test_dpp
  // CHECK: llvm.mlir.constant(257 : i32)
  // CHECK: llvm.mlir.constant(10 : i32)
  // CHECK: llvm.mlir.constant(15 : i32)
  // CHECK: llvm.mlir.constant(false)
  // CHECK: rocdl.mov.dpp %arg0 with %0, %1, %2, %3 : i32
  // CHECK: return %4 : i32
  %0 = amdgpu.dpp %arg0 { row_shl = 0x1 : i32, row_mask = 0xa : i32, bound_ctrl = false } : i32
    return %0 : i32
}

func.func @quad_dpp(%arg0: i32) -> i32 {
  // CHECK-LABEL: func @quad_dpp
  // CHECK: llvm.mlir.constant(145 : i32)
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(true)
  // CHECK: rocdl.mov.dpp %arg0 with %0, %1, %2, %3 : i32
  // CHECK: return %4 : i32
  %0 = amdgpu.dpp %arg0 { quad_perm = [1,0,1,2], row_mask = 0x1 : i32, bank_mask = 0x1 : i32, bound_ctrl = true } : i32
    return %0 : i32
}

func.func @quad_perm_dpp(%arg0: i32) -> i32 {
  // CHECK-LABEL: func @quad_perm_dpp
  // CHECK: llvm.mlir.constant(88 : i32)
  // CHECK: llvm.mlir.constant(15 : i32)
  // CHECK: llvm.mlir.constant(15 : i32)
  // CHECK: llvm.mlir.constant(false)
  // CHECK: rocdl.mov.dpp %arg0 with %0, %1, %2, %3 : i32
  // CHECK:return %4 : i32
  %0 = amdgpu.dpp %arg0 { quad_perm = [0,2,1,1] } : i32
    return %0 : i32
}

func.func @wave_shr_dpp(%arg0: i32) -> i32 {
  // CHECK-LABEL: func @wave_shr_dpp
  // CHECK: llvm.mlir.constant(312 : i32)
  // CHECK: llvm.mlir.constant(10 : i32)
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(true)
  // CHECK: rocdl.mov.dpp %arg0 with %0, %1, %2, %3 : i32
  // CHECK:return %4 : i32
  %0 = amdgpu.dpp %arg0 { wave_shr, row_mask = 0xa : i32, bank_mask = 0x1 : i32, bound_ctrl = true } : i32
    return %0 : i32
}

func.func @row_bcast_dpp(%arg0: i32) -> i32 {
  // CHECK-LABEL: func @row_bcast_dpp
  // CHECK: llvm.mlir.constant(323 : i32)
  // CHECK: llvm.mlir.constant(4 : i32)
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(false)
  // CHECK: rocdl.mov.dpp %arg0 with %0, %1, %2, %3 : i32
  // CHECK: return %4 : i32
  %0 = amdgpu.dpp %arg0 { row_bcast_31, row_mask = 0x4 : i32, bank_mask = 0x1 : i32} : i32
    return %0 : i32
}

func.func @row_bcast_dpp_f32(%arg0: f32) -> f32 {
  // CHECK-LABEL: func @row_bcast_dpp_f32
  // CHECK: llvm.bitcast %arg0 : f32 to i32
  // CHECK: llvm.mlir.constant(322 : i32)
  // CHECK: llvm.mlir.constant(15 : i32)
  // CHECK: llvm.mlir.constant(15 : i32)
  // CHECK: llvm.mlir.constant(true)
  // CHECK: rocdl.mov.dpp %0 with %1, %2, %3, %4 : i32
  // CHECK: llvm.bitcast %5 : i32 to f32
  // CHECK:return %6 : f32
  %0 = amdgpu.dpp %arg0 { row_bcast_15, bound_ctrl = true } : f32
    return %0 : f32
}

func.func @test_dpp_f32(%arg0: f32) -> f32 {
  // CHECK-LABEL: func @test_dpp_f32
  // CHECK: llvm.bitcast %arg0 : f32 to i32
  // CHECK: llvm.mlir.constant(320 : i32)
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(4 : i32)
  // CHECK: llvm.mlir.constant(true)
  // CHECK: rocdl.mov.dpp %0 with %1, %2, %3, %4 : i32
  // CHECK: llvm.bitcast %5 : i32 to f32
  // CHECK: return %6 : f32
  %0 = amdgpu.dpp %arg0 { row_mirror, row_mask = 0x1 : i32, bank_mask = 0x4 : i32, bound_ctrl = true } : f32
    return %0 : f32
}

func.func @test_dpp_f16(%arg0: f16) -> f16 {
  // CHECK-LABEL:  func @test_dpp_f16
  // CHECK: llvm.bitcast %arg0 : f16 to i16
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.insertelement %0, %1[%2 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %3 : vector<2xi16> to i32
  // CHECK: llvm.mlir.constant(273 : i32)
  // CHECK: llvm.mlir.constant(15 : i32)
  // CHECK: llvm.mlir.constant(3 : i32)
  // CHECK: llvm.mlir.constant(false)
  // CHECK: rocdl.mov.dpp %4 with %5, %6, %7, %8 : i32
  // CHECK: llvm.trunc %9 : i32 to i16
  // CHECK: llvm.bitcast %10 : i16 to f16
  // CHECK: return %11 : f16
  %0 = amdgpu.dpp %arg0 { row_shr = 0x1 : i32, bank_mask = 0x3 : i32 } : f16
    return %0 : f16
}

func.func @row_shl_dpp_i16(%arg0: i16) -> i16 {
  // CHECK-LABEL: func @row_shl_dpp_i16
  // CHECK: llvm.mlir.undef : vector<2xi16>
  // CHECK: llvm.mlir.constant(0 : i32) : i32
  // CHECK: llvm.insertelement %arg0, %0[%1 : i32] : vector<2xi16>
  // CHECK: llvm.bitcast %2 : vector<2xi16> to i32
  // CHECK: llvm.mlir.constant(298 : i32)
  // CHECK: llvm.mlir.constant(10 : i32)
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(false)
  // CHECK: rocdl.mov.dpp %3 with %4, %5, %6, %7 : i32
  // CHECK: llvm.trunc %8 : i32 to i16
  // CHECK: return %9 : i16
  %0 = amdgpu.dpp %arg0 { row_ror = 0xa : i32, row_mask = 0xa : i32, bank_mask = 0x1 : i32 } : i16
    return %0 : i16
}
