// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx940 | FileCheck %s

// CHECK-LABEL: packed_trunc
// CHECK-SAME: ([[value:.*]]: f32)
func.func @packed_trunc(%v: f32) -> vector<2xf16> {
  // CHECK: %[[undef:.*]] = llvm.mlir.undef : f32
  // CHECK: %[[packed:.*]] = rocdl.cvt.pkrtz %arg0, %[[undef]] : vector<2xf16>
  // CHECK: return %[[packed]]
  %ret = amdgpu.packed_trunc_fp16x2 %v, undef : f32 to vector<2xf16>
  func.return %ret : vector<2xf16>
}

// CHECK-LABEL: packed_truncx2
// CHECK-SAME: ([[value0:.*]]: f32, [[value1:.*]]: f32)
func.func @packed_truncx2(%v: f32, %w: f32) -> vector<2xf16> {
  // CHECK: %[[packed:.*]] = rocdl.cvt.pkrtz %arg0, %arg1 : vector<2xf16>
  // CHECK: return %[[packed]]
  %ret = amdgpu.packed_trunc_fp16x2 %v, %w : f32 to vector<2xf16>
  func.return %ret : vector<2xf16>
}
