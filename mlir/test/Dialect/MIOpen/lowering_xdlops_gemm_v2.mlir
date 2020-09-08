// XFAIL: *
// RUN: mlir-opt -miopen-lowering-step4 %s | FileCheck %s

func @miopen_xdlops_gemm_v2(%A : memref<12288xf32, 3>, %B : memref<12288xf32, 3>, %C : memref<128xf32, 5>) {
  %c0 = constant 0 : index
  %c_64 = constant 64 : index

  %d1 = miopen.xdlops_gemm_v2(%A, %B, %C, %c0, %c0, %c_64) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    instr = "mfma_f32_32x32x1f32",
    imm = [1, 1, 0],
    coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0) -> (d0 + 8192)>]}, {operand = 0 : i32, transforms = []}]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, memref<128xf32, 5>, index, index, index, vector<32xf32>
  // CHECK: %[[V0:.*]] = vector.type_cast %arg2 : memref<128xf32, 5> to memref<4xvector<32xf32>, 5>
  // CHECK: %[[C0:.*]] = load %[[V0]][%{{.*}}] : memref<4xvector<32xf32>, 5>
  // CHECK: %[[M0:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[C0]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M1:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M0]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M2:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M1]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M3:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M2]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M4:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M3]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M5:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M4]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M6:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M5]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M7:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M6]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M8:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M7]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M9:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M8]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M10:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M9]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M11:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M10]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M12:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M11]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M13:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M12]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M14:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M13]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M15:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M14]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M16:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M15]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M17:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M16]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M18:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M17]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M19:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M18]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M20:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M19]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M21:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M20]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M22:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M21]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M23:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M22]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M24:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M23]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M25:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M24]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M26:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M25]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M27:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M26]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M28:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M27]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M29:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M28]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M30:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M29]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
  // CHECK: %[[M31:.*]] = miopen.mfma_v2(%{{.*}}, %{{.*}}, %[[M30]]) {imm = [1, 1, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
 
  return
}
