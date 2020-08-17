// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32" %s | FileCheck %s --check-prefix=MFMA_F32
// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16" %s | FileCheck %s --check-prefix=MFMA_F16
// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16" %s | FileCheck %s --check-prefix=MFMA_BF16

module {
  func @mfma_f32(%a : f32, %b : f32, %c : vector<32xf32>) {
    %d0 = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x1f32", imm = [1, 0, 0]}: f32, vector<32xf32>
    // MFMA_F32: %[[D0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) {imm = [1, 0, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    %d1 = miopen.mfma_v2(%a, %b, %d0) { instr = "mfma_f32_32x32x1f32", imm = [1, 0, 0]}: f32, vector<32xf32>
    // MFMA_F32-NEXT: %[[D1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[D0]]) {imm = [1, 0, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>

    return
  }

  // ----

  func @mfma_f16(%a : vector<4xf16>, %b : vector<4xf16>, %c : vector<32xf32>) {
    %d0 = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x4f16", imm = [1, 0, 0]}: vector<4xf16>, vector<32xf32>
    // MFMA_F16: %[[D0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) {imm = [1, 0, 0], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    %d1 = miopen.mfma_v2(%a, %b, %d0) { instr = "mfma_f32_32x32x4f16", imm = [1, 0, 0]}: vector<4xf16>, vector<32xf32>
    // MFMA_F16: %[[D1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[D0]]) {imm = [1, 0, 0], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>

    return
  }

  // ----

  func @mfma_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>, %c : vector<32xf32>) {
    %d0 = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x2bf16", imm = [1, 0, 0]}: vector<2xbf16>, vector<32xf32>
    // MFMA_BF16: %[[D0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) {imm = [1, 0, 0], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>
    %d1 = miopen.mfma_v2(%a, %b, %d0) { instr = "mfma_f32_32x32x2bf16", imm = [1, 0, 0]}: vector<2xbf16>, vector<32xf32>
    // MFMA_BF16: %[[D1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[D0]]) {imm = [1, 0, 0], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>

    return
  }
}
