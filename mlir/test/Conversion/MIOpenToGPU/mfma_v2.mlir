// RUN: miopen-opt -convert-miopen-to-gpu %s | FileCheck %s

module {
  func @mfma_f32(%a : f32, %b : f32, %c : vector<32xf32>) attributes {kernel = 0 : i32} {
    %d0 = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x1f32", imm = [1, 0, 0]}: f32, vector<32xf32>
    // CHECK: %[[D0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) {imm = [1, 0, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    %d1 = miopen.mfma_v2(%a, %b, %d0) { instr = "mfma_f32_32x32x1f32", imm = [1, 0, 0]}: f32, vector<32xf32>
    // CHECK-NEXT: %[[D1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[D0]]) {imm = [1, 0, 0], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>

    return
  }

  // ----

  func @mfma_f16(%a : vector<4xf16>, %b : vector<4xf16>, %c : vector<32xf32>) attributes {kernel = 0 : i32} {
    %d0 = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x4f16", imm = [1, 0, 0]}: vector<4xf16>, vector<32xf32>
    // CHECK: %[[D0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) {imm = [1, 0, 0], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    %d1 = miopen.mfma_v2(%a, %b, %d0) { instr = "mfma_f32_32x32x4f16", imm = [1, 0, 0]}: vector<4xf16>, vector<32xf32>
    // CHECK: %[[D1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[D0]]) {imm = [1, 0, 0], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>

    return
  }

  // ----

  func @mfma_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>, %c : vector<32xf32>) attributes {kernel = 0 : i32} {
    %d0 = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x2bf16", imm = [1, 0, 0]}: vector<2xbf16>, vector<32xf32>
    // CHECK: %[[D0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) {imm = [1, 0, 0], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>
    %d1 = miopen.mfma_v2(%a, %b, %d0) { instr = "mfma_f32_32x32x2bf16", imm = [1, 0, 0]}: vector<2xbf16>, vector<32xf32>
    // CHECK: %[[D1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[D0]]) {imm = [1, 0, 0], instr = "mfma_f32_32x32x2bf16"} : vector<2xbf16>, vector<32xf32>

    return
  }
}
