// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32" %s | FileCheck %s --check-prefix=MFMA_F32
// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16" %s | FileCheck %s --check-prefix=MFMA_F16

module {
  func @mfma_f32(%a : f32, %b : f32, %c : memref<64xf32>) {
    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 64 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[IT0:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD0:.*]] = load %[[MV]][%[[IT0]]] : memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[MFMA0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD0]]) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    // MFMA_F32-NEXT: store %[[MFMA0]], %[[MV]][%[[IT0]]] : memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[IT1:.*]] = constant 1 : index
    // MFMA_F32-NEXT: %[[LD1:.*]] = load %[[MV]][%[[IT1]]] : memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[MFMA1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD1]]) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    // MFMA_F32-NEXT: store %[[MFMA1]], %[[MV]][%[[IT1]]] : memref<2xvector<32xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 32, n_per_wave = 64 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    // MFMA_F32-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>

    // ----
  
    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 32 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>
    // MFMA_F32-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
    // MFMA_F32-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 32, n_per_wave = 32 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // MFMA_F32-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // MFMA_F32-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2f32"} : f32, vector<16xf32>
    // MFMA_F32-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 16, n_per_wave = 16 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f32"} : f32, vector<4xf32>
    // MFMA_F32-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 16, n_per_wave = 64 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // MFMA_F32-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // MFMA_F32-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    // MFMA_F32-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 16 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // MFMA_F32-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // MFMA_F32-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x1f32"} : f32, vector<16xf32>
    // MFMA_F32-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 4, n_per_wave = 64 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    // MFMA_F32-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 8, n_per_wave = 64 }: f32, memref<64xf32>
    // MFMA_F32:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[IT0:.*]] = constant 0 : index
    // MFMA_F32-NEXT: %[[LD0:.*]] = load %[[MV]][%[[IT0]]] : memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[MFMA0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD0]]) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    // MFMA_F32-NEXT: store %[[MFMA0]], %[[MV]][%[[IT0]]] : memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[IT1:.*]] = constant 1 : index
    // MFMA_F32-NEXT: %[[LD1:.*]] = load %[[MV]][%[[IT1]]] : memref<16xvector<4xf32>>
    // MFMA_F32-NEXT: %[[MFMA1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD1]]) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : f32, vector<4xf32>
    // MFMA_F32-NEXT: store %[[MFMA1]], %[[MV]][%[[IT1]]] : memref<16xvector<4xf32>>

    return
  }

  // ----

  func @mfma_f16(%a : vector<4xf16>, %b : vector<4xf16>, %c : memref<64xf32>) {
    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 64 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[IT0:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD0:.*]] = load %[[MV]][%[[IT0]]] : memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[MFMA0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD0]]) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    // MFMA_F16-NEXT: store %[[MFMA0]], %[[MV]][%[[IT0]]] : memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[IT1:.*]] = constant 1 : index
    // MFMA_F16-NEXT: %[[LD1:.*]] = load %[[MV]][%[[IT1]]] : memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[MFMA1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD1]]) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    // MFMA_F16-NEXT: store %[[MFMA1]], %[[MV]][%[[IT1]]] : memref<2xvector<32xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 32, n_per_wave = 64 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    // MFMA_F16-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 32 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>
    // MFMA_F16-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x4f16"} : vector<4xf16>, vector<32xf32>
    // MFMA_F16-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 32, n_per_wave = 32 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // MFMA_F16-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // MFMA_F16-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x8f16"} : vector<4xf16>, vector<16xf32>
    // MFMA_F16-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 16, n_per_wave = 16 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x16f16"} : vector<4xf16>, vector<4xf32>
    // MFMA_F16-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 16, n_per_wave = 64 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // MFMA_F16-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // MFMA_F16-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    // MFMA_F16-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 16 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // MFMA_F16-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // MFMA_F16-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x4f16"} : vector<4xf16>, vector<16xf32>
    // MFMA_F16-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 4, n_per_wave = 64 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[IT:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    // MFMA_F16-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 8, n_per_wave = 64 }: vector<4xf16>, memref<64xf32>
    // MFMA_F16:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[IT0:.*]] = constant 0 : index
    // MFMA_F16-NEXT: %[[LD0:.*]] = load %[[MV]][%[[IT0]]] : memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[MFMA0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD0]]) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    // MFMA_F16-NEXT: store %[[MFMA0]], %[[MV]][%[[IT0]]] : memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[IT1:.*]] = constant 1 : index
    // MFMA_F16-NEXT: %[[LD1:.*]] = load %[[MV]][%[[IT1]]] : memref<16xvector<4xf32>>
    // MFMA_F16-NEXT: %[[MFMA1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD1]]) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x4f16"} : vector<4xf16>, vector<4xf32>
    // MFMA_F16-NEXT: store %[[MFMA1]], %[[MV]][%[[IT1]]] : memref<16xvector<4xf32>>

    return
  }
}
