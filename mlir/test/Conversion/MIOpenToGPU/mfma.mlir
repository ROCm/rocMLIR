// RUN: mlir-opt -convert-miopen-to-gpu="kernel-name=mfma" %s | FileCheck %s

module {
  func @mfma(%a : f32, %b : f32, %c : memref<64xf32>) {
    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 64 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[IT0:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD0:.*]] = load %[[MV]][%[[IT0]]] : memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[MFMA0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD0]]) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : vector<32xf32>
    // CHECK-NEXT: store %[[MFMA0]], %[[MV]][%[[IT0]]] : memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[IT1:.*]] = constant 1 : index
    // CHECK-NEXT: %[[LD1:.*]] = load %[[MV]][%[[IT1]]] : memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[MFMA1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD1]]) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : vector<32xf32>
    // CHECK-NEXT: store %[[MFMA1]], %[[MV]][%[[IT1]]] : memref<2xvector<32xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 32, n_per_wave = 64 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[IT:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : vector<32xf32>
    // CHECK-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>

    // ----
  
    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 32 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[IT:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>
    // CHECK-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 1 : i32], instr = "mfma_f32_32x32x1f32"} : vector<32xf32>
    // CHECK-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<2xvector<32xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 32, n_per_wave = 32 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // CHECK-NEXT: %[[IT:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // CHECK-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x2f32"} : vector<16xf32>
    // CHECK-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 16, n_per_wave = 16 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[IT:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x4f32"} : vector<4xf32>
    // CHECK-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 16, n_per_wave = 64 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // CHECK-NEXT: %[[IT:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // CHECK-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [2 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_16x16x1f32"} : vector<16xf32>
    // CHECK-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 64, n_per_wave = 16 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<4xvector<16xf32>>
    // CHECK-NEXT: %[[IT:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>
    // CHECK-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [0 : i32, 0 : i32, 4 : i32], instr = "mfma_f32_16x16x1f32"} : vector<16xf32>
    // CHECK-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<4xvector<16xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 8, n_per_wave = 64 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[IT:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD:.*]] = load %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[MFMA:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD]]) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : vector<4xf32>
    // CHECK-NEXT: store %[[MFMA]], %[[MV]][%[[IT]]] : memref<16xvector<4xf32>>

    // ----

    miopen.mfma(%a, %b, %c) { m_per_wave = 4, n_per_wave = 64 }: f32, f32, memref<64xf32>
    // CHECK:      %[[MV:.*]] = vector.type_cast %{{.*}} : memref<64xf32> to memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[IT0:.*]] = constant 0 : index
    // CHECK-NEXT: %[[LD0:.*]] = load %[[MV]][%[[IT0]]] : memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[MFMA0:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD0]]) {imm = [4 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : vector<4xf32>
    // CHECK-NEXT: store %[[MFMA0]], %[[MV]][%[[IT0]]] : memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[IT1:.*]] = constant 1 : index
    // CHECK-NEXT: %[[LD1:.*]] = load %[[MV]][%[[IT1]]] : memref<16xvector<4xf32>>
    // CHECK-NEXT: %[[MFMA1:.*]] = gpu.mfma(%{{.*}}, %{{.*}}, %[[LD1]]) {imm = [4 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_4x4x1f32"} : vector<4xf32>
    // CHECK-NEXT: store %[[MFMA1]], %[[MV]][%[[IT1]]] : memref<16xvector<4xf32>>

    return
  }
}
