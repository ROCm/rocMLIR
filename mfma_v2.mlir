// RUN: miopen-opt -convert-miopen-to-gpu %s | FileCheck %s

module {
  func @mfma_i8_4xi32(%a : vector<4xi8>, %b : vector<4xi8>, %result : memref<16xi32>) attributes {kernel = 0 : i32} {
    %vec = arith.constant dense<0> : vector<16xi32>
    %d0 = miopen.mfma_v2(%a, %b, %vec) { instr = "mfma_i32_32x32x8i8", imm = [0, 0, 0]}: vector<4xi8>, vector<16xi32>

    // Make sure that compiler don't optimize the above away
    %c0 = constant 0 : index
    %v0 = vector.extractelement %d0[%c0 : index] : vector<16xi32>
    memref.store %v0, %result[%c0] : memref<16xi32>
    return }
}
