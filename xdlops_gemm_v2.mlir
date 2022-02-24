#map = affine_map<(d0) -> (d0 + 512)>
module  {
  func @miopen_xdlops_gemm_v2_two_results(%arg0: memref<512xf32, 3>, %arg1: memref<512xf32, #map, 3>, %arg2: memref<2xvector<2xf32>, 5>, %arg3: memref<2xvector<2xf32>, 5>) -> (vector<32xf32>, vector<32xf32>) {
    %c512 = arith.constant 512 : index
    %cst = arith.constant dense<0.000000e+00> : vector<2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32xf32>
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    // thread id
    %0 = miopen.workitem_id : index
    // lane id
    %1 = arith.remui %0, %c64 : index
    // Store to buffer A
    // Original C++ logic.
    // static_if<!IsKReduction>{}([&](auto) {
    //   for(index_t m_i = 0; m_i < MRepeats; ++m_i)
    //     for(index_t k_i      = 0; k_i < K; ++k_i)
    //       a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
    // p_a_wave need to be offseted by waveOffsetA.

    // m_i = 0 : MRepeats
    affine.for %arg4 = 0 to 1 {
      %3 = arith.muli %arg4, %c64 : index
      %4 = arith.muli %arg4, %c2 : index
      // k_i = 0 : K
      affine.for %arg5 = 0 to 2 {
        %5 = arith.muli %arg5, %c128 : index
        %6 = arith.addi %5, %1 : index
        %7 = arith.addi %6, %3 : index
        // Assemble %11
        %8 = arith.muli %7, %c2 : index // = 2(64 * m_i + %1 + (128 x k_i))
        %9 = arith.addi %arg5, %4 : index // target, = k_i + m_i * K
        %10 = memref.load %arg0[%8] : memref<512xf32, 3>
        %11 = vector.insertelement %10, %cst[%c0 : index] : vector<2xf32>
        // Assemble %13
        %12 = arith.addi %8, %c1 : index // = %8 + 1
        %13 = memref.load %arg0[%12] : memref<512xf32, 3>
        // Merge %11 and %13
        %14 = vector.insertelement %13, %11[%c1 : index] : vector<2xf32>
        memref.store %14, %arg2[%9] : memref<2xvector<2xf32>, 5>
      }
    }
    // Store to buffer B
    affine.for %arg4 = 0 to 1 {
      %3 = arith.muli %arg4, %c64 : index
      %4 = arith.muli %arg4, %c2 : index
      affine.for %arg5 = 0 to 2 {
        %5 = arith.muli %arg5, %c128 : index
        %6 = arith.addi %5, %1 : index
        %7 = arith.addi %6, %3 : index
        %8 = arith.muli %7, %c2 : index
        %9 = arith.addi %8, %c512 : index
        %10 = arith.addi %arg5, %4 : index
        %11 = memref.load %arg1[%9] : memref<512xf32, #map, 3>
        %12 = vector.insertelement %11, %cst[%c0 : index] : vector<2xf32>
        %13 = arith.addi %9, %c1 : index
        %14 = memref.load %arg1[%13] : memref<512xf32, #map, 3>
        %15 = vector.insertelement %14, %12[%c1 : index] : vector<2xf32>
        memref.store %15, %arg3[%10] : memref<2xvector<2xf32>, 5>
      }
    }
    // k_i = 0 : K
    %2:2 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %cst_0, %arg6 = %cst_0) -> (vector<32xf32>, vector<32xf32>) {
      %3 = memref.load %arg2[%arg4] : memref<2xvector<2xf32>, 5> // = a[k_i]
      %4 = memref.load %arg3[%arg4] : memref<2xvector<2xf32>, 5> // = b[k_i]
      // k_i = 0 : KRepeats
      %5:2 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (vector<32xf32>, vector<32xf32>) {
      // Do mfma based on buffer A and buffer B
        %6 = arith.index_cast %arg7 : index to i32
        %7 = vector.extractelement %3[%6 : i32] : vector<2xf32>
        %8 = vector.extractelement %4[%6 : i32] : vector<2xf32>
        %9 = miopen.mfma_v2(%7, %8, %arg8) {imm = [1 : i32, 0 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
        %10 = miopen.mfma_v2(%7, %8, %arg9) {imm = [1 : i32, 1 : i32, 0 : i32], instr = "mfma_f32_32x32x1f32"} : f32, vector<32xf32>
        affine.yield %9, %10 : vector<32xf32>, vector<32xf32>
      }
      affine.yield %5#0, %5#1 : vector<32xf32>, vector<32xf32>
    }
    return %2#0, %2#1 : vector<32xf32>, vector<32xf32>
  }
}

