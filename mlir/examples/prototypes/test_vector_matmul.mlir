module {
  func @miopen_threadwise_gemm_high_level(%a : memref<2x8xf32>, %b : memref<2x8xf32>, %c : memref<8x8xf32>, %d : vector<8x8xf32>) {

    // Convert memrefs to memrefs of vectors.
    %mv_a = vector.type_cast %a : memref<2x8xf32> to memref<vector<2x8xf32>>
    %mv_b = vector.type_cast %b : memref<2x8xf32> to memref<vector<2x8xf32>>
    %mv_c = vector.type_cast %c : memref<8x8xf32> to memref<vector<8x8xf32>>

    // Load A.
    %v_a = load %mv_a[] : memref<vector<2x8xf32>>
    // Transpose(A).
    %vt_a = vector.transpose %v_a, [1, 0] : vector<2x8xf32> to vector<8x2xf32>

    // Load B.
    %v_b = load %mv_b[] : memref<vector<2x8xf32>>

    // Load C.
    %v_c = load %mv_c[] : memref<vector<8x8xf32>>

    // Flatten Transpose(A) and B.
    %vt1d_a = vector.shape_cast %vt_a : vector<8x2xf32> to vector<16xf32>
    %v1d_b = vector.shape_cast %v_b : vector<2x8xf32> to vector<16xf32>

    // C += Transpose(A) * B.
    %v1d_c = vector.matrix_multiply %vt1d_a, %v1d_b { lhs_rows = 8 : i32, lhs_columns = 2 : i32, rhs_columns = 8 : i32 } : (vector<16xf32>, vector<16xf32>) -> vector<64xf32>
    %v_c_new = vector.shape_cast %v1d_c : vector<64xf32> to vector<8x8xf32>
    %v_c_result = addf %v_c, %v_c_new : vector<8x8xf32>
    store %v_c_result, %mv_c[] : memref<vector<8x8xf32>>

    return
  }
}

