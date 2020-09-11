//func @test(%vector_a : vector<32xf32>, %vector_b : vector<32xf32>) -> vector<32xf32> {
//  %c = addf %vector_a, %vector_b : vector<32xf32>
//  return %c : vector<32xf32>
//}


//func @test(%memref_a : memref<32xf32>, %memref_b : memref<32xf32>) -> vector<32xf32> {
//  %memref_vector_a = vector.type_cast %memref_a : memref<32xf32> to memref<vector<32xf32>>
//  %memref_vector_b = vector.type_cast %memref_b : memref<32xf32> to memref<vector<32xf32>>
//
//  %vector_a = load %memref_vector_a[] : memref<vector<32xf32>>
//  %vector_b = load %memref_vector_b[] : memref<vector<32xf32>>
//
//  %c = addf %vector_a, %vector_b : vector<32xf32>
//  return %c : vector<32xf32>
//}

//gpu.module @mfma {
//  gpu.func @mfma_f32(%a : f32, %b : f32, %result_c : memref<1xf32>) kernel {
//    %c0f = constant 0.0 : f32
//    %vector_c = splat %c0f : vector<32xf32>
//    %d = gpu.mfma(%a, %b, %vector_c) : f32, vector<32xf32>
//
//    %c0 = constant 0 : index
//    %c0_i32 = constant 0 : i32
//    %d0 = vector.extractelement %d[%c0_i32 : i32] : vector<32xf32>
//    store %d0, %result_c[%c0] : memref<1xf32>
//    gpu.return
//  }
//}

//gpu.module @mfma {
//  gpu.func @mfma_f32(%a : f32, %b : f32, %result_c : memref<32xf32>) kernel {
//    %c0 = constant 0 : index
//    %c1 = constant 1 : index
//    %c4 = constant 4 : index
//    %c32 = constant 32 : index
//
//    %c0f = constant 0.0 : f32
//    %vector_c = splat %c0f : vector<4x32xf32>
//    %vector_c0 = vector.extract %vector_c[0] : vector<4x32xf32>
//    %vector_c1 = vector.extract %vector_c[1] : vector<4x32xf32>
//
//    %d0 = gpu.mfma(%a, %b, %vector_c0) { imm = [0, 0, 0] } : f32, vector<32xf32>
//    %e0 = gpu.mfma(%a, %b, %vector_c1) { imm = [0, 0, 1] } : f32, vector<32xf32>
//
//    %d1 = gpu.mfma(%a, %b, %d0) { imm = [0, 0, 0] } : f32, vector<32xf32>
//    %e1 = gpu.mfma(%a, %b, %e0) { imm = [0, 0, 1] } : f32, vector<32xf32>
//
//    %d2 = gpu.mfma(%a, %b, %d1) { imm = [0, 0, 0] } : f32, vector<32xf32>
//    %e2 = gpu.mfma(%a, %b, %e1) { imm = [0, 0, 1] } : f32, vector<32xf32>
//
//    %d3 = gpu.mfma(%a, %b, %d2) { imm = [0, 0, 0] } : f32, vector<32xf32>
//    %e3 = gpu.mfma(%a, %b, %e2) { imm = [0, 0, 1] } : f32, vector<32xf32>
//
//    scf.for %i = %c0 to %c32 step %c1 {
//      %i_i32 = index_cast %i : index to i32
//      %v0 = vector.extractelement %d3[%i_i32 : i32] : vector<32xf32>
//      %v1 = vector.extractelement %e3[%i_i32 : i32] : vector<32xf32>
//      %r0 = addf %v0, %v1 : f32
//      store %r0, %result_c[%i] : memref<32xf32>
//    }
//    gpu.return
//  }
//}

//gpu.module @mfma {
//  gpu.func @mfma_f32(%a : f32, %b : f32, %result_c : memref<32xf32>) kernel {
//    %c0 = constant 0 : index
//    %c1 = constant 1 : index
//    %c4 = constant 4 : index
//    %c32 = constant 32 : index
//
//    %c0f = constant 0.0 : f32
//    %vector_c = splat %c0f : vector<4x32xf32>
//    %vector_c0 = vector.extract %vector_c[0] : vector<4x32xf32>
//    %vector_c1 = vector.extract %vector_c[1] : vector<4x32xf32>
//
//    %d3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg_c0 = %vector_c0) -> (vector<32xf32>) {
//      %d = gpu.mfma(%a, %b, %arg_c0) { imm = [0, 0, 0] } : f32, vector<32xf32>
//      scf.yield %d : vector<32xf32>
//    }
//
//    %e3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg_c1 = %vector_c1) -> (vector<32xf32>) {
//      %d = gpu.mfma(%a, %b, %arg_c1) { imm = [0, 0, 1] } : f32, vector<32xf32>
//      scf.yield %d : vector<32xf32>
//    }
//
//    scf.for %i = %c0 to %c32 step %c1 {
//      %i_i32 = index_cast %i : index to i32
//      %v0 = vector.extractelement %d3[%i_i32 : i32] : vector<32xf32>
//      %v1 = vector.extractelement %e3[%i_i32 : i32] : vector<32xf32>
//      %r0 = addf %v0, %v1 : f32
//      store %r0, %result_c[%i] : memref<32xf32>
//    }
//    gpu.return
//  }
//}

//gpu.module @mfma {
//  gpu.func @mfma_f32(%a : f32, %b : f32, %result_c : memref<32xf32>) kernel {
//    %c0 = constant 0 : index
//    %c1 = constant 1 : index
//    %c4 = constant 4 : index
//    %c32 = constant 32 : index
//
//    %c0f = constant 0.0 : f32
//    %vector_c = splat %c0f : vector<4x32xf32>
//    %vector_c0 = vector.extract %vector_c[0] : vector<4x32xf32>
//    %vector_c1 = vector.extract %vector_c[1] : vector<4x32xf32>
//
//    %d3, %e3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg_c0 = %vector_c0, %arg_c1 = %vector_c1) -> (vector<32xf32>, vector<32xf32>) {
//      %d0 = gpu.mfma(%a, %b, %arg_c0) { imm = [1, 0, 0] } : f32, vector<32xf32>
//      %d1 = gpu.mfma(%a, %b, %arg_c1) { imm = [1, 1, 0] } : f32, vector<32xf32>
//      scf.yield %d0, %d1 : vector<32xf32>, vector<32xf32>
//    }
//
//    scf.for %i = %c0 to %c32 step %c1 {
//      %i_i32 = index_cast %i : index to i32
//      %v0 = vector.extractelement %d3[%i_i32 : i32] : vector<32xf32>
//      %v1 = vector.extractelement %e3[%i_i32 : i32] : vector<32xf32>
//      %r0 = addf %v0, %v1 : f32
//      store %r0, %result_c[%i] : memref<32xf32>
//    }
//    gpu.return
//  }
//}

gpu.module @mfma {
  gpu.func @mfma_f32(%a0 : f32, %b0 : f32, %a1 : f32, %b1 : f32, %result_c : memref<128xf32>) kernel {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c16 = constant 16 : index
    %c32 = constant 32 : index
    %c64 = constant 64 : index
    %c96 = constant 96 : index
    %c128 = constant 128 : index

    // miopen.gridwise_gemm will be revised to select the followings:
    //
    // - MFMA intrinsic to be used.
    // - Immediates supplied per MFMA intrinsic.
    // - Number of MFMA intrinsic to be issued per iteration.
    // - Loop iterations needed.
    // - Number of vectors to hold matrix C.
    //
    // Based on the following information:
    // - Data type : fp32 / fp16 / bf16
    // - MPerWave attribute.
    // - NPerWave attribute.
    //
    // All the code selection process will be carried out at miopen.gridwise_gemm level now.

    // miopen.gridwise_gemm will be revised to allocate and 0-initialize matrix C on VGPR like below.
    // vector.type_cast is no longer necessary.
    %c0f = constant 0.0 : f32
    %vector_c0 = splat %c0f : vector<32xf32>
    %vector_c1 = splat %c0f : vector<32xf32>

    // miopen.xdlops_gemm will be rewritten to produce this block.
    // Notice miopen.xdlops_gemm would now take variadic numbers of vectors.
    %d3, %e3 = scf.for %i = %c0 to %c16 step %c1 iter_args(%arg_c0 = %vector_c0, %arg_c1 = %vector_c1) -> (vector<32xf32>, vector<32xf32>) {
      %d0 = gpu.mfma(%a0, %b0, %arg_c0) { imm = [1, 0, 0] } : f32, vector<32xf32>
      %e0 = gpu.mfma(%a0, %b0, %arg_c1) { imm = [1, 1, 0] } : f32, vector<32xf32>
      scf.yield %d0, %e0 : vector<32xf32>, vector<32xf32>
    }

    // miopen.threadwise_copy will be revised to produce following blocks when given a vector typed input.
    scf.for %i = %c0 to %c32 step %c1 {
      %i_i32 = index_cast %i : index to i32
      %d = vector.extractelement %d3[%i_i32 : i32] : vector<32xf32>
      %j0 = addi %i, %c0 : index
      store %d, %result_c[%j0] : memref<128xf32>
    }
    // VERY IMPORTANT, each vector shall be read out individually to reduce VGPR register pressure!
    scf.for %i = %c0 to %c32 step %c1 {
      %i_i32 = index_cast %i : index to i32
      %e = vector.extractelement %e3[%i_i32 : i32] : vector<32xf32>
      %j1 = addi %i, %c32 : index
      store %e, %result_c[%j1] : memref<128xf32>
    }

    gpu.return
  }
}
 
// gpu.module @mfma {
//   gpu.func @blah() kernel {
//     //%vector_c2 = splat %c0f : vector<32xf32>
//     //%vector_c3 = splat %c0f : vector<32xf32>
//     //%f3, %g3 = scf.for %i = %c0 to %c16 step %c1 iter_args(%arg_c0 = %vector_c2, %arg_c1 = %vector_c3) -> (vector<32xf32>, vector<32xf32>) {
//     //  %f0 = gpu.mfma(%a1, %b1, %arg_c0) { imm = [1, 0, 0] } : f32, vector<32xf32>
//     //  %g0 = gpu.mfma(%a1, %b1, %arg_c1) { imm = [1, 1, 0] } : f32, vector<32xf32>
//     //  scf.yield %f0, %g0 : vector<32xf32>, vector<32xf32>
//     //}
//     //scf.for %i = %c0 to %c32 step %c1 {
//     //  %i_i32 = index_cast %i : index to i32
//     //  %f = vector.extractelement %f3[%i_i32 : i32] : vector<32xf32>
//     //  %j = addi %i, %c64 : index
//     //  store %f, %result_c[%j] : memref<128xf32>
//     //}
//     //scf.for %i = %c0 to %c32 step %c1 {
//     //  %i_i32 = index_cast %i : index to i32
//     //  %g = vector.extractelement %g3[%i_i32 : i32] : vector<32xf32>
//     //  %j = addi %i, %c96 : index
//     //  store %g, %result_c[%j] : memref<128xf32>
//     //}
// 
// 
//     // ScratchSize: 644
//     //%d3, %e3, %f3, %g3 = scf.for %i = %c0 to %c32 step %c1 iter_args(%arg_c0 = %vector_c0, %arg_c1 = %vector_c1, %arg_c2 = %vector_c2, %arg_c3 = %vector_c3) -> (vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>) {
//     //  %d0 = gpu.mfma(%a0, %b0, %arg_c0) { imm = [1, 0, 0] } : f32, vector<32xf32>
//     //  %e0 = gpu.mfma(%a0, %b0, %arg_c1) { imm = [1, 1, 0] } : f32, vector<32xf32>
//     //  %f0 = gpu.mfma(%a1, %b1, %arg_c2) { imm = [1, 0, 0] } : f32, vector<32xf32>
//     //  %g0 = gpu.mfma(%a1, %b1, %arg_c3) { imm = [1, 1, 0] } : f32, vector<32xf32>
//     //  scf.yield %d0, %e0, %f0, %g0 : vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>
//     //}
// 
//     gpu.return
//   }
// }

