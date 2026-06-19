// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for ThreadwiseGemmOp::verify, BlockwiseGemmOp::verify,
// COM: BlockwiseFillOp::verify and the remaining GlobalLoadToLDSOp::verify
// COM: branches in mlir/lib/Dialect/Rock/IR/RockDialect.cpp.

// COM: threadwise_gemm: K dimensions must match between A and B
func.func @threadwise_gemm_k_mismatch(%a: memref<4x8x1xf32, 5>, %b: memref<5x8x1xf32, 5>, %c: memref<8x8xf32, 5>) {
  // expected-error @+1 {{K dimensions don't match}}
  rock.threadwise_gemm %c += %a * %b : memref<8x8xf32, 5> += memref<4x8x1xf32, 5> * memref<5x8x1xf32, 5>
  return
}

// -----

// COM: threadwise_gemm: M dimensions must match between A and C
func.func @threadwise_gemm_m_mismatch(%a: memref<4x8x1xf32, 5>, %b: memref<4x8x1xf32, 5>, %c: memref<7x8xf32, 5>) {
  // expected-error @+1 {{M dimensions don't match}}
  rock.threadwise_gemm %c += %a * %b : memref<7x8xf32, 5> += memref<4x8x1xf32, 5> * memref<4x8x1xf32, 5>
  return
}

// -----

// COM: threadwise_gemm: N dimensions must match between B and C
func.func @threadwise_gemm_n_mismatch(%a: memref<4x8x1xf32, 5>, %b: memref<4x8x1xf32, 5>, %c: memref<8x7xf32, 5>) {
  // expected-error @+1 {{N dimensions don't match}}
  rock.threadwise_gemm %c += %a * %b : memref<8x7xf32, 5> += memref<4x8x1xf32, 5> * memref<4x8x1xf32, 5>
  return
}

// -----

// COM: threadwise_gemm: KPack dimensions must match between A and B
func.func @threadwise_gemm_kpack_mismatch(%a: memref<4x8x1xf32, 5>, %b: memref<4x8x2xf32, 5>, %c: memref<8x8xf32, 5>) {
  // expected-error @+1 {{KPack dimensions don't match}}
  rock.threadwise_gemm %c += %a * %b : memref<8x8xf32, 5> += memref<4x8x1xf32, 5> * memref<4x8x2xf32, 5>
  return
}

// -----

// COM: blockwise_gemm: K dimensions must match between A and B
func.func @blockwise_gemm_k_mismatch(%a: memref<8x128x1xf32, 3>, %b: memref<9x128x1xf32, 3>, %c: memref<8x8xf32, 5>) {
  // expected-error @+1 {{Mismatched k dimensions between A and B}}
  rock.blockwise_gemm %c += %a * %b {
    inMPerThread = 2 : i32, inNPerThread = 2 : i32,
    params = #rock.general_gemm_params<blockSize = 256, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 1, kPerThread = 1, mPerThread = 4, nPerThread = 4, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
  } : memref<8x8xf32, 5> += memref<8x128x1xf32, 3> * memref<9x128x1xf32, 3>
  return
}

// -----

// COM: blockwise_gemm: kPack must match between A and B
func.func @blockwise_gemm_kpack_mismatch(%a: memref<8x128x1xf32, 3>, %b: memref<8x128x2xf32, 3>, %c: memref<8x8xf32, 5>) {
  // expected-error @+1 {{Mismatched kPack between A and B}}
  rock.blockwise_gemm %c += %a * %b {
    inMPerThread = 2 : i32, inNPerThread = 2 : i32,
    params = #rock.general_gemm_params<blockSize = 256, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kpack = 1, kPerThread = 1, mPerThread = 4, nPerThread = 4, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
  } : memref<8x8xf32, 5> += memref<8x128x1xf32, 3> * memref<8x128x2xf32, 3>
  return
}

// -----

// COM: blockwise_fill: the memref must be flat (rank 1)
func.func @blockwise_fill_not_flat(%c1: f32) {
  %ldsbuf = rock.alloc() : memref<16x16xf32, #gpu.address_space<workgroup>>
  // expected-error @+1 {{Blockwise fill expects a flat memref}}
  rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<16x16xf32, #gpu.address_space<workgroup>>, f32
  return
}

// -----

// COM: blockwise_fill: the memref must live in workgroup memory
func.func @blockwise_fill_not_workgroup(%c1: f32) {
  %ldsbuf = rock.alloc() : memref<256xf32, #gpu.address_space<private>>
  // expected-error @+1 {{Memory space is expected to be workgroup}}
  rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<256xf32, #gpu.address_space<private>>, f32
  return
}

// -----

// COM: blockwise_fill: the vector length must divide the memref size
func.func @blockwise_fill_bad_vector_len(%c1: vector<4xf32>) {
  %ldsbuf = rock.alloc() : memref<255xf32, #gpu.address_space<workgroup>>
  // expected-error @+1 {{The vector length is not a factor in memref size.}}
  rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<255xf32, #gpu.address_space<workgroup>>, vector<4xf32>
  return
}

// -----

// COM: global_load_to_lds: the destination must live in workgroup memory
func.func @global_load_to_lds_dest_not_lds(%source: memref<64xf32>, %dest: memref<64xf32, #gpu.address_space<private>>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // expected-error @+1 {{Destination memref must live in workgroup memory}}
  rock.global_load_to_lds %source[%c0] -> %dest[%c0] if %true {transferType = f32} : memref<64xf32> -> memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

// COM: global_load_to_lds: only 128-bit and 32-bit transfers are supported
func.func @global_load_to_lds_bad_transfer(%source: memref<64xf16>, %dest: memref<64xf16, #gpu.address_space<workgroup>>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // expected-error @+1 {{Direct to LDS is implemented for 128bit and 32bit loads only}}
  rock.global_load_to_lds %source[%c0] -> %dest[%c0] if %true {transferType = f16} : memref<64xf16> -> memref<64xf16, #gpu.address_space<workgroup>>
  return
}
