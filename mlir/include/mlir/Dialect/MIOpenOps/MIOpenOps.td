//===- MIOpenOps.td - MIOpen operation definitions ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR MIOpen operations.
//
//===----------------------------------------------------------------------===//

#ifndef MIOPEN_OPS
#define MIOPEN_OPS

include "mlir/IR/OpBase.td"
//include "mlir/Transforms/LoopLikeInterface.td"

def MIOpen_Dialect : Dialect {
  let name = "miopen";
  let cppNamespace = "miopen";
}

// Base class for MIOpen dialect ops.
class MIOpen_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MIOpen_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
  }];
}

def MIOpen_Conv2DOp :
    MIOpen_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32], [4]>:$filter,
                   MemRefRankOf<[F32], [4]>:$input,
                   MemRefRankOf<[F32], [4]>:$output)> {
  let summary = "2D convolution forward";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution forward.
  }];
}

def MIOpen_Conv2DBwdDataOp :
    MIOpen_Op<"conv2d_bwd_data">,
    Arguments<(ins MemRefRankOf<[F32], [4]>:$filter,
                   MemRefRankOf<[F32], [4]>:$input,
                   MemRefRankOf<[F32], [4]>:$output)> {
  let summary = "2D convolution backward data";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution backward data.
  }];
}

def MIOpen_TransformOp :
    MIOpen_Op<"transform">,
    Arguments<(ins AnyMemRef:$input)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Tensor transformation";
  let description = [{
    The `miopen.transform` op transforms tensor coordinates.
  }];
}

def MIOpen_GridwiseGemmOp :
    MIOpen_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM.
  }];
}

// gridwise_gemm_ex would eventually be replaced with gridwise_gemm
// keep it as a separate op for now.
// the difference with gridwise_gemm is gridwise_gemm_ex carries tuning
// parameters. On the other hand, gridwise_gemm only carries convolution
// configuration.
def MIOpen_GridwiseGemmExOp:
    MIOpen_Op<"gridwise_gemm_ex">,
    Arguments<(ins MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>)> {
  let summary = "Gridwise GEMM extended";
  let description = [{
    The `miopen.gridwise_gemm_ex` op computes gridwise GEMM, with tuning
    parameters supplied in the attributes.
  }];
}

// Memory allocation on GPU memory hierachy.
// TBD: eventually replace this with std.alloc.
def MIOpen_GpuAllocOp:
    MIOpen_Op<"gpu_alloc">,
    Arguments<(ins Index:$sizeBytes, Index:$memorySpace)>,
    Results<(outs MemRefRankOf<[I8], [1]>:$output)> {
  let summary = "Memory allocation on GPU";
  let description = [{
    The `miopen.gpu_alloc` op allocates memory on GPU.
    - Address space 0 : global.
    - Address space 3 : LDS.
    - Address space 5 : private (VGPR).
    All other values would be considered as allocation on global.
  }];
}

// subview provides a subsection view of a memref.
// TBD: eventually replace this with std.view and/or std.subview.
def MIOpen_SubviewOp:
    MIOpen_Op<"subview">,
    Arguments<(ins AnyMemRef:$input,
                   Index:$offset)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Subview of memref on GPU";
  let description = [{
    The `miopen.subview` op creates a subview of a memref on GPU.
    The memref must be on either LDS (address space 3) or private (address
    space 5).
  }];
}

// TBD: eventually replace this with linalg.fill?
def MIOpen_FillOp:
    MIOpen_Op<"fill">,
    Arguments<(ins AnyMemRef:$input,
                   Index:$value)> {
  let summary = "Fill memory with constant value on GPU";
  let description = [{
    The `miopen.fill` op fills a memref on GPU with a constant value.
  }];
}

// TBD: eventually replace this with the op in GPU dialect?
def MIOpen_LdsBarrierOp:
    MIOpen_Op<"lds_barrier"> {
  let summary = "Setup an LDS barrier";
  let description = [{
    The `miopen.lds_barrier` op sets up a barrier on LDS.
  }];
}

// blockwise_copy
def MIOpen_BlockwiseCopyOp:
    MIOpen_Op<"blockwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest)> {
  let summary = "Blockwise GPU data movement";
  let description = [{
    The `miopen.blockwise` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to LDS (naive tensor).
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
  }];
}

// threadwise_copy
def MIOpen_ThreadwiseCopyOp:
    MIOpen_Op<"threadwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest)> {
  let summary = "Threadwise GPU data movement";
  let description = [{
    The `miopen.threadwise` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) to global (generic tensor).
    - LDS (naive tensor) to register (naive tensor).
  }];
}

// blockwise_gemm
def MIOpen_BlockwiseGemmOp:
    MIOpen_Op<"blockwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>,
                   MemRefRankOf<[F32], [2]>)> {
  let summary = "Blockwise GEMM";
  let description = [{
    The `miopen.block_gemm` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall reside on register (naive tensor).
  }];
}

// threadwise_gemm
def MIOpen_ThreadwiseGemmOp:
    MIOpen_Op<"threadwise_gemm">,
    Arguments<(ins AnyMemRef,
                   AnyMemRef,
                   AnyMemRef)> {
  let summary = "Threadwise GEMM";
  let description = [{
    The `miopen.threadwise_gemm` op does GEMM at thread level.
    - Matrix A and Matrix B shall reside on register (naive tensor).
    - Matrix C shall reside on LDS (naive tensor).
  }];
}

#endif // MIOPEN_OPS
