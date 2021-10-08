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

include "mlir/Dialect/MIOpen/MIOpenBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for MIOpen dialect ops.
class MIOpen_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MIOpen_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.

  let extraClassDeclaration = [{
  }];
}

def MIOpen_Conv2DOp :
    MIOpen_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [5]>:$filter,
                   MemRefRankOf<[F32, F16, I16], [5]>:$input,
                   MemRefRankOf<[F32, F16, I16], [5]>:$output)> {
  let summary = "2D convolution forward";
  let description = [{
    The `miopen.conv2d` op computes 2D convolution forward.
  }];
  let verifier = "{ return ::verifyConvOp(*this); }";
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

def MIOpen_Conv2DBwdDataOp :
    MIOpen_Op<"conv2d_bwd_data">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [5]>:$filter,
                   MemRefRankOf<[F32, F16, I16], [5]>:$input,
                   MemRefRankOf<[F32, F16, I16], [5]>:$output)> {
  let summary = "2D convolution backward data";
  let description = [{
    The `miopen.conv2d_bwd_data` op computes 2D convolution backward data.
  }];
  let verifier = "{ return ::verifyConvOp(*this); }";
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

def MIOpen_Conv2DBwdWeightOp :
    MIOpen_Op<"conv2d_bwd_weight">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [5]>:$filter,
                   MemRefRankOf<[F32, F16, I16], [5]>:$input,
                   MemRefRankOf<[F32, F16, I16], [5]>:$output)> {
  let summary = "2D convolution backward weight";
  let description = [{
    The `miopen.conv2d_bwd_weight` op computes 2D convolution backward weight.
  }];
  let verifier = "{ return ::verifyConvOp(*this); }";
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
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
  let builders = [
   // Custom builder to populate bounds of input and output memrefs as attributes.
   OpBuilder<(ins "Type": $output, "Value":$input, "ArrayRef<NamedAttribute>":$attributes, "bool":$populateBounds),
   [{
     $_state.addOperands({input});
     if (populateBounds) {
       MemRefType lowerMemRefType = input.getType().template cast<MemRefType>();
       MemRefType upperMemRefType = output.template cast<MemRefType>();
       $_state.addAttribute("lower_layer_bounds", buildMemRefShapeAttr($_builder, lowerMemRefType));
       $_state.addAttribute("upper_layer_bounds", buildMemRefShapeAttr($_builder, upperMemRefType));
     }
     $_state.addAttributes(attributes);
     $_state.addTypes(output);
   }]>,
  ];
  let extraClassDeclaration = [{
    // Utility static member function of TransformOp to populate an ArrayAttr to
    // track the bounds of a MemRefType
    static ArrayAttr buildMemRefShapeAttr(OpBuilder &b, MemRefType memRefType);
  }];
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input) `to` type($output)
  }];
}

def MIOpen_GridwiseGemmOp :
    MIOpen_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [3]>:$filter,
                   MemRefRankOf<[F32, F16, I16], [3]>:$input,
                   MemRefRankOf<[F32, F16, I16], [3]>:$output)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

// gridwise_gemm_v2
def MIOpen_GridwiseGemmV2Op :
    MIOpen_Op<"gridwise_gemm_v2">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [3]>:$filter,
                   MemRefRankOf<[F32, F16, I16], [3]>:$input,
                   MemRefRankOf<[F32, F16, I16], [3]>:$output)> {
  let summary = "Gridwise GEMM V2";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM with XDLOPS.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

// Memory allocation on GPU memory hierachy.
def MIOpen_GpuAllocOp:
    MIOpen_Op<"alloc">,
    Results<(outs Res<AnyMemRef, "", [MemAlloc]>:$output)> {
  let summary = "Memory allocation on GPU";
  let description = [{
    The `miopen.alloc` op allocates memory on GPU.
    - Address space 0 : global.
    - Address space 3 : LDS.
    - Address space 5 : private (VGPR).
    All other values would be considered as allocation on global.
  }];
  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($output)
  }];
}

// subview provides a subsection view of a memref.
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
  let assemblyFormat = [{
    `(` $input `,` $offset `)` attr-dict `:` type($input) `to` type($output)
  }];
}

// TBD: eventually replace this with linalg.fill?
def MIOpen_FillOp:
    MIOpen_Op<"fill", [AllElementTypesMatch<["input", "value"]>]>,
    Arguments<(ins AnyMemRef:$input,
                   AnyTypeOf<[AnyInteger, AnyFloat]>:$value)> {
  let summary = "Fill memory with constant value on GPU";
  let description = [{
    The `miopen.fill` op fills a memref on GPU with a constant value.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

def MIOpen_MovePosV2Op:
    MIOpen_Op<"move_pos_v2", [SameOperandsElementType,
      AllTypesMatch<["input", "output"]>]>,
    Arguments<(ins VectorOfRankAndType<[1], [AnyInteger, AnyFloat]>:$input,
                   Variadic<AnyTypeOf<[AnyInteger, AnyFloat]>>:$values)>,
    Results<(outs VectorOfRankAndType<[1], [AnyInteger, AnyFloat]>:$output)> {
    let summary = "Add values to indices within the vector.";
    let description = [{
      The `miopen.move_pos_v2` op adds values to indices within the vector.
    }];
    let assemblyFormat = [{
      `(` operands `)` attr-dict `:` type($input) `,` type($values)
    }];
}

def MIOpen_WorkgroupBarrierOp:
    MIOpen_Op<"workgroup_barrier"> {
  let summary = "Setup an workgroup barrier";
  let description = [{
    The `miopen.workgroup_barrier` op sets up a workgroup-level barrier.
  }];
  let assemblyFormat = "attr-dict";
}

def MIOpen_LDSBarrierOp:
    MIOpen_Op<"lds_barrier"> {
  let summary = "Setup an LDS barrier";
  let description = [{
    The `miopen.lds_barrier` op sets up a workgroup-level barrier on LDS activities.
  }];
  let assemblyFormat = "attr-dict";
}

def MIOpen_WorkgroupIdOp:
    MIOpen_Op<"workgroup_id">,
    Results<(outs Index:$id)> {
  let summary = "Get current workgroup ID";
  let description = [{
    The `miopen.workgroup_id` op gets the current workgroup ID.
  }];
  let assemblyFormat = "attr-dict `:` type($id)";
}

def MIOpen_WorkitemIdOp:
    MIOpen_Op<"workitem_id">,
    Results<(outs Index:$id)> {
  let summary = "Get current workitem ID";
  let description = [{
    The `miopen.workgroup_id` op gets the current workitem ID.
  }];
  let assemblyFormat = "attr-dict `:` type($id)";
}

// blockwise_copy
def MIOpen_BlockwiseCopyOp:
    MIOpen_Op<"blockwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   VectorOfLengthAndType<[3], [I32]>:$sourceCoordVector,
                   VectorOfLengthAndType<[3], [I32]>:$destCoordVector,
                   Optional<AnyMemRef>:$buffer)> {
  let summary = "Blockwise GPU data movement";
  let description = [{
    The `miopen.blockwise` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to LDS (naive tensor).
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

// blockwise_load
def MIOpen_BlockwiseLoadOp:
    MIOpen_Op<"blockwise_load">,
    Arguments<(ins AnyMemRef:$source,
                  VectorOfLengthAndType<[3], [I32]>:$sourceCoordVector)>,
    Results<(outs Variadic<AnyTypeOf<[F32, F16, I16,
                             VectorOfLengthAndType<[2, 4], [F32]>,
                             VectorOfLengthAndType<[2, 4, 8], [F16]>,
                             VectorOfLengthAndType<[2, 4, 8], [I16]>]>>:$result)> {
  let summary = "Blockwise GPU data load";
  let description = [{
    The `miopen.blockwise_load` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to register (naive tensor).
  }];
  let assemblyFormat = [{
    $source `[` $sourceCoordVector `]` attr-dict `:` type(operands)  `->` type(results)
  }];
}

// blockwise_store
def MIOpen_BlockwiseStoreOp:
    MIOpen_Op<"blockwise_store">,
    Arguments<(ins Variadic<AnyTypeOf<[F32, F16, I16,
                              VectorOfLengthAndType<[2, 4], [F32]>,
                              VectorOfLengthAndType<[2, 4, 8], [F16]>,
                              VectorOfLengthAndType<[2, 4, 8], [I16]>]>>:$data,
                   AnyMemRef:$dest,
                   VectorOfLengthAndType<[3], [I32]>:$destCoordVector)> {
  let summary = "Blockwise GPU data store";
  let description = [{
    The `miopen.blockwise_store` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) to LDS (naive tensor).
  }];
  let assemblyFormat = [{
    $data `->` $dest `[` $destCoordVector `]`  attr-dict `:` type($data) `->` type($dest) `,` type($destCoordVector)
  }];
}

// threadwise_copy
def MIOpen_ThreadwiseCopyOp:
    MIOpen_Op<"threadwise_copy">,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   Variadic<I32>:$sourceAndDestCoord)> {
  let summary = "Threadwise GPU data movement";
  let description = [{
    The `miopen.threadwise` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
    - LDS (naive tensor) to register (naive tensor).
    - Register (naive tensor) to global (generic tensor).
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type($source) `,` type($dest)
    `,` type($sourceAndDestCoord)
  }];
  let verifier = [{ return ::verify(*this); }];
}

// threadwise_load
def MIOpen_ThreadwiseLoadOp:
    MIOpen_Op<"threadwise_load">,
    Arguments<(ins AnyMemRef:$source,
                   Variadic<I32>:$sourceCoord)>,
    Results<(outs Variadic<AnyTypeOf<[F32, F16, I16,
                             VectorOfLengthAndType<[2, 4], [F32]>,
                             VectorOfLengthAndType<[2, 4, 8], [F16]>,
                             VectorOfLengthAndType<[2, 4, 8], [I16]>]>>:$result)> {
  let summary = "Threadwise GPU data load";
  let description = [{
    The `miopen.threadwise_load` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to register (naive tensor).
  }];
  let assemblyFormat = [{
    $source `[` $sourceCoord `]` attr-dict `:` type($source) `,` type($sourceCoord) `->` type($result)
  }];
}

// threadwise_store
def MIOpen_ThreadwiseStoreOp:
    MIOpen_Op<"threadwise_store", [AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<AnyTypeOf<[F32, F16, I16,
                              VectorOfLengthAndType<[2, 4], [F32]>,
                              VectorOfLengthAndType<[2, 4, 8], [F16]>,
                              VectorOfLengthAndType<[2, 4, 8], [I16]>]>>:$data,
                   AnyMemRef:$dest,
                   Variadic<I32>:$destCoord)> {
  let summary = "Threadwise GPU data store";
  let description = [{
    The `miopen.threadwise_store` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) to LDS (naive tensor).
  }];
  let assemblyFormat = [{
    $data `->` $dest `[` $destCoord `]` attr-dict `:` type($data) `->` type($dest) `,` type($destCoord)
  }];
}

// threadwise_copy_v2
def MIOpen_ThreadwiseCopyV2Op:
    MIOpen_Op<"threadwise_copy_v2">,
    Arguments<(ins VectorOfRankAndType<[1], [F32, F16]>:$source,
                   AnyMemRef:$dest,
                   I32: $sourceOffset,
                   Variadic<I32>:$sourceAndDestCoord)> {
  let summary = "Threadwise GPU data movement XDLOPS version";
  let description = [{
    The `miopen.threadwise_v2` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) as vector type to global (generic tensor).

    The op is used to store C matrices, stored as vectors, back to global VMEM.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type($source) `,` type($dest)
    `,` type($sourceOffset) `,` type($sourceAndDestCoord)
  }];
}

// blockwise_gemm
def MIOpen_BlockwiseGemmOp:
    MIOpen_Op<"blockwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [3]>:$matrixA,
                   MemRefRankOf<[F32, F16, I16], [3]>:$matrixB,
                   AnyMemRef:$matrixC,
                   Index:$threadOffsetA,
                   Index:$threadOffsetB)> {
  let summary = "Blockwise GEMM non-XDLOPS version";
  let description = [{
    The `miopen.block_gemm` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall reside on register (naive tensor).
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

// blockwise_gemm_v2
def MIOpen_BlockwiseGemmV2Op:
    MIOpen_Op<"blockwise_gemm_v2">,
    Arguments<(ins MemRefOf<[F32, F16, I16]>:$matrixA,
                   MemRefOf<[F32, F16, I16]>:$matrixB,
                   Index:$waveOffsetA,
                   Index:$waveOffsetB,
                   MemRefOf<[F32, F16, I16, VectorOfLengthAndType<[2, 4, 8], [F16, I16]>]>:$bufferA,
                   MemRefOf<[F32, F16, I16, VectorOfLengthAndType<[2, 4, 8], [F16, I16]>]>:$bufferB,
                   Variadic<VectorOfRankAndType<[1], [F32, F16]>>:$vectorCs)>,
    Results<(outs Variadic<VectorOfRankAndType<[1], [F32, F16]>>: $vectorDs)> {
  let summary = "Blockwise GEMM XDLOPS version";
  let description = [{
    The `miopen.block_gemm` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall be vectors.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type($vectorDs)
  }];
}

// threadwise_gemm
def MIOpen_ThreadwiseGemmOp:
    MIOpen_Op<"threadwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [3]>:$matrixA,
                   MemRefRankOf<[F32, F16, I16], [3]>:$matrixB,
                   MemRefRankOf<[F32, F16, I16], [3]>:$matrixC)> {
  let summary = "Threadwise GEMM non-XDLOPS version";
  let description = [{
    The `miopen.threadwise_gemm` op does GEMM at thread level.
    - Matrix A and Matrix B shall reside on register (naive tensor).
    - Matrix C shall reside on LDS (naive tensor).
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

// mfma_v2
def MIOpen_MFMAV2Op:
    MIOpen_Op<"mfma_v2", [AllTypesMatch<["sourceA", "sourceB"]>,
      AllTypesMatch<["destC", "destD"]>]>,
    Arguments<(ins AnyTypeOf<[F32, VectorOfLengthAndType<[4], [F16]>, VectorOfLengthAndType<[2], [I16]>]>: $sourceA,
                   AnyTypeOf<[F32, VectorOfLengthAndType<[4], [F16]>, VectorOfLengthAndType<[2], [I16]>]>: $sourceB,
                   VectorOfRankAndType<[1], [F32, F16]>: $destC)>,
    Results<(outs VectorOfRankAndType<[1], [F32, F16]>: $destD)> {
  let summary = "XDLOPS MFMA V2";
  let description = [{
    The `miopen.mfma_v2` op is an abstraction of XDLOPS.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type($sourceA) `,` type($destD)
  }];
}

// xdlops_gemm_V2
def MIOpen_XdlopsGemmV2Op:
    MIOpen_Op<"xdlops_gemm_v2">,
    Arguments<(ins MemRefOf<[F32, F16, I16]>:$matrixA,
                   MemRefOf<[F32, F16, I16]>:$matrixB,
                   Index:$waveOffsetA,
                   Index:$waveOffsetB,
                   MemRefOf<[F32, F16, I16, VectorOfLengthAndType<[2, 4, 8], [F16, I16]>]>:$bufferA,
                   MemRefOf<[F32, F16, I16, VectorOfLengthAndType<[2, 4, 8], [F16, I16]>]>:$bufferB,
                   Variadic<VectorOfRankAndType<[1], [F32, F16]>>:$vectorCs)>,
    Results<(outs Variadic<VectorOfRankAndType<[1], [F32, F16]>>:$vectorDs)> {
  let summary = "XDLOPS GEMM V2";
  let description = [{
    The `miopen.xdlops_gemm_v2` op is an abstraction of doing GEMM based on XDLOPS.
    It would employ a series of `miopen.mfma` operations.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type(results)
  }];
}

def MIOpen_DataConvertOp :
    MIOpen_Op<"data_convert">,
    Arguments<(ins F32:$in)>,
    Results<(outs I16:$out)> {
  let summary = "data type conversion";
  let description = [{
    The `miopen.data_convert` op will convert f32 to bf16, but we use i16 to replace bf16 here.
  }];
  let assemblyFormat = [{
    $in attr-dict `:` type($in) `to` type($out)
  }];
}
#endif // MIOPEN_OPS
