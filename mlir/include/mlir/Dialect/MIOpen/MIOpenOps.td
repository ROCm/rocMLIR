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

include "mlir/Dialect/MIOpen/MIOpenAttrDefs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for MIOpen dialect ops.
class MIOpen_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MIOpen_Dialect, mnemonic, traits> {
  let extraClassDeclaration = [{
  }];
}

def TransformsArrayAttr : TypedArrayAttrBase<MIOpen_TransformsAttr,
  "Coordinate transforms array attribute, giving the sequence of transforms applicable to a value, uppermost to lowermost"> {}

def ArgTransformsAttr : TypedArrayAttrBase<TransformsArrayAttr,
  "An array of arrays of transformations, one per operation argument">;

class ArgTransforms<int n> : Confined<ArgTransformsAttr, [ArrayCount<n>]>;

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
    Arguments<(ins AnyMemRef:$input, TransformsArrayAttr:$transforms)>,
    Results<(outs AnyMemRef:$output)> {
  let summary = "Tensor transformation";
  let description = [{
    Create a viee `output` of the tensor `input` with the same element type
    such that, when the coordinates used to index into `output` are passed through
    the coordinate transformations `transforms` from left to right,
    they become coordinates into the `input` tensor.
  }];
  let builders = [
   // Custom builder to populate bounds of input and output memrefs as attributes.
   OpBuilder<(ins "Value":$input, "TransformsAttr":$transforms,
    CArg<"unsigned", "0">:$memorySpace),
   [{
     $_state.addOperands({input});
     MemRefType lowerType = input.getType().template cast<MemRefType>();
     Type elemType = lowerType.getElementType();
     AffineMap lowerMap = lowerType.getLayout().getAffineMap();
     AffineMap composed = transforms.getMap().getAffineMap().compose(lowerMap);
     MemRefType upperType = MemRefType::get(transforms.getUpperBounds(), elemType,
      composed, memorySpace);
     $_state.addAttribute("transforms", $_builder.getArrayAttr({transforms}));
     $_state.addTypes(upperType);
   }]>,
  ];
  let assemblyFormat = [{
    $input `by` $transforms attr-dict `:` type($input) `to` type($output)
  }];
}

def MIOpen_GridwiseGemmOp :
    MIOpen_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [3]>:$a,
                   MemRefRankOf<[F32, F16, I16], [3]>:$b,
                   MemRefRankOf<[F32, F16, I16], [3]>:$c,
                   BoolArrayAttr:$aOobDims,
                   BoolArrayAttr:$bOobDims,
                   BoolArrayAttr:$cOobDims,
                   ArgTransforms<3>:$transforms,
                   MIOpen_PaddingInfoAttr:$paddingInfo)> {
  let summary = "Gridwise GEMM";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];

  let builders = [
  OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c,
    "ArrayAttr":$aOobDims, "ArrayAttr":$bOobDims, "ArrayAttr":$cOobDims,
    "PaddingInfoAttr":$paddingInfo, "ArrayRef<NamedAttribute>":$extraAttrs), [{
      $_state.addOperands({a, b, c});
      $_state.addAttribute(aOobDimsAttrName($_state.name), aOobDims);
      $_state.addAttribute(bOobDimsAttrName($_state.name), bOobDims);
      $_state.addAttribute(cOobDimsAttrName($_state.name), cOobDims);
      $_state.addAttribute(transformsAttrName($_state.name),
        noTransformsArray($_builder, 3));
      $_state.addAttribute(paddingInfoAttrName($_state.name), paddingInfo);
      $_state.addAttributes(extraAttrs);
      $_state.addTypes({});
    }]>
  ];
}

// gridwise_gemm_v2
def MIOpen_GridwiseGemmV2Op :
    MIOpen_Op<"gridwise_gemm_v2">,
    Arguments<(ins MemRefRankOf<[F32, F16, I16], [3]>:$a,
                   MemRefRankOf<[F32, F16, I16], [3]>:$b,
                   MemRefRankOf<[F32, F16, I16], [3]>:$c,
                   BoolArrayAttr:$aOobDims,
                   BoolArrayAttr:$bOobDims,
                   BoolArrayAttr:$cOobDims,
                   ArgTransforms<3>:$transforms,
                   MIOpen_PaddingInfoAttr:$paddingInfo,
                   InMemoryDataOperation:$storeOperation)> {
  let summary = "Gridwise GEMM V2";
  let description = [{
    The `miopen.gridwise_gemm` op computes gridwise GEMM with XDLOPS.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];

    let builders = [
  OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c,
    "ArrayAttr":$aOobDims, "ArrayAttr":$bOobDims, "ArrayAttr":$cOobDims,
    "PaddingInfoAttr":$paddingInfo, "InMemoryDataOperation":$storeOperation,
    "ArrayRef<NamedAttribute>":$extraAttrs), [{
      $_state.addOperands({a, b, c});
      $_state.addAttribute(aOobDimsAttrName($_state.name), aOobDims);
      $_state.addAttribute(bOobDimsAttrName($_state.name), bOobDims);
      $_state.addAttribute(cOobDimsAttrName($_state.name), cOobDims);
      $_state.addAttribute(transformsAttrName($_state.name),
        noTransformsArray($_builder, 3));
      $_state.addAttribute(paddingInfoAttrName($_state.name), paddingInfo);
      $_state.addAttribute(storeOperationAttrName($_state.name),
        $_builder.getI32IntegerAttr(static_cast<uint32_t>(storeOperation)));
      $_state.addAttributes(extraAttrs);
      $_state.addTypes({});
    }]>
  ];

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

// blockwise_load
def MIOpen_BlockwiseLoadOp:
    MIOpen_Op<"blockwise_load">,
    Arguments<(ins AnyMemRef:$source,
                  ArgTransforms<1>:$transforms,
                  MIOpen_PaddingInfoAttr:$paddingInfo,
                  BoolArrayAttr:$oobDims,
                  Variadic<Index>:$sourceCoord)>,
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
    $source `[` $sourceCoord `]`
      `with` $transforms attr-dict `:` type(operands)  `->` type(results)
  }];
}

// blockwise_store
def MIOpen_BlockwiseStoreOp:
    MIOpen_Op<"blockwise_store", [AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<AnyTypeOf<[F32, F16, I16,
                              VectorOfLengthAndType<[2, 4], [F32]>,
                              VectorOfLengthAndType<[2, 4, 8], [F16]>,
                              VectorOfLengthAndType<[2, 4, 8], [I16]>]>>:$data,
                   AnyMemRef:$dest,
                   Variadic<Index>:$destCoord)> {
  let summary = "Blockwise GPU data store";
  let description = [{
    The `miopen.blockwise_store` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) to LDS (naive tensor).
  }];
  let assemblyFormat = [{
    $data `->` $dest `[` $destCoord `]` attr-dict
    `:` type($data) `->` type($dest) `,` type($destCoord)
  }];
}

// threadwise_copy
def MIOpen_ThreadwiseCopyOp:
    MIOpen_Op<"threadwise_copy", [SameVariadicOperandSize]>,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   ArgTransforms<2>:$transforms,
                   MIOpen_PaddingInfoAttr:$paddingInfo,
                   BoolArrayAttr:$oobDims,
                   IndexAttr:$globalArg,
                   Variadic<Index>:$sourceCoord,
                   Variadic<Index>:$destCoord)> {
  let summary = "Threadwise GPU data movement";
  let description = [{
    The `miopen.threadwise_copy` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to LDS (naive tensor).
    - LDS (naive tensor) to register (naive tensor).
    - Register (naive tensor) to global (generic tensor).
  }];
  let assemblyFormat = [{
    $source `[` $sourceCoord `]` `->` $dest `[` $destCoord `]`
    `with` $transforms attr-dict
    `:` type($source) `,` type($sourceCoord) `->` type($dest) `,` type($destCoord)
  }];
  let verifier = [{ return ::verify(*this); }];
}

// threadwise_load
def MIOpen_ThreadwiseLoadOp:
    MIOpen_Op<"threadwise_load">,
    Arguments<(ins AnyMemRef:$source,
                   ArgTransforms<1>:$transforms,
                   MIOpen_PaddingInfoAttr:$paddingInfo,
                   BoolArrayAttr:$oobDims,
                   Variadic<Index>:$sourceCoord)>,
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
    $source `[` $sourceCoord `]`
    `with` $transforms
    attr-dict `:` type($source) `,` type($sourceCoord) `->` type($result)
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
                   Variadic<Index>:$destCoord)> {
  let summary = "Threadwise GPU data store";
  let description = [{
    The `miopen.threadwise_store` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) to LDS (naive tensor).
  }];
  let assemblyFormat = [{
    $data `->` $dest `[` $destCoord `]` attr-dict
    `:` type($data) `->` type($dest) `,` type($destCoord)
  }];
}

// threadwise_copy_v2
def MIOpen_ThreadwiseCopyV2Op:
    MIOpen_Op<"threadwise_copy_v2", [AttrSizedOperandSegments]>,
    Arguments<(ins VectorOfRankAndType<[1], [F32, F16]>:$source,
                   AnyMemRef:$dest,
                   ArgTransforms<2>:$transforms,
                   MIOpen_PaddingInfoAttr:$paddingInfo,
                   InMemoryDataOperation:$dataOperation,
                   BoolArrayAttr:$destOobDims,
                   IndexAttr:$sourceOffset,
                   Variadic<Index>:$sourceCoord,
                   Variadic<Index>:$destCoord)> {
  let summary = "Threadwise GPU data movement XDLOPS version";
  let description = [{
    The `miopen.threadwise_v2` op moves data on GPU. Following movements are
    allowed:
    - Register (naive tensor) as vector type to global (generic tensor).

    The op is used to store C matrices, stored as vectors, back to global VMEM.
  }];
  let assemblyFormat = [{
    $source `[` $sourceCoord `]`
    `->` $dest `[` $destCoord `]`
    `with` $transforms attr-dict
    `:` type($source) `,` type($sourceCoord) `->`
    type($dest) `,` type($destCoord)
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

// TODO(kdrewnia): upstream these
// Whether the vector's length is divisible by `divisor`
class IsVectorOfDivisibleLengthPred<int divisor> :
  And<[IsVectorTypePred,
       CPred<[{$_self.cast<::mlir::VectorType>().getNumElements()
                           % }] # divisor # [{ == 0 }]>]>;

// Any vector where the number of elements is from the given
// `allowedLengths` list
class VectorOfDivisibleLength<int divisor> : Type<
  IsVectorOfDivisibleLengthPred<divisor>,
  " with length divisble by " # divisor,
  "::mlir::VectorType">;
class VectorOfDivisibleLengthAndType<int  divisor,
                          list<Type> allowedTypes> : Type<
  And<[VectorOf<allowedTypes>.predicate,
       VectorOfDivisibleLength<divisor>.predicate]>,
  VectorOf<allowedTypes>.summary # VectorOfDivisibleLength<divisor>.summary,
  "::mlir::VectorType">;

defvar swizzleGroupSize = 4;
def MIOpen_InWarpTransposeOp :
    MIOpen_Op<"in_warp_transpose", [AllTypesMatch<["vector", "res"]>]>,
    Arguments<(ins VectorOfDivisibleLengthAndType<swizzleGroupSize,
                                                          [F32, I32]>:$vector,
                  Index:$laneId,
                  Confined<I32Attr, [IntMaxValue<swizzleGroupSize>]>:$size,
                  DefaultValuedAttr<Confined<I32ArrayAttr,
                        [ArrayCount<swizzleGroupSize>]>,
                    "{0, 1, 2, 3}">:$inGroupPerm)>,
    Results<(outs VectorOfDivisibleLengthAndType<swizzleGroupSize, [F32, I32]>:$res)> {
  let summary = "Transpose blocks of data distributed accross a warp";
  let description = [{
    `MIOpen.in_warp_transpose` takes a vector representing a matrix of values
    stored accross the threads in a warp and transposes `size`x`size` blocks
    of this matrix. The rows of the output (if we regard each thread's vector
    as a row in the matrix) are further permuted with the `inGroupPerm`
    permutation.

    It is an ***unchecked invariant*** that laneId contanes the ID of the current
    lane with the wave, and thus is an integer in [0, waveSize).
  }];

  let verifier = [{ return ::verify(*this); }];
  let assemblyFormat = [{
    attr-dict $vector `,` $laneId `:` type($vector) `,` type($laneId)
  }];

  let extraClassDeclaration = "static constexpr size_t swizzleGroupSize = "
    # !cast<string>(swizzleGroupSize) # ";\n";
}

#endif // MIOPEN_OPS
