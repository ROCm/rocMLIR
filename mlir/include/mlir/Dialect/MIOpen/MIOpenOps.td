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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for MIOpen dialect ops.
class MIOpen_Op<string mnemonic, list<Trait> traits = []> :
    Op<MIOpen_Dialect, mnemonic, traits> {
  let extraClassDeclaration = [{
  }];
}

def TransformMapArrayAttr : TypedArrayAttrBase<MIOpen_TransformMapAttr,
  "Coordinate transforms array attribute, giving the sequence of transform maps applicable to a value, uppermost to lowermost"> {}

def ArgTransformsAttr : TypedArrayAttrBase<TransformMapArrayAttr,
  "An array of arrays of transformations, one per operation argument">;

class ArgTransforms<int n> : Confined<ArgTransformsAttr, [ArrayCount<n>]>;

def MIOpen_Conv2DOp :
    MIOpen_Op<"conv2d">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16, I8], [5]>:$filter,
                   MemRefRankOf<[F32, F16, BF16, I8], [5]>:$input,
                   MemRefRankOf<[F32, F16, BF16, I32], [5]>:$output)> {
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
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [5]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [5]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [5]>:$output)> {
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
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [5]>:$filter,
                   MemRefRankOf<[F32, F16, BF16], [5]>:$input,
                   MemRefRankOf<[F32, F16, BF16], [5]>:$output,
                   Optional<MemRefRankOf<[F32], [5]>>:$workspace)> {
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
    MIOpen_Op<"transform", [NoSideEffect, ViewLikeOpInterface]>,
    Arguments<(ins AnyMemRef:$input, TransformMapArrayAttr:$transforms)>,
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
   OpBuilder<(ins "Value":$input, "TransformMapAttr":$transforms,
    CArg<"unsigned", "0">:$memorySpace),
   [{
     $_state.addOperands({input});
     MemRefType lowerType = input.getType().template cast<MemRefType>();
     Type elemType = lowerType.getElementType();
     AffineMap lowerMap = lowerType.getLayout().getAffineMap();
     AffineMap composed = lowerMap.compose(transforms.getMap().getAffineMap());
     MemRefType upperType = MemRefType::get(transforms.getUpperBounds(), elemType,
      composed, memorySpace);
     $_state.addAttribute("transforms", $_builder.getArrayAttr({transforms}));
     $_state.addTypes(upperType);
   }]>,
  ];
  let assemblyFormat = [{
    $input `by` $transforms attr-dict `:` type($input) `to` type($output)
  }];

  let extraClassDeclaration = [{
    Value getViewSource() { return input(); }
  }];
}

def MIOpen_GridwiseGemmOp :
    MIOpen_Op<"gridwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [3, 4]>:$a,
                   MemRefRankOf<[F32, F16, BF16], [3, 4]>:$b,
                   MemRefRankOf<[F32, F16, BF16], [3, 4]>:$c,
                   BoolArrayAttr:$aOobDims,
                   BoolArrayAttr:$bOobDims,
                   BoolArrayAttr:$cOobDims,
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
      $_state.addAttribute(paddingInfoAttrName($_state.name), paddingInfo);
      $_state.addAttributes(extraAttrs);
      $_state.addTypes({});
    }]>
  ];
}

// gridwise_gemm_v2
def MIOpen_GridwiseGemmV2Op :
    MIOpen_Op<"gridwise_gemm_v2">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16, I8], [3, 4]>:$a,
                   MemRefRankOf<[F32, F16, BF16, I8], [3, 4]>:$b,
                   MemRefRankOf<[F32, F16, BF16, I32], [3, 4]>:$c,
                   BoolArrayAttr:$aOobDims,
                   BoolArrayAttr:$bOobDims,
                   BoolArrayAttr:$cOobDims,
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

// extract_slice
def MIOpen_ExtractSliceOp :
    MIOpen_Op<"extract_slice", [NoSideEffect,
      AllElementTypesMatch<["vector", "result"]>]>,
    Arguments<(ins
      VectorOfRank<[1]>:$vector,
      Index:$coord)>,
    Results<(outs AnyType:$result)> {
  let summary = "Extract a slice from a vector";

  let description = [{
    Extracts `len(result)` contiguous elements starting at `coord`  in `vector`.
    If `result` is a scalar type, the length is 1 and this is just `vector.extractelement`.
    If this causes an out of bounds read, the result is undefined.
  }];

  let assemblyFormat = [{
    attr-dict $vector `[` $coord `]` `:` type($vector) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

// insert_slice
def MIOpen_InsertSliceOp :
    MIOpen_Op<"insert_slice", [NoSideEffect,
      AllElementTypesMatch<["source", "dest"]>,
      AllTypesMatch<["dest", "result"]>]>,
    Arguments<(ins
      AnyType:$source,
      VectorOfRank<[1]>:$dest,
      Index:$coord)>,
    Results<(outs VectorOfRank<[1]>:$result)> {
  let summary = "Insert a slice into a vector";

  let description = [{
    Inserts  `source` into contigious indices of `dest`, starting at `coord`.
    If `source` is a scalar type, this is just `vector.insertelement`.
    If the indices into `dest` go out of bounds, the result is undefined.
  }];

  let assemblyFormat = [{
    attr-dict $source `->` $dest `[` $coord `]` `:` type($source) `->` type($dest)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

// yield
def MIOpen_YieldOp :
    MIOpen_Op<"yield", [NoSideEffect, Terminator, ReturnLike]> {
  let summary = "yield for MIOpen loops";

  let description = [{
    Op that terminates the MIOpen looping constructs (currently transforming for).

    If the loops produce any values, then this op's arguments should match the types
    of those values. Otherwise, this op is implicit and can be omitted.

    Note that, in a transforming_for, you don't need to yield back the lower coordinates,
    as these are managed implicitly.
  }];

  // Yoinked from affine.yield
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// Transforming for
def MIOpen_TransformingForOp :
    MIOpen_Op<"transforming_for", [RecursiveSideEffects,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>,
      AttrSizedOperandSegments,
      SingleBlockImplicitTerminator<"::mlir::miopen::YieldOp">]>,
    Arguments<(ins Variadic<Index>:$upperInits,
      ArgTransformsAttr:$transforms,
      Variadic<AnyType>:$iterInits,
      IndexArrayAttr:$bounds,
      I32ElementsAttr:$lowerStarts,
      OptionalAttr<UnitAttr>:$useIndexDiffs,
      OptionalAttr<UnitAttr>:$forceUnroll)>,
    Results<(outs Variadic<AnyType>:$results)> {
  let summary = "for loop with coordinate transforms";
  let description = [{
    Loops over a rectangular region, using each point as the offset in upper coordinate space(s)
    before applying coordinate transforms.

    The loop
    ```mlir
    %res = miopen.transforming_for {forceUnroll, useIndexDiffs} (%a1, %a2, %a3) = [#transform_map1](%i0, %j0), (%i, %j) = [](%cst0, %cst0)
      iter_args(%arg0 = %c0 : index) bounds [4, 8] {
      %v = miopen.buffer_load {...} %place[%a1, %a2, %a3]
      miopen.buffer_store {...} %v -> %buffer[%i, %j]
      %cont = addi %arg0, %c1
      miopen.yield %cont : index
    }
    ```
    will lower to
    ```mlir
    %a0, %a1, %a2 = miopen.compute_transforms [#transform_map1](%x, %y)
    %i, %j = miopen.compute_transforms [](%c0, %c0)
    %a0_0_0, %a1_0_0 = miopen.index_diff #transform_map1[0, 0] (%x, %y) -> (%a0, %a1, %a2)
    %i_0_0 = addi %i, %cst0
    %j_0_0 = addi %j, %cst0
    %v_0_0 miopen.buffer_load {...} %place[%a0_0_0, %a1_0_0, %a2_0_0]
    miopen.buffer_store {...} %v -> %buffer[%i_0_0, %j_0_0]
    %res_0_0 = addi %c0, %c1 : inedx
    %a0_0_1, %a1_0_1, %a2_0_1 = miopen.index_diff #transform_map1[0, 1] (%x, %y) -> (%a0, %a1, %a2)
    ...
    %res_3_7 = addi %res_3_6, %c1 : index
  ```
  Or, if no unrolling or index diffs are called for
  ```mlir
  %res = affine for %d0 = 0 to 4 iter_args(%arg0 = %c0) {
    %res0 = affine.for %d1 = 0 to 8 iter_args(%arg1 = %arg0) {
      %upper_0_0 = addi %x, %d0
      %upper_0_1 = addi %y, %d1
      %i = addi %c0, %d0
      %j = addi %c0, %d1
      %a0, %a1, %a2 = affine.apply (#transform_map1.map) (%upper_0_0, %upper_0_1)
      %v = miopen.buffer_load {...} %place[%a0, %a1, %a2]
      miopen.buffer_store {...} %buffer[%i, %j]
      %cont = addi %arg1, %c1 : index
      affine.yield %cont : index
    }
    affine.yield %res0 : index
  }
  ```

  The purpose of this looping construct is to remove redundancy and make the
  application of coordinate transformations more explicit in the IR.
  }];

  let regions = (region SizedRegion<1>:$region);

  DerivedAttr upperLen = DerivedAttr<"uint32_t", [{
    return upperInits().size() / transforms().size();
  }], [{ $_builder.getI32IntegerAttr($_self) }]>;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayRef<Attribute>":$transforms,
    "ArrayRef<int64_t>":$bounds, "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "llvm::None">:$iterArgs)>,

    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayRef<Attribute>":$transforms,
    "ArrayAttr":$bounds, "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "llvm::None">:$iterArgs)>,

    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayAttr":$transforms,
    "ArrayRef<int64_t>":$bounds, "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "llvm::None">:$iterArgs)>,

    OpBuilder<(ins "ArrayRef<ValueRange>":$inits,
    "ArrayAttr":$transforms,
    "ArrayAttr":$bounds, "bool":$forceUnroll, "bool":$useIndexDiffs,
    CArg<"ValueRange", "llvm::None">:$iterArgs)>];

  let extraClassDeclaration = [{
    uint32_t lowerStart(uint32_t n) {
      return *(lowerStarts().getValues<uint32_t>().begin() + n);
    }

    Block::BlockArgListType getLowerCoords(uint32_t domain) {
      uint32_t start = lowerStart(domain);
      uint32_t end = lowerStart(domain + 1);
      return getBody()->getArguments().slice(start, end - start);
    }
    Block::BlockArgListType getLowerCoords() {
      assert(lowerStarts().size() == 2 && "Ambiguous call to getLowerCoords() with multple iteration domains");
      return getBody()->getArguments().take_front(lowerStart(1));
    }

    Block::BlockArgListType getIterArgs() {
      return getBody()->getArguments().drop_front(lowerStart(lowerStarts().size() - 1));
    }

    ValueTypeRange<Block::BlockArgListType> getIterArgTypes() {
      return ValueTypeRange<Block::BlockArgListType>(getIterArgs());
    }

    Operation::operand_range getUpperInits(uint32_t domain) {
      uint32_t theUpperLen = upperLen();
      return upperInits().slice(domain * theUpperLen, theUpperLen);
    }

    ArrayAttr getTransforms(uint32_t domain) {
      return transforms()[domain].cast<ArrayAttr>();
    }

    uint32_t domains() {
      return transforms().size();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// index_diff_update
def MIOpen_IndexDiffUpdateOp :
    MIOpen_Op<"index_diff_update", [NoSideEffect, AttrSizedOperandSegments,
      SameVariadicResultSize, AllTypesMatch<["lowerIndices", "lowerDiff"]>]>,
    Arguments<(ins MIOpen_TransformMapAttr:$map,
      Variadic<Index>:$upperDiffs,
      Variadic<Index>:$lowerOrig)>,
    Results<(outs Variadic<Index>:$lowerIndices,
      Variadic<Index>:$lowerDiff)> {
  let summary = "Compute change in lower indices (map output) using index diffs";
  let description = [{
    `miopen.index_diff_update' computes the change in the output of a transform_map
    from adding `upperDiffs` to the inputs that produced the output `lowerOrig'.

    The operation returns both `lowerIndices`, the new indices, and `lowerDiff`,
    the change in the lower indices compared to `lowerOrig'.
  }];

  let builders = [
    OpBuilder<(ins "TransformMapAttr":$transform, "ValueRange":$upperDiffs,
      "ValueRange":$lowerOrig)>
  ];

  let assemblyFormat = [{
    attr-dict $map `(` $upperDiffs `)` `+` `(` $lowerOrig `)` `:`
      type($lowerIndices)
  }];
  let hasVerifier = 1;
}

// buffer_load
def MIOpen_BufferLoadOp :
    MIOpen_Op<"buffer_load">,
    Arguments<(ins Arg<MemRefOf<[F32, F16, BF16, I8, I32]>,
        "buffer to load from", [MemRead]>:$source,
      BoolArrayAttr:$oobDims,
      Variadic<Index>:$coords)>,
    Results<(outs AnyTypeOf<[F32, F16, BF16, I8, I32,
              VectorOfLengthAndType<[2, 4], [F32]>,
              VectorOfLengthAndType<[2, 4, 8], [F16]>,
              VectorOfLengthAndType<[2, 4, 8], [BF16]>,
              VectorOfLengthAndType<[4], [I8]>,
              VectorOfLengthAndType<[2, 4], [I32]>]>:$result)> {
  let summary = "Load data from a global buffer";

  let description = [{
    miopen.buffer_load uses, if necessary, GPU buffer load intrinsics to
    ensure that out of bounds loads return 0 instead of a garbage value.

    The bounds on each coordinate are the size of the memref in that coordinate.

    This op can perform vector reads.

    The memref must be in global memory (memory space 0).

    `oobDims` are the dimensions to be checked for out of bounds indices.
  }];
  let assemblyFormat = [{
    $source `[` $coords `]` attr-dict
    `:` type($source) `,` type($coords) `->` type($result)
  }];
  let hasVerifier = 1;
}

// buffer_store
def MIOpen_BufferStoreOp :
    MIOpen_Op<"buffer_store", []>,
    Arguments<(ins AnyTypeOf<[F32, F16, BF16, I8, I32,
        VectorOfLengthAndType<[2, 4], [F32]>,
        VectorOfLengthAndType<[2, 4, 8], [F16]>,
        VectorOfLengthAndType<[2, 4, 8], [BF16]>,
        VectorOfLengthAndType<[4], [I8]>,
        VectorOfLengthAndType<[2, 4], [I32]>]>:$data,
      Arg<MemRefOf<[F32, F16, BF16, I8, I32]>,
        "Buffer to store to", [MemWrite]>:$dest,
      BoolArrayAttr:$oobDims,
      Variadic<Index>:$coords,
      OptionalAttr<MIOpen_PaddingInfoAttr>:$paddingInfo,
      OptionalAttr<InMemoryDataOperation>:$dataOperation)> {
  let summary = "Store data to a global buffer";

  let description = [{
    `miopen.buffer_store` stores data to a global buffer, ignoring out of bounds writes.

    The bounds on a write are the dimensions of the memref, though only the
    dimensions specified in `oobDims` are checked before the write is issued.

    Writes that our out of bounds are ignored.

    This op supports vector writes and uses buffer store intrinsics if needed.

    `paddingInfo` is used to apply backwards data store hacks if they're needed.

    The buffer must reside in global memory.
  }];
  let assemblyFormat = [{
    $data `->` $dest `[` $coords `]` attr-dict
    `:` type($data) `->` type($dest) `,` type($coords)
  }];
  let hasVerifier = 1;
}

// in_bounds_load
def MIOpen_InBoundsLoadOp :
    MIOpen_Op<"in_bounds_load", [AllElementTypesMatch<["source", "result"]>]>,
    Arguments<(ins Arg<AnyMemRef,
        "buffer to load from", [MemRead]>:$source,
      Variadic<Index>:$coords)>,
    Results<(outs AnyType:$result)> {
  let summary = "Load one or more contiguous items from `source`";

  let description = [{
    miopen.in_bounds_loads reads either one scalar value or a vector of values
    from a memref. The memref can be in any memory space. The read begins at
    `coords`.
  }];
  let assemblyFormat = [{
    $source `[` $coords `]` attr-dict
    `:` type($source) `,` type($coords) `->` type($result)
  }];
  let hasVerifier = 1;
}

// in_bounds_store
def MIOpen_InBoundsStoreOp :
    MIOpen_Op<"in_bounds_store", [AllElementTypesMatch<["data", "dest"]>]>,
    Arguments<(ins AnyType:$data,
      Arg<AnyMemRef,
        "Buffer to store to", [MemWrite]>:$dest,
      Variadic<Index>:$coords)> {
  let summary = "Store one or more items to contiguous indices in `dest`";

  let description = [{
    `miopen.in_bounds_store` stores the item or items in `data` (which can be
    a scalar or a vector) to the buffer `dest`, starting at `coords`.
    The memref can be in any memory space.
    This op assumes that all indices will land in bounds.
  }];
  let assemblyFormat = [{
    $data `->` $dest `[` $coords `]` attr-dict
    `:` type($data) `->` type($dest) `,` type($coords)
  }];
  let hasVerifier = 1;
}


// TODO(kdrewnia): Remove these - they're left in to ensure a smooth transition
// of fusion passes to the transforming_for structure

// threadwise_copy
def MIOpen_ThreadwiseCopyOp:
    MIOpen_Op<"threadwise_copy", [SameVariadicOperandSize]>,
    Arguments<(ins AnyMemRef:$source,
                   AnyMemRef:$dest,
                   IndexArrayAttr:$bounds,
                   ArgTransforms<2>:$transforms,
                   MIOpen_PaddingInfoAttr:$paddingInfo,
                   BoolArrayAttr:$oobDims,
                   Variadic<Index>:$sourceCoord,
                   Variadic<Index>:$destCoord,
                   OptionalAttr<BoolAttr>:$legacyLoad,
                   OptionalAttr<BoolAttr>:$legacyStore)> {
  let summary = "Threadwise GPU data movement";
  let description = [{
    The `miopen.threadwise_copy` op moves data on GPU. Following movements are
    allowed:
    - Global (generic tensor) to register (naive tensor).
    - Register (naive tensor) to global (generic tensor).
  }];
  let assemblyFormat = [{
    $source `[` $sourceCoord `]` `->` $dest `[` $destCoord `]`
    `with` $transforms attr-dict
    `:` type($source) `,` type($sourceCoord) `->` type($dest) `,` type($destCoord)
  }];
  let verifier = [{ return ::verify(*this); }];
}

// threadwise_copy_v2
def MIOpen_ThreadwiseCopyV2Op:
    MIOpen_Op<"threadwise_copy_v2", [AttrSizedOperandSegments]>,
    Arguments<(ins VectorOfRankAndType<[1], [F32, F16, I32]>:$source,
                   AnyMemRef:$dest,
                   IndexArrayAttr:$bounds,
                   ArgTransforms<2>:$transforms,
                   MIOpen_PaddingInfoAttr:$paddingInfo,
                   InMemoryDataOperation:$dataOperation,
                   BoolArrayAttr:$destOobDims,
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
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [3, 4]>:$matrixA,
                   MemRefRankOf<[F32, F16, BF16], [3, 4]>:$matrixB,
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
    Arguments<(ins MemRefOf<[F32, F16, BF16, I8]>:$matrixA,
                   MemRefOf<[F32, F16, BF16, I8]>:$matrixB,
                   IndexAttr:$ldsBufferOffsetA,
                   IndexAttr:$ldsBufferOffsetB,
                   Index:$waveOffsetA,
                   Index:$waveOffsetB,
                   MemRefOf<[F32, F16, BF16, I8,
                             VectorOfLengthAndType<[2, 4], [F32]>,
                             VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                             VectorOfLengthAndType<[4], [I8]>]>:$bufferA,
                   MemRefOf<[F32, F16, BF16, I8,
                             VectorOfLengthAndType<[2, 4], [F32]>,
                             VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                             VectorOfLengthAndType<[4], [I8]>]>:$bufferB,
                   Variadic<VectorOfRankAndType<[1], [F32, F16, I32]>>:$vectorCs)>,
    Results<(outs Variadic<VectorOfRankAndType<[1], [F32, F16, I32]>>: $vectorDs)> {
  let summary = "Blockwise GEMM XDLOPS version";
  let description = [{
    The `miopen.block_gemm` op does GEMM at workgroup (block) level.
    - Matrix A and Matrix B shall reside on LDS (naive tensor).
    - Matrix C shall be vectors.

    There are only two slots for argument transforms since the buffers are
    not slices of a larger object.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type($vectorDs)
  }];
}

// threadwise_gemm
def MIOpen_ThreadwiseGemmOp:
    MIOpen_Op<"threadwise_gemm">,
    Arguments<(ins MemRefRankOf<[F32, F16, BF16], [3, 4]>:$matrixA,
                   MemRefRankOf<[F32, F16, BF16], [3, 4]>:$matrixB,
                   MemRefRankOf<[F32, F16, BF16], [3, 4]>:$matrixC)> {
  let summary = "Threadwise GEMM non-XDLOPS version";
  let description = [{
    The `miopen.threadwise_gemm` op does GEMM at thread level.
    - Matrix A and Matrix B shall reside on register (naive tensor).
    - Matrix C is in registers.
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

// mfma_v2
def MIOpen_MFMAV2Op:
    MIOpen_Op<"mfma_v2", [AllTypesMatch<["sourceA", "sourceB"]>,
      AllTypesMatch<["destC", "destD"]>]>,
    Arguments<(ins AnyTypeOf<[F32,
                              VectorOfLengthAndType<[4], [F16]>,
                              VectorOfLengthAndType<[2], [BF16]>,
                              VectorOfLengthAndType<[4], [I8]>]>: $sourceA,
                   AnyTypeOf<[F32,
                              VectorOfLengthAndType<[4], [F16]>,
                              VectorOfLengthAndType<[2], [BF16]>,
                              VectorOfLengthAndType<[4], [I8]>]>: $sourceB,
                   VectorOfRankAndType<[1], [I32, F32, F16]>: $destC)>,
    Results<(outs VectorOfRankAndType<[1], [I32, F32, F16]>: $destD)> {
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
    Arguments<(ins MemRefOf<[F32, F16, BF16, I8]>:$matrixA,
                   MemRefOf<[F32, F16, BF16, I8]>:$matrixB,
                   IndexAttr:$ldsBufferOffsetA,
                   IndexAttr:$ldsBufferOffsetB,
                   Index:$waveOffsetA,
                   Index:$waveOffsetB,
                   MemRefOf<[F32, F16, BF16, I8,
                             VectorOfLengthAndType<[2, 4], [F32]>,
                             VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                             VectorOfLengthAndType<[4], [I8]>]>:$bufferA,
                   MemRefOf<[F32, F16, BF16, I8,
                             VectorOfLengthAndType<[2, 4], [F32]>,
                             VectorOfLengthAndType<[2, 4, 8], [F16, BF16]>,
                             VectorOfLengthAndType<[4], [I8]>]>:$bufferB,
                   Variadic<VectorOfRankAndType<[1], [F32, F16, I32]>>:$vectorCs)>,
    Results<(outs Variadic<VectorOfRankAndType<[1], [F32, F16, I32]>>:$vectorDs)> {
  let summary = "XDLOPS GEMM V2";
  let description = [{
    The `miopen.xdlops_gemm_v2` op is an abstraction of doing GEMM based on XDLOPS.
    It would employ a series of `miopen.mfma` operations.

    Matrices A and B reside in LDS, the buffers live in registers, C is a vector
  }];
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type(results)
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
