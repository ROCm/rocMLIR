// RUN: rocmlir-opt -mhal-emulate-narrow-type --verify-diagnostics \
// RUN:   --split-input-file %s -o /dev/null

// COM: Exercises the four pattern-rejection branches of
// COM: ExtractStridedMetadataFromOldFuncArgs::matchAndRewrite in
// COM: external/mlir-hal/lib/Dialect/MHAL/Transforms/EmulateNarrowType.cpp.
// COM: When this helper pattern rejects a memref.extract_strided_metadata op,
// COM: no other pattern in the second applyPartialConversion can legalize it
// COM: (because the type converter has marked it illegal for the un-converted
// COM: memref<...xi4> type). That makes applyPartialConversion fail and the
// COM: pass emit the "failed to legalize operation
// COM: 'memref.extract_strided_metadata' that was explicitly marked illegal"
// COM: diagnostic, which simultaneously covers the signalPassFailure path
// COM: taken when the second conversion fails.

// COM: ---- 1: getBaseBuffer().use_empty() == false. The pattern bails when
// COM: the base_buffer result is consumed.

func.func @base_buffer_used(%arg0: memref<8xi4>) -> memref<i4> {
  // expected-error @+1 {{failed to legalize operation 'memref.extract_strided_metadata'}}
  %base, %off, %sz, %st = memref.extract_strided_metadata %arg0
    : memref<8xi4> -> memref<i4>, index, index, index
  return %base : memref<i4>
}

// -----

// COM: ---- 2: !sourceType.hasStaticShape() (first OR clause of the static-
// COM: shape / identity-layout reject).

func.func @dynamic_shape(%arg0: memref<?xi4>) -> index {
  // expected-error @+1 {{failed to legalize operation 'memref.extract_strided_metadata'}}
  %base, %off, %sz, %st = memref.extract_strided_metadata %arg0
    : memref<?xi4> -> memref<i4>, index, index, index
  return %sz : index
}

// -----

// COM: ---- 3: !sourceType.getLayout().isIdentity() (second OR clause of the
// COM: static-shape / identity-layout reject). A strided / offset-bearing
// COM: layout map disqualifies the source.

#layout = affine_map<(d0)[s0] -> (d0 + s0)>
func.func @non_identity_layout(%arg0: memref<8xi4, #layout>) -> index {
  // expected-error @+1 {{failed to legalize operation 'memref.extract_strided_metadata'}}
  %base, %off, %sz, %st = memref.extract_strided_metadata %arg0
    : memref<8xi4, #layout> -> memref<i4>, index, index, index
  return %sz : index
}

// -----

// COM: ---- 4: !isa<BlockArgument>(castOp.getInputs()[0]).
// COM: Here the unrealized_conversion_cast feeds an SSA value defined by
// COM: memref.alloc rather than a function block argument, so the pattern
// COM: rejects even though the metadata extract itself looks well-formed.

func.func @non_blockarg_cast_input() -> index {
  %a = memref.alloc() : memref<4xi8>
  %cast = builtin.unrealized_conversion_cast %a : memref<4xi8> to memref<8xi4>
  // expected-error @+1 {{failed to legalize operation 'memref.extract_strided_metadata'}}
  %base, %off, %sz, %st = memref.extract_strided_metadata %cast
    : memref<8xi4> -> memref<i4>, index, index, index
  return %sz : index
}
