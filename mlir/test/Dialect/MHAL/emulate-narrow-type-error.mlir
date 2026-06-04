// RUN: rocmlir-opt -mhal-emulate-narrow-type --verify-diagnostics \
// RUN:   --split-input-file %s -o /dev/null

// COM: Exercises the four pattern-rejection branches of
// COM: ExtractStridedMetadataFromOldFuncArgs in external/mlir-hal/lib/Dialect/
// COM: MHAL/Transforms/EmulateNarrowType.cpp lines 65-70. When this helper
// COM: pattern rejects a memref.extract_strided_metadata op, no other pattern
// COM: in the second applyPartialConversion can legalize it (because the type
// COM: converter has marked it illegal for the un-converted memref<...xi4>
// COM: type). That makes applyPartialConversion fail and the pass emit the
// COM: "failed to legalize operation 'memref.extract_strided_metadata' that
// COM: was explicitly marked illegal" diagnostic, which simultaneously
// COM: covers the signalPassFailure path at lines 154-155.

// COM: ---- 1: getBaseBuffer().use_empty() == false (line 65-66). The pattern
// COM: bails when the base_buffer result is consumed.

func.func @base_buffer_used(%arg0: memref<8xi4>) -> memref<i4> {
  // expected-error @+1 {{failed to legalize operation 'memref.extract_strided_metadata'}}
  %base, %off, %sz, %st = memref.extract_strided_metadata %arg0
    : memref<8xi4> -> memref<i4>, index, index, index
  return %base : memref<i4>
}

// -----

// COM: ---- 2: !sourceType.hasStaticShape() (line 67-68, first OR clause).

func.func @dynamic_shape(%arg0: memref<?xi4>) -> index {
  // expected-error @+1 {{failed to legalize operation 'memref.extract_strided_metadata'}}
  %base, %off, %sz, %st = memref.extract_strided_metadata %arg0
    : memref<?xi4> -> memref<i4>, index, index, index
  return %sz : index
}

// -----

// COM: ---- 3: !sourceType.getLayout().isIdentity() (line 67-68, second OR
// COM: clause). A strided / offset-bearing layout map disqualifies the
// COM: source.

#layout = affine_map<(d0)[s0] -> (d0 + s0)>
func.func @non_identity_layout(%arg0: memref<8xi4, #layout>) -> index {
  // expected-error @+1 {{failed to legalize operation 'memref.extract_strided_metadata'}}
  %base, %off, %sz, %st = memref.extract_strided_metadata %arg0
    : memref<8xi4, #layout> -> memref<i4>, index, index, index
  return %sz : index
}

// -----

// COM: ---- 4: !isa<BlockArgument>(castOp.getInputs()[0]) (line 69-70).
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
