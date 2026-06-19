// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for rock::TransformAttr::verify and
// COM: rock::TransformMapAttr::verify in
// COM: mlir/lib/Dialect/Rock/IR/RockDialect.cpp. Each section trips exactly one
// COM: verifier branch; the malformed #rock.transform_map attribute is rejected
// COM: at parse time via getChecked. Two diagnostics are emitted per case: the
// COM: verifier message and the generic "failed to parse ... parameter 'ops'"
// COM: wrapper from the attribute parser.

// COM: upperNames.size() != upperDims.size()
func.func @transform_upper_names_dims_mismatch(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Have 2 names for 1 dimensions}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a", "b"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: lowerNames.size() != lowerDims.size()
func.func @transform_lower_names_dims_mismatch(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Have 2 names for 1 dimensions}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a", "b"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: non-AddDim transform with no outputs
func.func @transform_no_outputs(%arg0: memref<64xf32>) {
  // expected-error @+2 {{The transformation must define outputs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> [] at []>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: non-ConstDim transform with no inputs
func.func @transform_no_inputs(%arg0: memref<64xf32>) {
  // expected-error @+2 {{The transformation must have at least one input}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough [] at [] -> ["a"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: PassThrough must have matching input/output rank
func.func @transform_passthrough_rank(%arg0: memref<64xf32>) {
  // expected-error @+2 {{PassThrough must have the same number of inputs and outputs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["a", "b"] at [0, 1] -> ["a"] at [0]>] bounds = [64, 64] -> [64]> : memref<64x64xf32> to memref<64xf32>
  return
}

// -----

// COM: PassThrough takes no parameters
func.func @transform_passthrough_params(%arg0: memref<64xf32>) {
  // expected-error @+2 {{PassThrough has no parameters}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough{1} ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: Embed can only have one output argument
func.func @transform_embed_one_output(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Embed and unmerge can only have one output argument}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Embed{1, 1} ["a", "b"] at [0, 1] -> ["x", "y"] at [0, 1]>] bounds = [64, 64] -> [64, 64]> : memref<64x64xf32> to memref<64x64xf32>
  return
}

// -----

// COM: Embed must specify one coefficient per input dimension
func.func @transform_embed_coeffs(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Embed and unmerge must specify one coefficient per input dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<Embed{1} ["a", "b"] at [0, 1] -> ["x"] at [0]>] bounds = [64, 3] -> [64]> : memref<64x3xf32> to memref<64xf32>
  return
}

// -----

// COM: Merge can only have one input dimension
func.func @transform_merge_one_input(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Merge and unfold can only have one input dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<Merge{16, 4} ["a", "b"] at [0, 1] -> ["x", "y"] at [0, 1]>] bounds = [16, 4] -> [16, 4]> : memref<16x4xf32> to memref<16x4xf32>
  return
}

// -----

// COM: Merge has one parameter per output dimension
func.func @transform_merge_params(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Merge and unfold have one parameter per output dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0, d0)> by [<Merge{16} ["a"] at [0] -> ["x", "y"] at [0, 1]>] bounds = [64] -> [16, 4]> : memref<64xf32> to memref<16x4xf32>
  return
}

// -----

// COM: AddDim can only add one dimension at a time
func.func @transform_adddim_one(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Can only add one dimension at a time}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> ()> by [<AddDim{16, 4} ["a", "b"] at [0, 1] -> [] at []>] bounds = [16, 4] -> []> : memref<16x4xf32> to memref<f32>
  return
}

// -----

// COM: AddDim must supply a size parameter for each dimension
func.func @transform_adddim_size(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Must supply a size parameter for each dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> ()> by [<AddDim ["a"] at [0] -> [] at []>] bounds = [16] -> []> : memref<16xf32> to memref<f32>
  return
}

// -----

// COM: AddDim output cannot be mapped anywhere
func.func @transform_adddim_mapped(%arg0: memref<64xf32>) {
  // expected-error @+2 {{The added dimension cannot be mapped anywhere}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<AddDim{16} ["a"] at [0] -> ["x"] at [0]>] bounds = [16] -> [16]> : memref<16xf32> to memref<16xf32>
  return
}

// -----

// COM: Broadcast must have same rank
func.func @transform_broadcast_rank(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Broadcast must have same rank}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<Broadcast{1} ["a", "b"] at [0, 1] -> ["x"] at [0]>] bounds = [64, 64] -> [64]> : memref<64x64xf32> to memref<64xf32>
  return
}

// -----

// COM: Broadcast must specify the output length for each dimension
func.func @transform_broadcast_lengths(%arg0: memref<64xf32>) {
  // expected-error @+2 {{Broadcast must specify the output length for each dimension}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<Broadcast{1, 2} ["a"] at [0] -> ["x"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: ConstDim must not take any inputs
func.func @transform_constdim_inputs(%arg0: memref<64xf32>) {
  // expected-error @+2 {{ConstDim must not take any inputs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<ConstDim{0, 8} ["a"] at [0] -> ["x"] at [0]>] bounds = [64] -> [8]> : memref<64xf32> to memref<8xf32>
  return
}

// -----

// COM: ConstDim is parameterized by [value, length] pairs
func.func @transform_constdim_pairs(%arg0: memref<64xf32>) {
  // expected-error @+2 {{ConstDim is parameterized by [value, length] pairs}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<() -> (0)> by [<ConstDim{1} [] at [] -> ["x"] at [0]>] bounds = [] -> [8]> : memref<f32> to memref<8xf32>
  return
}

// -----

// COM: ConstDim value must be less than dimension length
func.func @transform_constdim_value(%arg0: memref<64xf32>) {
  // expected-error @+2 {{constant value 8 must be less than dimension length 8}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<() -> (0)> by [<ConstDim{8, 8} [] at [] -> ["x"] at [0]>] bounds = [] -> [8]> : memref<f32> to memref<8xf32>
  return
}

// -----

// COM: TransformMapAttr: affine map input count must match upper bounds
func.func @transform_map_input_count(%arg0: memref<64xf32>) {
  // expected-error @+1 {{Affine map has 2 inputs but there are 1 input dimensions}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: TransformMapAttr: affine map output count must match lower bounds
func.func @transform_map_output_count(%arg0: memref<64xf32>) {
  // expected-error @+1 {{Affine map has 2 outputs but there are 1 outut dimensions}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0, d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: TransformMapAttr: negative upper bound rejected
func.func @transform_map_negative_upper(%arg0: memref<64xf32>) {
  // expected-error @+1 {{Upper bound/shape component less than 0}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [-1] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: TransformMapAttr: negative lower bound rejected
func.func @transform_map_negative_lower(%arg0: memref<64xf32>) {
  // expected-error @+1 {{Lower bound/shape component less than 0}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [-1]> : memref<64xf32> to memref<64xf32>
  return
}

// -----

// COM: TransformAttr::parse rejects an unknown transform name
func.func @transform_unknown_name(%arg0: memref<64xf32>) {
  // expected-error @+3 {{expected a name of a known transform}}
  // expected-note @+2 {{The transforms are PassThrough, Pad, Slice, Embed, Unmerge, Merge, Unfold}}
  // expected-error @+1 {{failed to parse Rock_TransformMapAttr parameter 'ops'}}
  %0 = rock.transform %arg0 by <affine_map<(d0) -> (d0)> by [<NotARealTransform ["a"] at [0] -> ["a"] at [0]>] bounds = [64] -> [64]> : memref<64xf32> to memref<64xf32>
  return
}
