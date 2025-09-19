// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// PassThrough: valid (same number of inputs and outputs, no params)
//===----------------------------------------------------------------------===//
func.func @pass_through_valid() {
  // CHECK: #rock.transform<PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1]>
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// PassThrough: invalid (different number of inputs and outputs)
//===----------------------------------------------------------------------===//
func.func @pass_through_dim_mismatch() {
  // expected-error @+1 {{PassThrough must have the same number of inputs and outputs}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x"] at [0] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// PassThrough: invalid (has params)
//===----------------------------------------------------------------------===//
func.func @pass_through_params() {
  // expected-error @+1 {{PassThrough has no parameters}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough{1} ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Embed: valid (one output, params per input)
//===----------------------------------------------------------------------===//
func.func @embed_valid() {
  // CHECK: #rock.transform<Embed{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>
  %0 = "test.use_attr"() {attr = #rock.transform<Embed{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Embed: invalid (wrong number of params)
//===----------------------------------------------------------------------===//
func.func @embed_wrong_params() {
  // expected-error @+1 {{Embed and unmerge must specify one coefficient per input dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Embed{2} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Embed: invalid (more than one output)
//===----------------------------------------------------------------------===//
func.func @embed_too_many_outputs() {
  // expected-error @+1 {{Embed and unmerge can only have one output argument}}
  %0 = "test.use_attr"() {attr = #rock.transform<Embed{2,3} ["x","y"] at [0,1] -> ["z","w"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Unmerge: valid (one output, params per input)
//===----------------------------------------------------------------------===//
func.func @unmerge_valid() {
  // CHECK: #rock.transform<Unmerge{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>
  %0 = "test.use_attr"() {attr = #rock.transform<Unmerge{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Unmerge: invalid (wrong number of params)
//===----------------------------------------------------------------------===//
func.func @unmerge_wrong_params() {
  // expected-error @+1 {{Embed and unmerge must specify one coefficient per input dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Unmerge{2} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Unmerge: invalid (more than one output)
//===----------------------------------------------------------------------===//
func.func @unmerge_too_many_outputs() {
  // expected-error @+1 {{Embed and unmerge can only have one output argument}}
  %0 = "test.use_attr"() {attr = #rock.transform<Unmerge{2,3} ["x","y"] at [0,1] -> ["z","w"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Merge: valid (one input, params per output)
//===----------------------------------------------------------------------===//
func.func @merge_valid() {
  // CHECK: #rock.transform<Merge{2,3} ["x"] at [0] -> ["y","z"] at [0,1]>
  %0 = "test.use_attr"() {attr = #rock.transform<Merge{2,3} ["x"] at [0] -> ["y","z"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Merge: invalid (more than one input)
//===----------------------------------------------------------------------===//
func.func @merge_too_many_inputs() {
  // expected-error @+1 {{Merge and unfold can only have one input dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Merge{2,3} ["x","y"] at [0,1] -> ["z","w"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Merge: invalid (wrong number of params)
//===----------------------------------------------------------------------===//
func.func @merge_wrong_params() {
  // expected-error @+1 {{Merge and unfold have one parameter per output dimension (its size)}}
  %0 = "test.use_attr"() {attr = #rock.transform<Merge{2} ["x"] at [0] -> ["y","z"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// AddDim: valid (one input, one param, no outputs)
//===----------------------------------------------------------------------===//
func.func @add_dim_valid() {
  // CHECK: #rock.transform<AddDim{4} ["x"] at [0] -> [] at []>
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4} ["x"] at [0] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// AddDim: invalid (more than one input)
//===----------------------------------------------------------------------===//
func.func @add_dim_too_many_inputs() {
  // expected-error @+1 {{Can only add one dimension at a time}}
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4} ["x","y"] at [0,1] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// AddDim: invalid (wrong number of params)
//===----------------------------------------------------------------------===//
func.func @add_dim_wrong_params() {
  // expected-error @+1 {{Must supply a size parameter for each dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4,5} ["x"] at [0] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// AddDim: invalid (outputs not empty)
//===----------------------------------------------------------------------===//
func.func @add_dim_outputs_not_empty() {
  // expected-error @+1 {{The added dimension cannot be mapped anywhere}}
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4} ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Broadcast: valid (params per output, matching ranks)
//===----------------------------------------------------------------------===//
func.func @broadcast_valid() {
  // CHECK: #rock.transform<Broadcast{2,3} ["x","y"] at [0,1] -> ["a","b"] at [0,1]>
  %0 = "test.use_attr"() {attr = #rock.transform<Broadcast{2,3} ["x","y"] at [0,1] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Broadcast: invalid (rank mismatch)
//===----------------------------------------------------------------------===//
func.func @broadcast_rank_mismatch() {
  // expected-error @+1 {{Broadcast must have same rank}}
  %0 = "test.use_attr"() {attr = #rock.transform<Broadcast{2,3} ["x"] at [0] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Broadcast: invalid (wrong number of params)
//===----------------------------------------------------------------------===//
func.func @broadcast_wrong_params() {
  // expected-error @+1 {{Broadcast must specify the output length for each dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Broadcast{2} ["x","y"] at [0,1] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// ConstDim: valid (no inputs, params are [value, length] pairs)
//===----------------------------------------------------------------------===//
func.func @constdim_valid() {
  // CHECK: #rock.transform<ConstDim{1,4,2,5} [] at [] -> ["a","b"] at [0,1]>
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{1,4,2,5} [] at [] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// ConstDim: invalid (has inputs)
//===----------------------------------------------------------------------===//
func.func @constdim_has_inputs() {
  // expected-error @+1 {{ConstDim must not take any inputs}}
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{1,4,2,5} ["x"] at [0] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// ConstDim: invalid (wrong number of params)
//===----------------------------------------------------------------------===//
func.func @constdim_wrong_params() {
  // expected-error @+1 {{ConstDim is parameterized by [value, length] pairs}}
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{1,4,2} [] at [] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// ConstDim: invalid (value >= length)
//===----------------------------------------------------------------------===//
func.func @constdim_value_ge_length() {
  // expected-error @+1 {{For constant dimension 0 constant value 4 must be less than dimension length 4}}
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{4,4} [] at [] -> ["a"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Slice: valid (no extra checks, just parse/verify)
//===----------------------------------------------------------------------===//
func.func @slice_valid() {
  // CHECK: #rock.transform<Slice ["x"] at [0] -> ["y"] at [0]>
  %0 = "test.use_attr"() {attr = #rock.transform<Slice ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Pad: valid (no extra checks, just parse/verify)
//===----------------------------------------------------------------------===//
func.func @pad_valid() {
  // CHECK: #rock.transform<Pad ["x"] at [0] -> ["y"] at [0]>
  %0 = "test.use_attr"() {attr = #rock.transform<Pad ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: upperNames/upperDims mismatch
//===----------------------------------------------------------------------===//
func.func @upper_names_dims_mismatch() {
  // expected-error @+1 {{Have 1 names for 2 dimensions}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x"] at [0,1] -> ["y","z"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: lowerNames/lowerDims mismatch
//===----------------------------------------------------------------------===//
func.func @lower_names_dims_mismatch() {
  // expected-error @+1 {{Have 2 names for 1 dimensions}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x","y"] at [0,1] -> ["z","w"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: non-AddDim with empty lowerDims
//===----------------------------------------------------------------------===//
func.func @no_outputs() {
  // expected-error @+1 {{The transformation must define outputs}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x"] at [0] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: non-ConstDim with empty upperDims
//===----------------------------------------------------------------------===//
func.func @no_inputs() {
  // expected-error @+1 {{The transformation must have at least one input}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough [] at [] -> ["y"] at [0]>} : () -