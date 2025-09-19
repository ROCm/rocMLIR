// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: PassThrough, no params, matching upper/lower names/dims
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<PassThrough [] at [] -> [] at []>
func.func @pass_through() {
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough [] at [] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: PassThrough, with names and dims
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1]>
func.func @pass_through_names_dims() {
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: PassThrough, params not allowed
//===----------------------------------------------------------------------===//
func.func @pass_through_params_invalid() {
  // expected-error @+1 {{PassThrough has no parameters}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough{1} ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: PassThrough, upper/lower dims mismatch
//===----------------------------------------------------------------------===//
func.func @pass_through_dim_mismatch() {
  // expected-error @+1 {{PassThrough must have the same number of inputs and outputs}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x"] at [0] -> ["y","z"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: Embed, params per input, one output
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<Embed{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>
func.func @embed_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<Embed{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Embed, wrong number of params
//===----------------------------------------------------------------------===//
func.func @embed_wrong_params() {
  // expected-error @+1 {{Embed and unmerge must specify one coefficient per input dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Embed{2} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Embed, more than one output
//===----------------------------------------------------------------------===//
func.func @embed_too_many_outputs() {
  // expected-error @+1 {{Embed and unmerge can only have one output argument}}
  %0 = "test.use_attr"() {attr = #rock.transform<Embed{2,3} ["x","y"] at [0,1] -> ["z","w"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: Merge, one input, params per output
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<Merge{2,3} ["x"] at [0] -> ["y","z"] at [0,1]>
func.func @merge_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<Merge{2,3} ["x"] at [0] -> ["y","z"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Merge, more than one input
//===----------------------------------------------------------------------===//
func.func @merge_too_many_inputs() {
  // expected-error @+1 {{Merge and unfold can only have one input dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Merge{2,3} ["x","y"] at [0,1] -> ["z","w"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Merge, wrong number of params
//===----------------------------------------------------------------------===//
func.func @merge_wrong_params() {
  // expected-error @+1 {{Merge and unfold have one parameter per output dimension (its size)}}
  %0 = "test.use_attr"() {attr = #rock.transform<Merge{2} ["x"] at [0] -> ["y","z"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: AddDim, one input, one param, no outputs
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<AddDim{4} ["x"] at [0] -> [] at []>
func.func @add_dim_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4} ["x"] at [0] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: AddDim, more than one input
//===----------------------------------------------------------------------===//
func.func @add_dim_too_many_inputs() {
  // expected-error @+1 {{Can only add one dimension at a time}}
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4,5} ["x","y"] at [0,1] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: AddDim, wrong number of params
//===----------------------------------------------------------------------===//
func.func @add_dim_wrong_params() {
  // expected-error @+1 {{Must supply a size parameter for each dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4,5} ["x"] at [0] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: AddDim, outputs not empty
//===----------------------------------------------------------------------===//
func.func @add_dim_outputs_not_empty() {
  // expected-error @+1 {{The added dimension cannot be mapped anywhere}}
  %0 = "test.use_attr"() {attr = #rock.transform<AddDim{4} ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: Broadcast, params per output, matching ranks
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<Broadcast{2,3} ["x","y"] at [0,1] -> ["a","b"] at [0,1]>
func.func @broadcast_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<Broadcast{2,3} ["x","y"] at [0,1] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Broadcast, rank mismatch
//===----------------------------------------------------------------------===//
func.func @broadcast_rank_mismatch() {
  // expected-error @+1 {{Broadcast must have same rank}}
  %0 = "test.use_attr"() {attr = #rock.transform<Broadcast{2,3} ["x"] at [0] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Broadcast, wrong number of params
//===----------------------------------------------------------------------===//
func.func @broadcast_wrong_params() {
  // expected-error @+1 {{Broadcast must specify the output length for each dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Broadcast{2} ["x","y"] at [0,1] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: ConstDim, no inputs, params are [value, length] pairs
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<ConstDim{1,4,2,5} [] at [] -> ["a","b"] at [0,1]>
func.func @constdim_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{1,4,2,5} [] at [] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: ConstDim, has inputs
//===----------------------------------------------------------------------===//
func.func @constdim_has_inputs() {
  // expected-error @+1 {{ConstDim must not take any inputs}}
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{1,4,2,5} ["x"] at [0] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: ConstDim, wrong number of params
//===----------------------------------------------------------------------===//
func.func @constdim_wrong_params() {
  // expected-error @+1 {{ConstDim is parameterized by [value, length] pairs}}
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{1,4,2} [] at [] -> ["a","b"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: ConstDim, value >= length
//===----------------------------------------------------------------------===//
func.func @constdim_value_ge_length() {
  // expected-error @+1 {{For constant dimension 0 constant value 4 must be less than dimension length 4}}
  %0 = "test.use_attr"() {attr = #rock.transform<ConstDim{4,4} [] at [] -> ["a"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: Unmerge, params per input, one output
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<Unmerge{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>
func.func @unmerge_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<Unmerge{2,3} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Unmerge, wrong number of params
//===----------------------------------------------------------------------===//
func.func @unmerge_wrong_params() {
  // expected-error @+1 {{Embed and unmerge must specify one coefficient per input dimension}}
  %0 = "test.use_attr"() {attr = #rock.transform<Unmerge{2} ["x","y"] at [0,1] -> ["z"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Unmerge, more than one output
//===----------------------------------------------------------------------===//
func.func @unmerge_too_many_outputs() {
  // expected-error @+1 {{Embed and unmerge can only have one output argument}}
  %0 = "test.use_attr"() {attr = #rock.transform<Unmerge{2,3} ["x","y"] at [0,1] -> ["z","w"] at [0,1]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: Slice (no extra checks, just parse)
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<Slice ["x"] at [0] -> ["y"] at [0]>
func.func @slice_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<Slice ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: Pad (no extra checks, just parse)
//===----------------------------------------------------------------------===//
// CHECK: #rock.transform<Pad ["x"] at [0] -> ["y"] at [0]>
func.func @pad_valid() {
  %0 = "test.use_attr"() {attr = #rock.transform<Pad ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Unknown transform type
//===----------------------------------------------------------------------===//
func.func @unknown_transform_type() {
  // expected-error @+1 {{expected a name of a known transform}}
  %0 = "test.use_attr"() {attr = #rock.transform<Unknown ["x"] at [0] -> ["y"] at [0]>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: Syntax error (missing >)
//===----------------------------------------------------------------------===//
func.func @missing_gt() {
  // expected-error @+1 {{expected '>'}}
  %0 = "test.use_attr"() {attr = #rock.transform<PassThrough ["x"] at [0] -> ["y"] at [0]} : () -> ()
}