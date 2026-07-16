// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_i32_wants_i8(%a: memref<1x16x16xf32>,
                        %b: memref<1x16x16xf32>,
                        %c: memref<1x16x16xi32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op floating-point input type 'f32' requires a floating-point output type, but the output type is 'i32'}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x16x16xi32> = memref<1x16x16xf32> * memref<1x16x16xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_i8_wants_i32(%a: memref<1x16x16xi8>,
                        %b: memref<1x16x16xi8>,
                        %c: memref<1x16x16xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op integer input type 'i8' requires an integer output type, but the output type is 'f32'}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x16x16xf32> = memref<1x16x16xi8> * memref<1x16x16xi8>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_m_too_big(%a: memref<1x1x2147483648xf32>,
                        %b: memref<1x1x1xf32>,
                        %c: memref<1x2147483648x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x2147483648x1xf32> = memref<1x1x2147483648xf32> * memref<1x1x1xf32>
  func.return
}

// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_k_too_big(%a: memref<1x2147483648x1xf32>,
                        %b: memref<1x2147483648x1xf32>,
                        %c: memref<1x1x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op K dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x1x1xf32> = memref<1x2147483648x1xf32> * memref<1x2147483648x1xf32>
  func.return
}
// -----

#general_gemm_params0 = #rock.general_gemm_params<blockSize = 256, mPerBlock = 16, kPerBlock = 16, nPerBlock = 16, mPerThread = 1, kPerThread = 16, nPerThread = 1, kpack = 1, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @gridwise_gemm_m_too_big(%a: memref<1x1x1xf32>,
                        %b: memref<1x1x2147483648xf32>,
                        %c: memref<1x1x2147483648xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op N dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  rock.gridwise_gemm %c = %a * %b storeMethod(set) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #general_gemm_params0}
  : memref<1x1x2147483648xf32> = memref<1x1x1xf32> * memref<1x1x2147483648xf32>
  func.return
}

// -----

func.func @expand_strides_rank_mismatch(%input: tensor<4x24xf16>, %output: tensor<4x48x24xf16>) -> tensor<4x48x24xf16> {
  // expected-error@+1 {{'rock.expand_strides' op input and output must have the same rank}}
  %result = rock.expand_strides %input into %output : tensor<4x24xf16> into tensor<4x48x24xf16> -> tensor<4x48x24xf16>
  return %result : tensor<4x48x24xf16>
}

// -----

func.func @expand_strides_output_too_small(%input: tensor<4x24x24xf16>, %output: tensor<4x20x24xf16>) -> tensor<4x20x24xf16> {
  // expected-error@+1 {{'rock.expand_strides' op output dimension 20 is smaller than input dimension 24}}
  %result = rock.expand_strides %input into %output : tensor<4x24x24xf16> into tensor<4x20x24xf16> -> tensor<4x20x24xf16>
  return %result : tensor<4x20x24xf16>
}

// -----

func.func @expand_strides_element_type_mismatch(%input: tensor<4x24x24xf16>, %output: tensor<4x48x24xf32>) -> tensor<4x48x24xf32> {
  // expected-error@+1 {{'rock.expand_strides' op input and output must have the same element type}}
  %result = rock.expand_strides %input into %output : tensor<4x24x24xf16> into tensor<4x48x24xf32> -> tensor<4x48x24xf32>
  return %result : tensor<4x48x24xf32>
}


// -----

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

// -----

// COM: Negative coverage for rock::TransformingForOp::parse and
// COM: rock::TransformingForOp::verify in
// COM: mlir/lib/Dialect/Rock/IR/RockDialect.cpp. Each section exercises a single
// COM: parser or verifier error branch of the rock.transforming_for op.

// COM: parse: no transforms but lower/upper arg counts differ
func.func @tfor_no_transform_arg_count() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Expected same number of lower and upper arguments when transforms absent}}
  rock.transforming_for (%a, %b) = [](%c0) (%v) = validity bounds [2] strides [1] {
    rock.yield
  }
  return
}

// -----

// COM: parse: the transform list element is not a transform_map attribute
func.func @tfor_not_transform_map() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Expected transform map attributes}}
  rock.transforming_for (%a) = [0 : i32](%c0) (%v) = validity bounds [4] strides [1] {
    rock.yield
  }
  return
}

// -----

#unmerge0 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + 4 * d0)>
    by [<Unmerge{16, 4} ["1", "0"] at [0, 1] -> ["r"] at [0]>]
    bounds = [16, 4] -> [64]>

// COM: parse: number of upper inits doesn't match the transform sequence inputs
func.func @tfor_wrong_num_inputs() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Transformation sequence expected 2 inputs}}
  rock.transforming_for (%l) = [#unmerge0](%c0) (%v) = validity bounds [16, 4] strides [1, 1] {
    rock.yield
  }
  return
}

// -----

#unmerge1 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + 4 * d0)>
    by [<Unmerge{16, 4} ["1", "0"] at [0, 1] -> ["r"] at [0]>]
    bounds = [16, 4] -> [64]>

// COM: parse: number of lower coords doesn't match the transform sequence outputs
func.func @tfor_wrong_num_outputs() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Transformation sequence expected 1 outputs}}
  rock.transforming_for (%l0, %l1) = [#unmerge1](%c0, %c0) (%v) = validity bounds [16, 4] strides [1, 1] {
    rock.yield
  }
  return
}

// -----

// COM: parse: number of validity arguments must equal the number of domains
func.func @tfor_wrong_num_validities() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Expected 1 validity arguments, one per domain, but found 2}}
  rock.transforming_for (%a) = [](%c0) (%v0, %v1) = validity bounds [4] strides [1] {
    rock.yield
  }
  return
}

// -----

// COM: parse: iter_args count must match the number of result types
func.func @tfor_iter_args_type_mismatch() {
  %c0 = arith.constant 0 : index
  %init = arith.constant 0.0 : f32
  // expected-error @+1 {{Mismatch between number of iter_args and types}}
  rock.transforming_for (%a) = [](%c0) (%v) = validity iter_args (%x = %init) -> (f32, f32) bounds [4] strides [1] {
    rock.yield %x : f32
  }
  return
}

// -----

// COM: verify: bounds and strides lists must have the same length
func.func @tfor_bounds_strides_length() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Bounds list and strides list must have same length}}
  rock.transforming_for (%a, %b) = [](%c0, %c0) (%v) = validity bounds [2, 3] strides [1] {
    rock.yield
  }
  return
}

// -----

// COM: verify: zero/negative strides are rejected
func.func @tfor_zero_stride() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Negative and zero strides are not permitted}}
  rock.transforming_for (%a, %b) = [](%c0, %c0) (%v) = validity bounds [2, 3] strides [0, 1] {
    rock.yield
  }
  return
}

// -----

// COM: verify: each bound must evenly divide its stride
func.func @tfor_bound_not_divisible() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{does not evenly divide the stride}}
  rock.transforming_for (%a, %b) = [](%c0, %c0) (%v) = validity bounds [3, 3] strides [2, 1] {
    rock.yield
  }
  return
}
