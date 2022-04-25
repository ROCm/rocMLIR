// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: @identity_token
func @identity_token(%arg0: !async.token) -> !async.token {
  // CHECK: return %arg0 : !async.token
  return %arg0 : !async.token
}

// CHECK-LABEL: @identity_value
func @identity_value(%arg0 : !async.value<f32>) -> !async.value<f32> {
  // CHECK: return %arg0 : !async.value<f32>
  return %arg0 : !async.value<f32>
}

// CHECK-LABEL: @empty_async_execute
func @empty_async_execute() -> !async.token {
  // CHECK: async.execute
  %token = async.execute {
    async.yield
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @return_async_value
func @return_async_value() -> !async.value<f32> {
  // CHECK: async.execute -> !async.value<f32>
  %token, %results = async.execute -> !async.value<f32> {
    %cst = arith.constant 1.000000e+00 : f32
    async.yield %cst : f32
  }

  // CHECK: return %results : !async.value<f32>
  return %results : !async.value<f32>
}

// CHECK-LABEL: @return_captured_value
func @return_captured_value() -> !async.token {
  %cst = arith.constant 1.000000e+00 : f32
  // CHECK: async.execute -> !async.value<f32>
  %token, %results = async.execute -> !async.value<f32> {
    async.yield %cst : f32
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @return_async_values
func @return_async_values() -> (!async.value<f32>, !async.value<f32>) {
  %token, %results:2 = async.execute -> (!async.value<f32>, !async.value<f32>) {
    %cst1 = arith.constant 1.000000e+00 : f32
    %cst2 = arith.constant 2.000000e+00 : f32
    async.yield %cst1, %cst2 : f32, f32
  }

  // CHECK: return %results#0, %results#1 : !async.value<f32>, !async.value<f32>
  return %results#0, %results#1 : !async.value<f32>, !async.value<f32>
}

// CHECK-LABEL: @async_token_dependencies
func @async_token_dependencies(%arg0: !async.token) -> !async.token {
  // CHECK: async.execute [%arg0]
  %token = async.execute [%arg0] {
    async.yield
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @async_value_operands
func @async_value_operands(%arg0: !async.value<f32>) -> !async.token {
  // CHECK: async.execute (%arg0 as %arg1: !async.value<f32>) -> !async.value<f32>
  %token, %results = async.execute (%arg0 as %arg1: !async.value<f32>) -> !async.value<f32> {
    async.yield %arg1 : f32
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @async_token_and_value_operands
func @async_token_and_value_operands(%arg0: !async.token, %arg1: !async.value<f32>) -> !async.token {
  // CHECK: async.execute [%arg0] (%arg1 as %arg2: !async.value<f32>) -> !async.value<f32>
  %token, %results = async.execute [%arg0] (%arg1 as %arg2: !async.value<f32>) -> !async.value<f32> {
    async.yield %arg2 : f32
  }

  // CHECK: return %token : !async.token
  return %token : !async.token
}

// CHECK-LABEL: @empty_tokens_or_values_operands
func @empty_tokens_or_values_operands() {
  // CHECK: async.execute {
  %token0 = async.execute [] () -> () { async.yield }
  // CHECK: async.execute {
  %token1 = async.execute () -> () { async.yield }
  // CHECK: async.execute {
  %token2 = async.execute -> () { async.yield }
  // CHECK: async.execute {
  %token3 = async.execute () { async.yield }
  // CHECK: async.execute {
  %token4 = async.execute [] { async.yield }
  return
}

// CHECK-LABEL: @await_token
func @await_token(%arg0: !async.token) {
  // CHECK: async.await %arg0
  async.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @await_value
func @await_value(%arg0: !async.value<f32>) -> f32 {
  // CHECK: async.await %arg0
  %0 = async.await %arg0 : !async.value<f32>
  return %0 : f32
}

// CHECK-LABEL: @create_group_and_await_all
func @create_group_and_await_all(%arg0: !async.token,
                                 %arg1: !async.value<f32>) -> index {
  %c = arith.constant 2 : index
  %0 = async.create_group %c : !async.group

  // CHECK: async.add_to_group %arg0
  // CHECK: async.add_to_group %arg1
  %1 = async.add_to_group %arg0, %0 : !async.token
  %2 = async.add_to_group %arg1, %0 : !async.value<f32>
  async.await_all %0

  %3 = arith.addi %1, %2 : index
  return %3 : index
}

func @kernel_func0() attributes {kernel} {
    return
}

// CHECK-LABEL: @empty_async_launch
func @empty_async_launch() -> !async.token {
  // CHECK: async.launch
  %token = async.launch @kernel_func0() : () -> ()

  // CHECK: return %token : !async.token
  return %token : !async.token
}

/// async.launch tests
func @kernel_func1(%arg0: tensor<8x8xf32>, %arg1: tensor<1x8xf32>) -> tensor<8x8xf32> attributes {kernel} {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
}

func @kernel_func2(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> attributes {kernel} {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: @single_async_launch
func @single_async_launch(%arg0: tensor<8x8xf32>, %arg1: tensor<1x8xf32>) -> tensor<8x8xf32> {
  // CHECK: async.launch
  %token,%results = async.launch @kernel_func1(%arg0, %arg1) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.await
  async.await %token : !async.token
  // CHECK: return %results : tensor<8x8xf32>
  return %results : tensor<8x8xf32>
}

// CHECK-LABEL: @chain_async_launch
func @chain_async_launch(%arg0: tensor<8x8xf32>, %arg1: tensor<1x8xf32>, %arg2: tensor<1x8xf32>) -> tensor<8x8xf32> {
  // CHECK: async.launch
  %token,%results = async.launch @kernel_func1(%arg0, %arg1) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.launch
  %token1,%results_1 = async.launch @kernel_func1 [%token] (%results, %arg2) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.await
  async.await %token1 : !async.token
  // CHECK: return %results_1 : tensor<8x8xf32>
  return %results_1 : tensor<8x8xf32>
}

// CHECK-LABEL: @independent_async_launch
func @independent_async_launch(%arg0: tensor<8x8xf32>, %arg1: tensor<1x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<1x8xf32>) -> tensor<8x8xf32> {
  // CHECK: async.launch
  %token,%results = async.launch @kernel_func1(%arg0, %arg1) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.launch
  %token1,%results_1 = async.launch @kernel_func1(%arg2, %arg3) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.await
  async.await %token : !async.token
  async.await %token1 : !async.token
  // CHECK: return %results_1 : tensor<8x8xf32>
  return %results_1 : tensor<8x8xf32>
}

// CHECK-LABEL: @v_async_launch
func @v_async_launch(%arg0: tensor<8x8xf32>, %arg1: tensor<1x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<1x8xf32>) -> tensor<8x8xf32> {
  // CHECK: async.launch
  %token,%results = async.launch @kernel_func1(%arg0, %arg1) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.launch
  %token1,%results_1 = async.launch @kernel_func1(%arg2, %arg3) : (tensor<8x8xf32>, tensor<1x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.launch
  %token2,%results_3 = async.launch @kernel_func2 [%token, %token1] (%results, %results_1) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK: async.await
  async.await %token2 : !async.token
  // CHECK: return %results_3 : tensor<8x8xf32>
  return %results_3 : tensor<8x8xf32>
}
