// Simple MIGraphX Dialect Example
// This example demonstrates basic MIGraphX operations:
// - Matrix multiplication (migraphx.dot)
// - Element-wise addition (migraphx.add)
// - ReLU activation (migraphx.relu)
//
// To run this example:
// 1. From the build directory:
//    ./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir --kernel-pipeline=gpu --arch=gfx1150
//
// 2. Full compilation pipeline:
//    ./bin/rocmlir-driver ../examples/migraphx-dialect/migraphx_simple_example.mlir -c --arch=gfx1150

module {
  // Simple function: output = relu(dot(A, B) + bias)
  // This is a common pattern in neural networks
  func.func @simple_gemm_relu(
    %A: !migraphx.shaped<2x4xf32, 4x1>,      // Input matrix A: 2x4
    %B: !migraphx.shaped<4x3xf32, 3x1>,      // Weight matrix B: 4x3
    %bias: !migraphx.shaped<2x3xf32, 3x1>    // Bias: 2x3
  ) -> !migraphx.shaped<2x3xf32, 3x1> 
    attributes {kernel = "mixr", arch = "gfx1150", block_size = 64 : i32, grid_size = 1 : i32} {
    
    // Step 1: Matrix multiplication (GEMM)
    // Result shape: 2x3 (from 2x4 @ 4x3)
    %gemm = migraphx.dot %A, %B : 
      <2x4xf32, 4x1>, <4x3xf32, 3x1> -> <2x3xf32, 3x1>
    
    // Step 2: Add bias (element-wise)
    %add = migraphx.add %gemm, %bias : 
      <2x3xf32, 3x1>, <2x3xf32, 3x1> -> <2x3xf32, 3x1>
    
    // Step 3: Apply ReLU activation
    %output = migraphx.relu %add : 
      <2x3xf32, 3x1> -> <2x3xf32, 3x1>
    
    return %output : !migraphx.shaped<2x3xf32, 3x1>
  }
}
