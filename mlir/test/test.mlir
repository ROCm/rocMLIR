module {
  func.func @dot(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, 
      %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, 
      %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1>{
    // 
    %0 = migraphx.dot %arg0, %arg1 
        : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    return %0 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }

  func.func @dot_two(%arg0: !migraphx.shaped<1x1x5x4xf32, 20x20x4x1>, 
      %arg1: !migraphx.shaped<1x1x4x3xf32, 12x12x3x1>, 
      %arg2: !migraphx.shaped<1x1x5x3xf32, 15x15x3x1>) -> !migraphx.shaped<1x1x5x3xf32, 15x15x3x1>{ 
    %0 = migraphx.dot %arg0, %arg1 
        : <1x1x5x4xf32, 20x20x4x1>, <1x1x4x3xf32, 12x12x3x1> -> <1x1x5x3xf32, 15x15x3x1>
    return %0 : !migraphx.shaped<1x1x5x3xf32, 15x15x3x1>
  }

  // func.func @linalg_func_one(%arg0: tensor<1x5x4xf32>,%arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xf32>) 
  //   -> tensor<1x5x3xf32>{
  //   linalg.batch_matmul
  //       ins(%arg0, %arg1: tensor<1x5x4xf32>, tensor<1x4x3xf32>)
  //       outs(%arg2 : tensor<1x5x3xf32>) -> tensor<1x5x3xf32>

  //   func.return %arg2 : tensor<1x5x3xf32>
  // }

  // func.func @linalg_func_two(%arg0: tensor<1x1x5x4xf32>, %arg1: tensor<1x1x4x3xf32>) -> tensor<1x1x5x3xf32>{
  //   %arg2 = arith.constant dense<0.0>: tensor<1x1x5x3xf32>
  //   linalg.generic {
  //       indexing_maps = [
  //           affine_map<(batch, d0, d1, d2, d3) -> (batch, d0, d1, d3)>,
  //           affine_map<(batch, d0, d1, d2, d3) -> (batch, d0, d3, d2)>,
  //           affine_map<(batch, d0, d1, d2, d3) -> (batch, d0, d1, d2)>
  //       ],
  //       iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
  //   } 
  //       ins(%arg0, %arg1: tensor<1x1x5x4xf32>, tensor<1x1x4x3xf32>)
  //       outs(%arg2: tensor<1x1x5x3xf32>) 
  //   {
  //       ^bb0(%a: f32, %b: f32, %c: f32):
  //           %d = arith.mulf %a, %b : f32
  //           %e = arith.addf %d, %c : f32
  //           linalg.yield %e: f32
  //   } -> tensor<1x1x5x3xf32>

  //   func.return %arg2 : tensor<1x1x5x3xf32>
  // }
}

