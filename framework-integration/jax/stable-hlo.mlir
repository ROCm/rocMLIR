module @jit_gemm_add attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>) -> (tensor<10x10xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<10x10xf32>
    return %1 : tensor<10x10xf32>
  }
}