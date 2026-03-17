// RUN: rocmlir-opt --migraphx-to-tosa -verify-diagnostics %s

// Scaled quant_dot with K=100, which is not a multiple of blockSize (32).
func.func @quant_dot_k_not_multiple_of_block_size(
    %arg0: !migraphx.shaped<1x64x100xf4E2M1FN, 6400x100x1>,
    %arg1: !migraphx.shaped<1x100x64xf4E2M1FN, 6400x64x1>,
    %arg2: !migraphx.shaped<1x64x100xf8E8M0FNU, 6400x100x1>,
    %arg3: !migraphx.shaped<1x100x64xf8E8M0FNU, 6400x64x1>
) -> !migraphx.shaped<1x64x64xf32, 4096x64x1> attributes {kernel} {
  // expected-error @+2 {{K dimension (100) must be a multiple of blockSize (32)}}
  // expected-error @+1 {{failed to legalize operation 'migraphx.quant_dot' that was explicitly marked illegal}}
  %0 = migraphx.quant_dot %arg0 scaled by %arg2, %arg1 scaled by %arg3
    : <1x64x100xf4E2M1FN, 6400x100x1> scaled by !migraphx.shaped<1x64x100xf8E8M0FNU, 6400x100x1>,
      <1x100x64xf4E2M1FN, 6400x64x1> scaled by !migraphx.shaped<1x100x64xf8E8M0FNU, 6400x64x1>
    -> <1x64x64xf32, 4096x64x1>
  return %0 : !migraphx.shaped<1x64x64xf32, 4096x64x1>
}
