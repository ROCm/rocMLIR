// Example lowered (unpacked) MIGraphX/MIXR-style MLIR kernel for the given graph.
// Assumes earlier FP4 realization: fp4x2 packed inputs already unpacked to fp4.
//
// func.func @quant_dot_fp4(
//   %x1 : !migraphx.shaped<1x2048xf4E2M1FN, 2048x1>,          // from unpack(x1)
//   %x2T: !migraphx.shaped<2048x1000xf4E2M1FN, 1x2048>,       // transpose(unpack(x2))
//   %scaleA : !migraphx.shaped<1x2048xf32, 2048x1>,           // reshape(mb(x3))
//   %scaleB : !migraphx.shaped<2048x1000xf32, 1000x1>,        // reshape(mb(x4))
//   %bias : !migraphx.shaped<1x1000xf32, 1000x1> )
//   -> !migraphx.shaped<1x1000xf32, 1000x1>
//
// Pseudocode semantics:
//   out[0,j] = bias[0,j] + sum_k ( (x1[0,k]*scaleA[0,k]) * (x2T[k,j]*scaleB[k,j]) )
//
module {
  func.func @quant_dot_fp4(%x1: !migraphx.shaped<1x2048xf4E2M1FN, 2048x1>,
                           %x2: !migraphx.shaped<1000x2048xf4E2M1FN, 2048x1>,
                           %x3: !migraphx.shaped<1x64x1xf32, 64x1x1>,
                           %x4: !migraphx.shaped<64x1x1000xf32, 1000x1000x1>,
                           %x5: !migraphx.shaped<1x1000xf32, 1000x1>)
        -> !migraphx.shaped<1x1000xf32, 1000x1> attributes {kernel, arch="gfx950"} {
    // Transpose weights (already unpacked fp4)
    %wT = migraphx.transpose %x2 {permutation = [1,0]}
          : <1000x2048xf4E2M1FN, 2048x1> -> <2048x1000xf4E2M1FN, 1x2048>

    // Broadcast/reshape scale A: (1,64,32) -> (1,2048)
    %mbA = migraphx.multibroadcast %x3 {out_lens = [1,64,32]}
           : <1x64x1xf32, 64x1x1> -> <1x64x32xf32, 64x1x0>
    %sA = migraphx.reshape %mbA {dims = [1,2048]}
          : <1x64x32xf32, 64x1x0> -> <1x2048xf32, 2048x1>

    // Broadcast/reshape scale B: (64,32,1000) -> (2048,1000)
    %mbB = migraphx.multibroadcast %x4 {out_lens = [64,32,1000]}
           : <64x1x1000xf32, 1000x1000x1> -> <64x32x1000xf32, 1000x0x1>
    %sB = migraphx.reshape %mbB {dims = [2048,1000]}
          : <64x32x1000xf32, 1000x0x1> -> <2048x1000xf32, 1000x1>

    %sE8A = migraphx.convert %sA : !migraphx.shaped<1x2048xf32, 2048x1> to !migraphx.shaped<1x2048xf8E8M0FNU, 2048x1>
    %sE8B = migraphx.convert %sB : !migraphx.shaped<2048x1000xf32, 1000x1> to !migraphx.shaped<2048x1000xf8E8M0FNU, 1000x1>

    // Quant dot (assumed dequant inside)
    %qd = migraphx.quant_dot %x1 scaled by %sE8A, %wT scaled by %sE8B
          : !migraphx.shaped<1x2048xf4E2M1FN, 2048x1> scaled by !migraphx.shaped<1x2048xf8E8M0FNU, 2048x1>,
            !migraphx.shaped<2048x1000xf4E2M1FN, 1x2048> scaled by !migraphx.shaped<2048x1000xf8E8M0FNU, 1000x1>
            -> !migraphx.shaped<1x1000xf32, 1000x1>

    %out = migraphx.add %qd, %x5
          : !migraphx.shaped<1x1000xf32, 1000x1>,
            !migraphx.shaped<1x1000xf32, 1000x1>
            -> !migraphx.shaped<1x1000xf32, 1000x1>
    return %out : !migraphx.shaped<1x1000xf32, 1000x1>
  }
}