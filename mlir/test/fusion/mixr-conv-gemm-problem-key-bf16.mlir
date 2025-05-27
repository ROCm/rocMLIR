
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t bf16 -f N01GC -I N01GC -transC true -transO false -n 2 -c 64 -H 32 -W 32 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -gemmO 16
module 
{
  func.func private @mlir_conv_gemm(%arg0: !migraphx.shaped<2x64x32x32xbf16, 65536x1x2048x64>,
                                    %arg1: !migraphx.shaped<64x64x3x3xbf16, 576x1x192x64>,
                                    %arg2: !migraphx.shaped<1x64x16xbf16, 1024x1x64>,
                                    %arg3: !migraphx.shaped<1x2048x64xbf16, 0x0x1>) 
                                    -> (!migraphx.shaped<1x2048x16xbf16, 32768x16x1>)  attributes {kernel, arch = "gfx942", num_cu = 304 : i64} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x64x32x32xbf16, 65536x1x2048x64>, <64x64x3x3xbf16, 576x1x192x64> -> <2x64x32x32xbf16, 65536x1x2048x64>
    %1 = migraphx.transpose %0 {permutation = [0, 2, 3, 1]} : <2x64x32x32xbf16, 65536x1x2048x64> -> <2x32x32x64xbf16, 65536x2048x64x1>
    %2 = migraphx.reshape %1 {dims = [1, 2048, 64]} : <2x32x32x64xbf16, 65536x2048x64x1> -> <1x2048x64xbf16, 131072x64x1>
    %biased = migraphx.add %2, %arg3 : <1x2048x64xbf16, 131072x64x1>, <1x2048x64xbf16, 0x0x1> -> <1x2048x64xbf16, 131072x64x1>
    %3 = migraphx.dot %biased, %arg2: <1x2048x64xbf16, 131072x64x1>, <1x64x16xbf16, 1024x1x64> -> <1x2048x16xbf16, 32768x16x1>
    return %3 : !migraphx.shaped<1x2048x16xbf16, 32768x16x1>
  }
}
