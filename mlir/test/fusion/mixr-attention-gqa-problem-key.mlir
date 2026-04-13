
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f16 -transQ false -transK true -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 32 -num_heads_kv 8 -g 1 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 128 -head_dim_v 128

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x48x1x128xf16, 6144x128x128x1>, %arg1: !migraphx.shaped<1x8x64x128xf16, 65536x8192x128x1>, %arg2: !migraphx.shaped<1x8x64x128xf16, 65536x8192x128x1>, %arg3: !migraphx.shaped<1x1x1xsi32, 1x1x1>) -> !migraphx.shaped<1x1x4096xf16, 4096x4096x1> attributes {rock.kernel, rock.arch = "gfx942", rock.num_cu = 304 : i64} {
    %0 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xsi32>) : <64xsi32, 1>
    %1 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<8.837890e-02> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 1, 1, 64]} : <64xsi32, 1> -> <1x1x1x64xsi32, 0x0x0x1>
    %4 = migraphx.slice %arg0 {axes = [1], ends = [32], starts = [0]} : <1x48x1x128xf16, 6144x128x128x1> -> <1x32x1x128xf16, 6144x128x128x1>
    %5 = migraphx.reshape %arg1 {dims = [1, 8, 1, 64, 128]} : <1x8x64x128xf16, 65536x8192x128x1> -> <1x8x1x64x128xf16, 65536x8192x8192x128x1>
    %6 = migraphx.reshape %arg2 {dims = [1, 8, 1, 64, 128]} : <1x8x64x128xf16, 65536x8192x128x1> -> <1x8x1x64x128xf16, 65536x8192x8192x128x1>
    %7 = migraphx.multibroadcast %6 {out_dyn_dims = [], out_lens = [1, 8, 4, 64, 128]} : <1x8x1x64x128xf16, 65536x8192x8192x128x1> -> <1x8x4x64x128xf16, 65536x8192x0x128x1>
    %8 = migraphx.reshape %7 {dims = [1, 32, 64, 128]} : <1x8x4x64x128xf16, 65536x8192x0x128x1> -> <1x32x64x128xf16, 262144x8192x128x1>
    %9 = migraphx.transpose %5 {permutation = [0, 1, 2, 4, 3]} : <1x8x1x64x128xf16, 65536x8192x8192x128x1> -> <1x8x1x128x64xf16, 65536x8192x8192x1x128>
    %10 = migraphx.multibroadcast %9 {out_dyn_dims = [], out_lens = [1, 8, 4, 128, 64]} : <1x8x1x128x64xf16, 65536x8192x8192x1x128> -> <1x8x4x128x64xf16, 65536x8192x0x1x128>
    %11 = migraphx.reshape %10 {dims = [1, 32, 128, 64]} : <1x8x4x128x64xf16, 65536x8192x0x1x128> -> <1x32x128x64xf16, 262144x8192x64x1>
    %12 = migraphx.dot %4, %11 : <1x32x1x128xf16, 6144x128x128x1>, <1x32x128x64xf16, 262144x8192x64x1> -> <1x32x1x64xf16, 2048x64x64x1>
    %13 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 32, 1, 64]} : <1xf16, 1> -> <1x32x1x64xf16, 0x0x0x0>
    %14 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 32, 1, 64]} : <1xf16, 1> -> <1x32x1x64xf16, 0x0x0x0>
    %15 = migraphx.mul %12, %14 : <1x32x1x64xf16, 2048x64x64x1>, <1x32x1x64xf16, 0x0x0x0> -> <1x32x1x64xf16, 2048x64x64x1>
    %16 = migraphx.broadcast %arg3 {axis = 0 : i64, out_lens = [1, 1, 1, 64]} : <1x1x1xsi32, 1x1x1> -> <1x1x1x64xsi32, 1x1x1x0>
    %17 = migraphx.greater %3, %16 : <1x1x1x64xsi32, 0x0x0x1>, <1x1x1x64xsi32, 1x1x1x0> -> <1x1x1x64xsi32, 0x0x0x1>
    %18 = migraphx.convert %17 {target_type = 0 : i64} : <1x1x1x64xsi32, 0x0x0x1> to <1x1x1x64xsi8, 0x0x0x1>
    %19 = migraphx.multibroadcast %18 {out_dyn_dims = [], out_lens = [1, 32, 1, 64]} : <1x1x1x64xsi8, 0x0x0x1> -> <1x32x1x64xsi8, 0x0x0x1>
    %20 = migraphx.where %19, %13, %15 : <1x32x1x64xsi8, 0x0x0x1>, <1x32x1x64xf16, 0x0x0x0>, <1x32x1x64xf16, 2048x64x64x1> -> <1x32x1x64xf16, 2048x64x64x1>
    %21 = migraphx.convert %20 {target_type = 2 : i64} : <1x32x1x64xf16, 2048x64x64x1> to <1x32x1x64xf32, 2048x64x64x1>
    %22 = migraphx.reshape %21 {dims = [1, 32, 1, 64]} : <1x32x1x64xf32, 2048x64x64x1> -> <1x32x1x64xf32, 2048x64x64x1>
    %23 = migraphx.reduce_max %22 {axes = [3]} : <1x32x1x64xf32, 2048x64x64x1> -> <1x32x1x1xf32, 32x1x1x1>
    %24 = migraphx.reshape %23 {dims = [1, 32, 1, 1]} : <1x32x1x1xf32, 32x1x1x1> -> <1x32x1x1xf32, 32x1x1x1>
    %25 = migraphx.multibroadcast %24 {out_dyn_dims = [], out_lens = [1, 32, 1, 64]} : <1x32x1x1xf32, 32x1x1x1> -> <1x32x1x64xf32, 32x1x1x0>
    %26 = migraphx.sub %21, %25 : <1x32x1x64xf32, 2048x64x64x1>, <1x32x1x64xf32, 32x1x1x0> -> <1x32x1x64xf32, 2048x64x64x1>
    %27 = migraphx.exp %26 : <1x32x1x64xf32, 2048x64x64x1> -> <1x32x1x64xf32, 2048x64x64x1>
    %28 = migraphx.reshape %27 {dims = [1, 32, 1, 64]} : <1x32x1x64xf32, 2048x64x64x1> -> <1x32x1x64xf32, 2048x64x64x1>
    %29 = migraphx.reduce_sum %28 {axes = [3]} : <1x32x1x64xf32, 2048x64x64x1> -> <1x32x1x1xf32, 32x1x1x1>
    %30 = migraphx.reshape %29 {dims = [1, 32, 1, 1]} : <1x32x1x1xf32, 32x1x1x1> -> <1x32x1x1xf32, 32x1x1x1>
    %31 = migraphx.multibroadcast %30 {out_dyn_dims = [], out_lens = [1, 32, 1, 64]} : <1x32x1x1xf32, 32x1x1x1> -> <1x32x1x64xf32, 32x1x1x0>
    %32 = migraphx.div %27, %31 : <1x32x1x64xf32, 2048x64x64x1>, <1x32x1x64xf32, 32x1x1x0> -> <1x32x1x64xf32, 2048x64x64x1>
    %33 = migraphx.convert %32 {target_type = 1 : i64} : <1x32x1x64xf32, 2048x64x64x1> to <1x32x1x64xf16, 2048x64x64x1>
    %34 = migraphx.dot %33, %8 : <1x32x1x64xf16, 2048x64x64x1>, <1x32x64x128xf16, 262144x8192x128x1> -> <1x32x1x128xf16, 4096x128x128x1>
    %35 = migraphx.transpose %34 {permutation = [0, 2, 1, 3]} : <1x32x1x128xf16, 4096x128x128x1> -> <1x1x32x128xf16, 4096x128x128x1>
    %36 = migraphx.reshape %35 {dims = [1, 1, 4096]} : <1x1x32x128xf16, 4096x128x128x1> -> <1x1x4096xf16, 4096x4096x1>
    return %36 : !migraphx.shaped<1x1x4096xf16, 4096x4096x1>
  }
}
