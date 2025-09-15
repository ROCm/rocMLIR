module {
  func.func private @mlir_attention_int4_convert(%arg0: !migraphx.shaped<4096x4096xf32, 4096x1>, %arg1: !migraphx.shaped<4096x4096xf32, 4096x1>, %arg2: !migraphx.shaped<4096x4096xf16, 4096x1>, %arg3: !migraphx.shaped<4096x2048xui8, 2048x1>, %arg4: !migraphx.shaped<4096x4096xf16, 4096x1>) -> !migraphx.shaped<4096x4096xf16, 4096x1> attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = migraphx.add %arg0, %arg1 : <4096x4096xf32, 4096x1>, <4096x4096xf32, 4096x1> -> <4096x4096xf32, 4096x1>
    %1 = migraphx.convert %0 {target_type = 0 : i64} : <4096x4096xf32, 4096x1> to <4096x4096xf16, 4096x1>
    %2 = migraphx.dot %arg2, %1 : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %3 = migraphx.softmax %2 {axis = 1 : i64} : <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %4 = migraphx.dot %3, %arg4 {perf_config = "attn:v2:64,128,128,128,16,16,1,1,1,2,1"} : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    return %4 : !migraphx.shaped<4096x4096xf16, 4096x1>
  }
}