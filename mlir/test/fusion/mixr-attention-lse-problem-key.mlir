
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f16 -transQ false -transK true -transV false -transO false -causal false -return_lse true -g 8 -seq_len_q 32 -seq_len_k 32 -head_dim_qk 32 -head_dim_v 32

module {
  func.func private @mlir_attention(%v: !migraphx.shaped<2x2x32x32xf16, 2048x1024x32x1> {mhal.read_access}, 
                            %q: !migraphx.shaped<2x4x32x32xf16, 4096x1024x32x1> {mhal.read_access}, 
                            %k: !migraphx.shaped<2x2x32x32xf16, 2048x1024x32x1> {mhal.read_access}) 
                            -> (!migraphx.shaped<2x4x32x32xf16, 4096x1024x32x1> {mhal.write_access}, !migraphx.shaped<2x4x32x1xf16, 128x32x1x1> {mhal.write_access}) attributes {kernel, arch = "gfx942", num_cu = 304 : i64} {
    %vbroadcast = migraphx.multibroadcast %v {out_dyn_dims = [], out_lens = [2, 2, 2, 32, 32]} : <2x2x32x32xf16, 2048x1024x32x1> -> <2x2x2x32x32xf16, 2048x1024x0x32x1>
    %vreshaped = migraphx.reshape %vbroadcast {dims = [2, 4, 32, 32]} : <2x2x2x32x32xf16, 2048x1024x0x32x1> -> <2x4x32x32xf16, 2048x1024x32x1>
    %kbroadcast = migraphx.multibroadcast %k {out_dyn_dims = [], out_lens = [2, 2, 2, 32, 32]} : <2x2x32x32xf16, 2048x1024x32x1> -> <2x2x2x32x32xf16, 2048x1024x0x32x1>
    %kreshaped = migraphx.reshape %kbroadcast {dims = [2, 4, 32, 32]} : <2x2x2x32x32xf16, 2048x1024x0x32x1> -> <2x4x32x32xf16, 2048x1024x32x1>
    %kt = migraphx.transpose %kreshaped {permutation = [0, 1, 3, 2]} : <2x4x32x32xf16, 2048x1024x32x1> -> <2x4x32x32xf16, 2048x1024x32x1>
    %qk = migraphx.dot %q, %kt : <2x4x32x32xf16, 4096x1024x32x1>, <2x4x32x32xf16, 2048x1024x32x1> -> <2x4x32x32xf16, 4096x1024x32x1>
    %max = migraphx.reduce_max %qk {axes = [3 : i64]} : <2x4x32x32xf16, 4096x1024x32x1> -> <2x4x32x1xf16, 128x32x1x1>
    %norm = migraphx.sub %qk, %max : <2x4x32x32xf16, 4096x1024x32x1>, <2x4x32x1xf16, 128x32x1x1> -> <2x4x32x32xf16, 4096x1024x32x1>
    %exp = migraphx.exp %norm : <2x4x32x32xf16, 4096x1024x32x1> -> <2x4x32x32xf16, 4096x1024x32x1>
    %se = migraphx.reduce_sum %exp {axes = [3 : i64]} : <2x4x32x32xf16, 4096x1024x32x1> -> <2x4x32x1xf16, 128x32x1x1>
    %recip = migraphx.recip %se : <2x4x32x1xf16, 128x32x1x1> -> <2x4x32x1xf16, 128x32x1x1>
    %att = migraphx.mul %exp, %recip : <2x4x32x32xf16, 4096x1024x32x1>, <2x4x32x1xf16, 128x32x1x1> -> <2x4x32x32xf16, 4096x1024x32x1>
    
    %lse = migraphx.log %se : <2x4x32x1xf16, 128x32x1x1> -> <2x4x32x1xf16, 128x32x1x1>
    %lse_add = migraphx.add %lse, %max : <2x4x32x1xf16, 128x32x1x1>, <2x4x32x1xf16, 128x32x1x1> -> <2x4x32x1xf16, 128x32x1x1>
    %res = migraphx.dot %att, %vreshaped : <2x4x32x32xf16, 4096x1024x32x1>, <2x4x32x32xf16, 2048x1024x32x1> -> <2x4x32x32xf16, 4096x1024x32x1>
    return %res, %lse_add : !migraphx.shaped<2x4x32x32xf16, 4096x1024x32x1>, !migraphx.shaped<2x4x32x1xf16, 128x32x1x1>
  }
}
