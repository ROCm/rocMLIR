// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s

module {
  func.func private @gemm_multi_reduce(%a: !migraphx.shaped<1x128x64xf32, 8192x64x1> {mhal.read_access}, 
                                       %b: !migraphx.shaped<1x64x256xf32, 16384x256x1> {mhal.read_access}) 
                                       -> (!migraphx.shaped<1x128x1xf32, 128x1x1> {mhal.write_access}, 
                                           !migraphx.shaped<1x128x1xf32, 128x1x1> {mhal.write_access}) 
                                       attributes {kernel, arch = "gfx942", num_cu = 120 : i64} {
    %gemm = migraphx.dot %a, %b : <1x128x64xf32, 8192x64x1>, <1x64x256xf32, 16384x256x1> -> <1x128x256xf32, 32768x256x1>
    
    // First reduction: reduce_sum(x)
    %reduce1 = migraphx.reduce_sum %gemm {axes = [2 : i64]} : <1x128x256xf32, 32768x256x1> -> <1x128x1xf32, 128x1x1>
    
    // Second reduction: reduce_sum(x * x)
    %square = migraphx.mul %gemm, %gemm : <1x128x256xf32, 32768x256x1>, <1x128x256xf32, 32768x256x1> -> <1x128x256xf32, 32768x256x1>
    %reduce2 = migraphx.reduce_sum %square {axes = [2 : i64]} : <1x128x256xf32, 32768x256x1> -> <1x128x1xf32, 128x1x1>
    
    return %reduce1, %reduce2 : !migraphx.shaped<1x128x1xf32, 128x1x1>, !migraphx.shaped<1x128x1xf32, 128x1x1>
  }
}

