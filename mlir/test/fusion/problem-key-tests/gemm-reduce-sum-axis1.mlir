// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s

module {
  func.func private @gemm_reduce_sum_axis1(%a: !migraphx.shaped<1x128x64xf32, 8192x64x1> {mhal.read_access}, 
                                           %b: !migraphx.shaped<1x64x256xf32, 16384x256x1> {mhal.read_access}) 
                                           -> (!migraphx.shaped<1x1x256xf32, 256x256x1> {mhal.write_access}) 
                                           attributes {kernel, arch = "gfx942", num_cu = 120 : i64} {
    %gemm = migraphx.dot %a, %b : <1x128x64xf32, 8192x64x1>, <1x64x256xf32, 16384x256x1> -> <1x128x256xf32, 32768x256x1>
    %result = migraphx.reduce_sum %gemm {axes = [1 : i64]} : <1x128x256xf32, 32768x256x1> -> <1x1x256xf32, 256x256x1>
    return %result : !migraphx.shaped<1x1x256xf32, 256x256x1>
  }
}

