// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s

module {
  func.func private @gemm_gemm_reduce_sum_axis1(%a: !migraphx.shaped<1x128x64xf32, 8192x64x1> {mhal.read_access}, 
                                                %b: !migraphx.shaped<1x64x256xf32, 16384x256x1> {mhal.read_access},
                                                %c: !migraphx.shaped<1x256x128xf32, 32768x128x1> {mhal.read_access}) 
                                                -> (!migraphx.shaped<1x1x128xf32, 128x128x1> {mhal.write_access}) 
                                                attributes {kernel, arch = "gfx942", num_cu = 120 : i64} {
    %gemm0 = migraphx.dot %a, %b : <1x128x64xf32, 8192x64x1>, <1x64x256xf32, 16384x256x1> -> <1x128x256xf32, 32768x256x1>
    %gemm1 = migraphx.dot %gemm0, %c : <1x128x256xf32, 32768x256x1>, <1x256x128xf32, 32768x128x1> -> <1x128x128xf32, 16384x128x1>
    %result = migraphx.reduce_sum %gemm1 {axes = [1 : i64]} : <1x128x128xf32, 16384x128x1> -> <1x1x128xf32, 128x128x1>
    return %result : !migraphx.shaped<1x1x128xf32, 128x128x1>
  }
}

