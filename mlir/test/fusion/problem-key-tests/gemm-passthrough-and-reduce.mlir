// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s

module {
  func.func private @gemm_passthrough_reduce(%a: !migraphx.shaped<1x128x64xf32, 8192x64x1> {mhal.read_access}, 
                                             %b: !migraphx.shaped<1x64x256xf32, 16384x256x1> {mhal.read_access}) 
                                             -> (!migraphx.shaped<1x128x256xf32, 32768x256x1> {mhal.write_access}, 
                                                 !migraphx.shaped<1x128x1xf32, 128x1x1> {mhal.write_access}) 
                                             attributes {kernel, arch = "gfx942", num_cu = 120 : i64} {
    %gemm = migraphx.dot %a, %b : <1x128x64xf32, 8192x64x1>, <1x64x256xf32, 16384x256x1> -> <1x128x256xf32, 32768x256x1>
    
    // Output 1: passthrough the gemm result
    // Output 2: reduce_sum
    %reduce = migraphx.reduce_sum %gemm {axes = [2 : i64]} : <1x128x256xf32, 32768x256x1> -> <1x128x1xf32, 128x1x1>
    
    return %gemm, %reduce : !migraphx.shaped<1x128x256xf32, 32768x256x1>, !migraphx.shaped<1x128x1xf32, 128x1x1>
  }
}

