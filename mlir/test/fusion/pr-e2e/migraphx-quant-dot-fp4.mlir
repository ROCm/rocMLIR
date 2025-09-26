// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_quant_dot_fp4 %s | rocmlir-driver  -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut mlir_quant_dot_fp4_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]

module {
  func.func @mlir_quant_dot_fp4(%x1: !migraphx.shaped<1x1024xf8E4M3FN, 1024x1>,
                           %x2: !migraphx.shaped<1000x1024xf8E4M3FN, 1024x1>,
                           %x3: !migraphx.shaped<1x64x1xf32, 64x1x1>,
                           %x4: !migraphx.shaped<64x1x1000xf32, 1000x1000x1>,
                           %x5: !migraphx.shaped<1x1000xf32, 1000x1>)
        -> !migraphx.shaped<1x1000xf32, 1000x1>  attributes {kernel, arch="gfx950"} {
    %wT = migraphx.transpose %x2 {permutation = [1,0]}
          : <1000x1024xf8E4M3FN, 1024x1> -> <1024x1000xf8E4M3FN, 1x1024>
    %wTUnpacked = migraphx.unpack %wT {axis = 0}
          : <1024x1000xf8E4M3FN, 1x1024> -> <2048x1000xf8E4M3FN, 1x2048>
    %x1Unpacked = migraphx.unpack %x1 {axis = 1}
          : <1x1024xf8E4M3FN, 1024x1> -> <1x2048xf8E4M3FN, 2048x1>
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

    // Quant dot (assumed dequant inside by scales )
    %qd = migraphx.quant_dot %x1Unpacked scaled by %sE8A, %wTUnpacked scaled by %sE8B
          : !migraphx.shaped<1x2048xf8E4M3FN, 2048x1> scaled by !migraphx.shaped<1x2048xf8E8M0FNU, 2048x1>,
            !migraphx.shaped<2048x1000xf8E4M3FN, 1x2048> scaled by !migraphx.shaped<2048x1000xf8E8M0FNU, 1000x1>
            -> !migraphx.shaped<1x1000xf32, 1000x1>

    %out = migraphx.add %qd, %x5
          : !migraphx.shaped<1x1000xf32, 1000x1>,
            !migraphx.shaped<1x1000xf32, 1000x1>
            -> !migraphx.shaped<1x1000xf32, 1000x1>
    return %out : !migraphx.shaped<1x1000xf32, 1000x1>
  }
}