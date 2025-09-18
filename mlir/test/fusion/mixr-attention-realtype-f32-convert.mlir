// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -targets %arch | rocmlir-driver -arch %arch -c --mlir-print-ir-after=rock-gridwise-gemm-to-blockwise -o /dev/null 2>&1 -debug-only=rock-gridwise-to-blockwise | FileCheck %s
// CHECK: elemTypeQLoad: f16
// CHECK: elemTypeKLoad: f32
// CHECK: elemTypeVLoad: f16
// CHECK: qVectorLen: 8
// CHECK: kVectorLen: 4
// CHECK: vVectorLen: 8
module {
  // CHECK: %[[TRANS0:.*]] = rock.transform %{{.*}} <Unmerge{32, 16, 8} ["k_loop", "k_thread", "k_iter"] at [0, 4, 6] -> ["k"] at [1]>
  // CHECK: %[[TRANS1:.*]] = rock.transform %[[TRANS0]]
  // CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%[[TRANS1]])
  func.func private @mlir_attention_f32(%arg0: !migraphx.shaped<4096x4096xf32, 4096x1>, %arg1: !migraphx.shaped<4096x4096xf32, 4096x1>, %arg2: !migraphx.shaped<4096x4096xf16, 4096x1>, %arg3: !migraphx.shaped<4096x4096xf16, 4096x1>) -> !migraphx.shaped<4096x4096xf16, 4096x1> attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = migraphx.add %arg0, %arg1 : <4096x4096xf32, 4096x1>, <4096x4096xf32, 4096x1> -> <4096x4096xf32, 4096x1>
    %1 = migraphx.convert %0 {target_type = 0 : i64} : <4096x4096xf32, 4096x1> to <4096x4096xf16, 4096x1>
    %2 = migraphx.dot %arg2, %1 : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %3 = migraphx.softmax %2 {axis = 1 : i64} : <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %4 = migraphx.dot %3, %arg3 {perf_config = "attn:v2:128,128,128,128,128,16,1,1,1,2,1"} : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    return %4 : !migraphx.shaped<4096x4096xf16, 4096x1>
  }
}
