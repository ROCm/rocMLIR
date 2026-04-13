// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -targets %arch | rocmlir-driver -arch %arch -c --mlir-print-ir-after=rock-blockwise-load-tile-to-threadwise -o /dev/null 2>&1 -debug-only=rock-gridwise-to-blockwise | FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/gfx942/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -targets gfx942 | rocmlir-driver -arch gfx942 -c --mlir-print-ir-after=rock-blockwise-load-tile-to-threadwise -o /dev/null 2>&1 -debug-only=rock-gridwise-to-blockwise | FileCheck %s --check-prefix=VECTORIZATION

// CHECK: elemTypeQLoad: f16
// CHECK: elemTypeKLoad: i4
// CHECK: elemTypeVLoad: f16
// VECTORIZATION: qVectorLen: 8
// VECTORIZATION: kVectorLen: 32
// VECTORIZATION: vVectorLen: 8
module {
  // VECTORIZATION: %[[TRANS0:.*]] = rock.transform %{{.*}} <Unmerge{64, 2, 32} ["m_block", "m_thread", "m_iter"] at [2, 5, 7] -> ["m"] at [2]>
  // VECTORIZATION: %[[TRANS1:.*]] = rock.transform %[[TRANS0]]
  // VECTORIZATION: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%[[TRANS1]])
  func.func private @mlir_attention_int4(%arg0: !migraphx.shaped<4096x4096xf16, 8192x1>, %arg1: !migraphx.shaped<4096xf16, 1>, %arg2: !migraphx.shaped<4096xf16, 1>, %arg3: !migraphx.shaped<4096x2048xui8, 2048x1>, %arg4: !migraphx.shaped<4096x4096xf16, 4096x1>) -> !migraphx.shaped<4096x4096xf16, 4096x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.unpack %arg3 {axis = 1 : i64} : <4096x2048xui8, 2048x1> -> <4096x4096xi8, 4096x1>
    %1 = migraphx.broadcast %arg1 {axis = 0 : i64, out_lens = [4096, 4096]} : <4096xf16, 1> -> <4096x4096xf16, 0x1>
    %2 = migraphx.broadcast %arg2 {axis = 0 : i64, out_lens = [4096, 4096]} : <4096xf16, 1> -> <4096x4096xf16, 0x1>
    %3 = migraphx.reshape %1 {dims = [4096, 4096]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
    %4 = migraphx.reshape %2 {dims = [4096, 4096]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
    %5 = migraphx.dequantizelinear %0, %3, %4 : <4096x4096xi8, 4096x1>, <4096x4096xf16, 16536x2>, !migraphx.shaped<4096x4096xf16, 16536x2> -> <4096x4096xf16, 4096x1>
    %6 = migraphx.dot %2, %5 : <4096x4096xf16, 0x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %7 = migraphx.softmax %6 {axis = 1 : i64} : <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %8 = migraphx.dot %7, %arg4 {perf_config = "attn:v2:64,128,128,128,16,16,1,1,1,2,1"} : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    return %8 : !migraphx.shaped<4096x4096xf16, 4096x1>
  }
}
