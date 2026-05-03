// Verify the implicit `rock.kernel`-polarity contract between
// `migraphx-transform` (host-side AttentionDecompose) and
// `migraphx-attention-to-rock` (kernel-side AttentionToRock). Both passes
// must run in sequence on the same module without stepping on each other:
// the host path must fully decompose `migraphx.attention` into primitive
// migraphx ops, and the kernel path must produce exactly one
// `rock.attention` with no leftover `migraphx.attention`. The two
// pre-existing single-pass tests (attention-decompose.mlir and
// attention-to-rock.mlir) only cover each side in isolation; this file
// pins the cross-pass contract so anyone reordering or reshaping the
// pipeline (or accidentally flipping a polarity guard) breaks here.
//
// RUN: rocmlir-opt --migraphx-transform --migraphx-attention-to-rock %s | FileCheck %s

// Host (no rock.kernel attribute): MIGraphXTransform decomposes
// migraphx.attention to dot + softmax + dot. MIGraphXAttentionToRock then
// runs but should be a no-op because the function isn't a kernel and
// because there's no migraphx.attention left to lower anyway. Final IR
// must contain neither migraphx.attention nor rock.attention.
// CHECK-LABEL: func.func @host_attention_decomposed
// CHECK: migraphx.dot
// CHECK: migraphx.softmax
// CHECK: migraphx.dot
// CHECK-NOT: migraphx.attention
// CHECK-NOT: rock.attention
func.func @host_attention_decomposed(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v {
  } : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// Kernel (rock.kernel attribute set): MIGraphXTransform skips this
// function (its AttentionDecompose pattern guards on !hasAttr("rock.kernel")),
// so migraphx.attention survives. MIGraphXAttentionToRock then converts
// it to rock.attention exactly once. Final IR must contain no
// migraphx.attention and exactly one rock.attention.
// CHECK-LABEL: func.func @kernel_attention_to_rock
// CHECK-SAME: attributes {arch = "", rock.kernel}
// CHECK-NOT: migraphx.attention
// CHECK: rock.attention
// CHECK-NOT: migraphx.attention
// CHECK-NOT: rock.attention
func.func @kernel_attention_to_rock(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> attributes {rock.kernel, arch = ""} {
  %0 = migraphx.attention %q, %k, %v {
  } : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}
