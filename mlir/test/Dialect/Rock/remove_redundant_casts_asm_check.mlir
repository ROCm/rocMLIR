// This test checks that we are able to remove redundant cast chains. In this
// case, we want to check the final generated assembly to make sure that there
// are no remaining casts after the transformation.

//RUN: sed -e 's/##TOKEN_ARCH##/%arch/g' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-driver -arch gfx942:sramecc+:xnack- -c --debug-only=serialize-to-isa | FileCheck %s

//CHECK-NOT: v_cvt

module {
  func.func @dot_add(%arg0: !migraphx.shaped<1x5x4xf16, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf16, 12x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes{kernel, arch = "##TOKEN_ARCH##"} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf16, 20x4x1>, <1x4x3xf16, 12x3x1> -> <1x5x3xf16, 15x3x1>
    %1 = migraphx.convert %0 : <1x5x3xf16, 15x3x1> to <1x5x3xf32, 15x3x1>
    return %1 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}