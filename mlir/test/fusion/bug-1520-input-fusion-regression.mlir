// RUN: rocmlir-driver --kernel-pipeline=migraphx,highlevel %s | FileCheck %s
// Note: if we move bufferization, this should be a check for three emptys or
// alloc_tensors unless there's a wider refactoring.
// CHECK-COUNT-3: memref.alloc
// CHECK-NOT: memref.alloc
module {
  func.func @mlir_add_sigmoid_mul_slice_reshape_transpose_slice_dot_add_sigmoid_mul_add_add_tanh_sub_mul_add(%arg0: !migraphx.shaped<2x5xf32, 5x1>, %arg1: !migraphx.shaped<2x5xf32, 5x1>, %arg2: !migraphx.shaped<2x5xf32, 5x1>, %arg3: !migraphx.shaped<2x5xf32, 5x1>, %arg4: !migraphx.shaped<2x5xf32, 5x1>, %arg5: !migraphx.shaped<2x5xf32, 5x1>, %arg6: !migraphx.shaped<2x5xf32, 5x1>, %arg7: !migraphx.shaped<2x5xf32, 5x1>, %arg8: !migraphx.shaped<2x15x5xf32, 75x5x1>) -> !migraphx.shaped<2x5xf32, 5x1> attributes {arch = "gfx908:sramecc+:xnack-", enable_splitk_for_tuning = true, kernel = "mixr", num_cu = 120 : i64} {
    %0 = migraphx.add %arg0, %arg1 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %1 = migraphx.sigmoid %0 : <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %2 = migraphx.mul %1, %arg2 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %3 = migraphx.slice %arg8 {axes = [0], ends = [1], starts = [0]} : <2x15x5xf32, 75x5x1> -> <1x15x5xf32, 75x5x1>
    %4 = migraphx.reshape %3 {dims = [15, 5]} : <1x15x5xf32, 75x5x1> -> <15x5xf32, 5x1>
    %5 = migraphx.transpose %4 {permutation = [1, 0]} : <15x5xf32, 5x1> -> <5x15xf32, 1x5>
    %6 = migraphx.slice %5 {axes = [1], ends = [15], starts = [10]} : <5x15xf32, 1x5> -> <5x5xf32, 15x1>
    %7 = migraphx.dot %2, %6 : <2x5xf32, 5x1>, <5x5xf32, 15x1> -> <2x5xf32, 5x1>
    %8 = migraphx.add %arg3, %arg4 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %9 = migraphx.sigmoid %8 : <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %10 = migraphx.mul %9, %arg2 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %11 = migraphx.add %7, %arg5 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %12 = migraphx.add %arg6, %11 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %13 = migraphx.tanh %12 : <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %14 = migraphx.sub %arg7, %9 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %15 = migraphx.mul %14, %13 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    %16 = migraphx.add %15, %10 : <2x5xf32, 5x1>, <2x5xf32, 5x1> -> <2x5xf32, 5x1>
    return %16 : !migraphx.shaped<2x5xf32, 5x1>
  }
}
