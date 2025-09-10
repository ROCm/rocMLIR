// RUN: rocmlir-gen --arch gfx942 --operation conv_gemm -t f32 -p | FileCheck %s

// CHECK: func.func @rock_conv_gemm
// CHECK-SAME: attributes {enable_splitk_for_tuning

// RUN: rocmlir-gen --arch gfx942 --operation conv_gemm -t f32 -p -disable-split-k-for-tuning | FileCheck %s --check-prefix=CHECK-NOSPLITK

// CHECK-NOSPLITK: func.func @rock_conv_gemm
// CHECK-NOSPLITK-NOT: enable_splitk_for_tuning
