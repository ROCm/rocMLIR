// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full | FileCheck %s
// CHECK: 64,32,32,4,2,4
