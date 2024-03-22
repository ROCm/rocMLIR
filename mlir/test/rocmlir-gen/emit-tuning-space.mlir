// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-NAVI
// CHECK-NAVI: v2:64,32,32,4,2,4,1

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=32 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: v2:64,64,8,16,16,4,2,1,1
