// RUN: rocmlir-gen --arch gfx900 --operation gemm -p -ph --kernel-repeats=5 | FileCheck %s
// CHECK-LABEL: @rock_gemm_gpu
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[five:.*]] = arith.constant 5 : index
// CHECK: scf.for %{{.*}} = %[[zero]] to %[[five]] step %[[one]] {
// CHECK-NEXT: func.call @rock_gemm
// CHECK-NEXT: }
