// RUN: rocmlir-gen --arch gfx900 --operation gemm -p -ph --kernel-repeats=5 | FileCheck %s --check-prefix=GEMM
// RUN: rocmlir-gen --arch gfx900 --operation conv -p -ph --kernel-repeats=5 | FileCheck %s --check-prefix=CONV

// GEMM-LABEL: @rock_gemm_gpu
// GEMM-DAG: %[[zero:.*]] = arith.constant 0 : index
// GEMM-DAG: %[[one:.*]] = arith.constant 1 : index
// GEMM-DAG: %[[five:.*]] = arith.constant 5 : index
// GEMM: scf.for %{{.*}} = %[[zero]] to %[[five]] step %[[one]] {
// GEMM-NEXT: func.call @rock_gemm
// GEMM-NEXT: }

// CONV-LABEL: @rock_conv_gkc01_ngc01_ngk01_gpu
// CONV-DAG: %[[zero:.*]] = arith.constant 0 : index
// CONV-DAG: %[[one:.*]] = arith.constant 1 : index
// CONV-DAG: %[[five:.*]] = arith.constant 5 : index
// CONV: scf.for %{{.*}} = %[[zero]] to %[[five]] step %[[one]] {
// CONV: func.call @rock_conv_gkc01_ngc01_ngk01
// CONV: }
