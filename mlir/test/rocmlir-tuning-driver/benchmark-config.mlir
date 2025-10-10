// RUN: rocmlir-gen --operation gemm --arch %arch -t f32 -out_datatype f32 -transA=false -transB=false -g 1 -m 384 -n 768 -k 3072 --perf_config="v3:32,64,8,32,16,8,1,1,2,1,1" | rocmlir-tuning-driver --benchmark-config="v3:32,64,8,32,16,8,1,1,2,1,1" | FileCheck %s
// CHECK: {{v3:32,64,8,32,16,8,1,1,2,1,1[\t ].*}}
// CHECK-EMPTY:
