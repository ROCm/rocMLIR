// This is E2E compilation test to ensure gfx908 specific inlineAsm hack
// does not blow up in the backend compiler.

// RUN: rocmlir-gen -operation gemm -t f32 -out_datatype f32 --arch gfx908:sramecc+:xnack- --num_cu 120 -g 1 -m 24576 -k 768 -n 3072 -transA=False -transB=False --kernel-repeats 5 --perf_config= | rocmlir-driver -c | FileCheck %s
// CHECK: gpu.binary
