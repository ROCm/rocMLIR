// RUN: not rocmlir-gen -operation gemm -t f32 -out_datatype f32 --arch %arch -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --kernel-repeats=100 2>&1 | FileCheck %s --check-prefix=ERR_KERNEL_REPEATS
// ERR_KERNEL_REPEATS: --kernel-repeats is only supported with host harness (-ph) or CPU validation (-pv).
