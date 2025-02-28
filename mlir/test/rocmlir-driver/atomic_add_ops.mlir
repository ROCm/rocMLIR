// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_COMPSWAP
// RUN: rocmlir-gen --arch gfx90a --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=ATOMICRMW

// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_COMPSWAP
// RUN: rocmlir-gen --arch gfx942 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=ATOMICRMW

// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx950 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=ATOMICRMW

// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_COMPSWAP
// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_COMPSWAP
// RUN: rocmlir-gen --arch gfx1100 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=ATOMICRMW

// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t f16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t bf16 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=BUFFER_ATOMIC_ADD
// RUN: rocmlir-gen --arch gfx1201 --store-method atomic_add --operation gemm -t f32 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=ATOMICRMW

// BUFFER_ATOMIC_ADD: rocdl.raw.ptr.buffer.atomic.fadd
// ATOMICRMW: llvm.atomicrmw
// BUFFER_ATOMIC_COMPSWAP: rocdl.raw.ptr.buffer.atomic.cmpswap
