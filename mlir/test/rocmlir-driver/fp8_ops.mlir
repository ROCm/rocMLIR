// RUN: rocmlir-gen --arch gfx940 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=MFMA
// RUN: rocmlir-gen --arch gfx940 --operation gemm -mfma=off -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=MFMA_OFF
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=GFX11
// COM: This runs the kernel pipeline so that we still get a good test with the
// COM: host pipeline off as in the static library build, using the fact that
// COM: the fp8 expander isn't limited to GPU code.
// RUN: rocmlir-gen --arch gfx940 --operation gemm -t fp8 -p -pv | rocmlir-driver -kernel-pipeline=full | FileCheck %s --check-prefix=HOST

// MFMA: rocdl.mfma.f32.32x32x16.fp8.fp8
// MFMA-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// MFMA_OFF: rocdl.cvt.f32.fp8
// MFMA_OFF-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// GFX11: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// HOST: memref.global "private" constant @__rocmlir_extf_tbl_f8E4M3FNUZ
