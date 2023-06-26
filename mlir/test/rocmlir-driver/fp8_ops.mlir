// RUN: rocmlir-gen --arch gfx940 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=MFMA
// RUN: rocmlir-gen --arch gfx940 --operation gemm -mfma=off -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=MFMA_OFF
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t fp8 -p | rocmlir-driver --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=GFX11
// RUN: rocmlir-gen --arch gfx940 --operation gemm -t fp8 -p -pv | rocmlir-driver -c | FileCheck %s --check-prefix=HOST

// MFMA: rocdl.mfma.f32.32x32x16.fp8.fp8
// MFMA-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// MFMA_OFF: rocdl.cvt.f32.fp8
// MFMA_OFF-NOT: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// GFX11: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
// HOST: llvm.mlir.global private constant @__rocmlir_extf_tbl_f8E4M3FNUZ
