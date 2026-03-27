// RUN: rocmlir-gen -g 1 -m 64 -k 32 -n 32 --transA=true --transB=false -t fp8 --operation gemm --arch gfx950 --perf_config "v3:64,32,32,32,32,8,1,4,2,1,1" | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=MFMA_32x32x64
// RUN: rocmlir-gen -g 1 -m 32 -k 32 -n 16 --transA=true --transB=false -t fp8 --operation gemm --arch gfx950 --perf_config "v3:32,32,16,16,16,16,1,4,2,1,1" | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=MFMA_16x16x128

// Verify that FP8 GEMM using scaled MFMA with neutral scales generates
// non-scaled MFMA ISA instruction. The LLVM backend optimization
// (UnscaledMFMAOptimizationPat) converts scaled MFMA with scale=0 to
// the corresponding non-scaled instruction.

// MFMA_32x32x64: v_mfma_f32_32x32x64_f8f6f4
// MFMA_32x32x64-NOT: v_mfma_scale

// MFMA_16x16x128: v_mfma_f32_16x16x128_f8f6f4
// MFMA_16x16x128-NOT: v_mfma_scale
