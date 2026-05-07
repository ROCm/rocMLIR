// Regression test for ROCm/rocMLIR-internal#2124:
// FP4 scaled-GEMM tuning previously crashed the AMDGPU backend at optLevel=3
// for specific perfConfigs on gfx950. The fix landed via upstream LLVM AMDGPU
// codegen + scaled-GEMM lowering. This test ensures the previously crashing
// perfConfigs continue to compile and execute end-to-end on gfx950.
//
// rocmlir-tuning-driver hard-codes optLevel=3 (see
// mlir/tools/rocmlir-tuning-driver/rocmlir-tuning-driver.cpp), so this
// exercises the same optimization level that originally triggered the crash.

// Original failing perfConfig from the issue body
// (-m 256 -n 50272 -k 768 with f8E8M0FNU scales).
// RUN: rocmlir-gen --arch %arch --operation gemm \
// RUN:   -t f4E2M1FN -out_datatype f32 -scaledGemm \
// RUN:   -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU \
// RUN:   -g 1 -m 256 -n 50272 -k 768 \
// RUN:   | rocmlir-tuning-driver --benchmark-config="v3:128,256,8,32,16,32,1,1,2,1,1" \
// RUN:   | FileCheck %s --check-prefix=ORIGINAL
// ORIGINAL: {{v3:128,256,8,32,16,32,1,1,2,1,1[\t ]+[0-9].*}}

// Additional failing perfConfig reported in the issue comments
// (same shape as above).
// RUN: rocmlir-gen --arch %arch --operation gemm \
// RUN:   -t f4E2M1FN -out_datatype f32 -scaledGemm \
// RUN:   -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU \
// RUN:   -g 1 -m 256 -n 50272 -k 768 \
// RUN:   | rocmlir-tuning-driver --benchmark-config="v3:32,256,8,32,16,32,1,1,2,1,1" \
// RUN:   | FileCheck %s --check-prefix=COMMENT_A
// COMMENT_A: {{v3:32,256,8,32,16,32,1,1,2,1,1[\t ]+[0-9].*}}

// Second case from the issue comments: -m 1 -n 1000 -k 2048 with f32 scales,
// transposed B. Originally failed at both optLevel=2 and optLevel=0.
// RUN: rocmlir-gen --arch %arch --operation gemm \
// RUN:   -t f4E2M1FN -out_datatype f32 -transA=false -transB=true \
// RUN:   -scaledGemm -scale_a_dtype f32 -scale_b_dtype f32 \
// RUN:   -transScaleA=false -transScaleB=true \
// RUN:   -g 1 -m 1 -n 1000 -k 2048 \
// RUN:   | rocmlir-tuning-driver --benchmark-config="v3:256,256,4,128,32,32,1,1,2,1,1" \
// RUN:   | FileCheck %s --check-prefix=COMMENT_B
// COMMENT_B: {{v3:256,256,4,128,32,32,1,1,2,1,1[\t ]+[0-9].*}}
