// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.

// RUN: mlir-miopen-driver -p -x2 | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -x2 -c -target=rocdl | miopen-translate -mlir-to-rocdlir

// fp16 tests.

// RUN: mlir-miopen-driver -p -t f16 -x2 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -t f16 -x2 -c -target=rocdl | miopen-translate -mlir-to-rocdlir

// bf16(i16) tests.

// RUN: mlir-miopen-driver -p -t bf16 -x2 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -t bf16 -x2 -c -target=rocdl | miopen-translate -mlir-to-rocdlir
