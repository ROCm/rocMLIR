// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR.

// fp32 tests.

// RUN: mlir-miopen-driver -p -x2 | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-lowering | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 | mlir-opt
// FIXME: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | mlir-opt
// FIXME: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | mlir-opt
// FIXME: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | mlir-opt
// FIXME: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-opt

// fp16 tests.

// RUN: mlir-miopen-driver -p -t f16 -x2 | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 | mlir-opt
// FIXME: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | mlir-opt
// FIXME: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | mlir-opt
// FIXME: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | mlir-opt
// FIXME: mlir-miopen-driver -p -t f16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-opt

// bf16(i16) tests.

// RUN: mlir-miopen-driver -p -t bf16 -x2 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-opt
