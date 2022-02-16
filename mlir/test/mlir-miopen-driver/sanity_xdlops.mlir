// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.

// RUN: miopen-gen -p -x2 | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -kernel-pipeline=rocdl | miopen-translate -gpu-module-to-rocdlir

// fp16 tests.

// RUN: miopen-gen -p -t f16 -x2 | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -kernel-pipeline=rocdl | miopen-translate -gpu-module-to-rocdlir

// bf16(i16) tests.

// RUN: miopen-gen -p -t bf16 -x2 | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -kernel-pipeline=rocdl | miopen-translate -gpu-module-to-rocdlir
