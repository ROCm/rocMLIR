// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.

// RUN: mlir-miopen-driver -p | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p --operation conv2d -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// fp16 tests.

// RUN: mlir-miopen-driver -p -t f16 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t f16 -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t f16 --operation conv2d -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// bf16(i16) tests.

// RUN: mlir-miopen-driver -p -t bf16 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t bf16 -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t bf16 --operation conv2d -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
