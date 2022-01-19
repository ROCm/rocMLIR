// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.

// RUN: miopen-gen -p | miopen-opt
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl
    
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p | mlir-miopen-driver -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p --operation conv2d | mlir-miopen-driver -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// fp16 tests.

// RUN: miopen-gen -p -t f16 | miopen-opt
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t f16 | mlir-miopen-driver -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t f16 --operation conv2d | mlir-miopen-driver -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// bf16(i16) tests.

// RUN: miopen-gen -p -t bf16 | miopen-opt
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t bf16 | mlir-miopen-driver -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t bf16 --operation conv2d | mlir-miopen-driver -c -target=rocdl | miopen-translate -mlir-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
