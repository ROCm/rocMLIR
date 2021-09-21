// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.

// RUN: mlir-miopen-driver -p | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-opt
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip
// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -c -target=rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p --operation conv2d -c -target=rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900

// fp16 tests.

// RUN: mlir-miopen-driver -p -t f16 | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-opt
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip
// RUN: mlir-miopen-driver -p -t f16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t f16 -c -target=rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t f16 --operation conv2d -c -target=rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900

// bf16(i16) tests.

// RUN: mlir-miopen-driver -p -t bf16 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-opt
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip
// RUN: mlir-miopen-driver -p -t bf16 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu -convert-gpu-to-rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t bf16 -c -target=rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900
// RUN: mlir-miopen-driver -p -t bf16 --operation conv2d -c -target=rocdl | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx900
