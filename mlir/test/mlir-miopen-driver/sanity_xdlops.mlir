// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.

// RUN: miopen-gen -p -x2 | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-opt
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -chip=gfx908 | miopen-translate -gpu-module-to-rocdlir

// fp16 tests.

// RUN: miopen-gen -p -t f16 -x2 | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-opt
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t f16 -x2 | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -chip=gfx908 | miopen-translate -gpu-module-to-rocdlir

// bf16(i16) tests.

// RUN: miopen-gen -p -t bf16 -x2 | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-opt
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t bf16 -x2 | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -chip=gfx908 | miopen-translate -gpu-module-to-rocdlir

// i8 tests.

// RUN: miopen-gen -p -t i8 -x2 | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-opt
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-clean-math -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t i8 -x2 | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -chip=gfx908 | miopen-translate -gpu-module-to-rocdlir
