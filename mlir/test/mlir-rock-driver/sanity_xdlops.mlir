// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.

// RUN: rock-gen -p -x2 | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-opt
// RUN: rock-gen -p -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p -x2 | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rock-translate -gpu-module-to-rocdlir

// fp16 tests.

// RUN: rock-gen -p -t f16 -x2 | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-opt
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p -t f16 -x2 | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rock-translate -gpu-module-to-rocdlir

// bf16(i16) tests.

// RUN: rock-gen -p -t bf16 -x2 | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-opt
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p -t bf16 -x2 | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rock-translate -gpu-module-to-rocdlir

// i8 tests.

// RUN: rock-gen -p -t i8 -x2 | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-opt
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p -t i8 -x2 | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rock-translate -gpu-module-to-rocdlir
