// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.

// RUN: miopen-gen -p | miopen-opt
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"

// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: miopen-gen -p | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p --operation conv2d | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// fp16 tests.

// RUN: miopen-gen -p -t f16 | miopen-opt
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: miopen-gen -p -t f16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t f16 | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t f16 --operation conv2d | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// bf16(i16) tests.

// RUN: miopen-gen -p -t bf16 | miopen-opt
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: miopen-gen -p -t bf16 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t bf16 | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t bf16 --operation conv2d | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// i8 tests

// RUN: miopen-gen -p -t i8 | miopen-opt
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: miopen-gen -p -t i8 | miopen-opt -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise -miopen-blockwise-gemm-to-threadwise -miopen-threadwise-gemm-lowering -miopen-sugar-to-loops -miopen-loops-to-cf -convert-miopen-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t i8 | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: miopen-gen -p -t i8 --operation conv2d | mlir-miopen-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | miopen-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
