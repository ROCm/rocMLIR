// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.

// RUN: rock-gen -p | rock-opt
// RUN: rock-gen -p | rock-opt -rock-affix-params
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"

// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rock-gen -p | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p --operation conv2d | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// fp16 tests.

// RUN: rock-gen -p -t f16 | rock-opt
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rock-gen -p -t f16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p -t f16 | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p -t f16 --operation conv2d | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// bf16(i16) tests.

// RUN: rock-gen -p -t bf16 | rock-opt
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rock-gen -p -t bf16 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p -t bf16 | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p -t bf16 --operation conv2d | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// i8 tests

// RUN: rock-gen -p -t i8 | rock-opt
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rock-gen -p -t i8 | rock-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p -t i8 | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rock-gen -p -t i8 --operation conv2d | mlir-rock-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rock-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
