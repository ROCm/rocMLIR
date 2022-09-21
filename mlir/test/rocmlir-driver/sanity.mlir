// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.

// RUN: rocmlir-gen -p | rocmlir-opt
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"

// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rocmlir-gen -p | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p --operation conv2d | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// fp16 tests.

// RUN: rocmlir-gen -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rocmlir-gen -p -t f16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p -t f16 --operation conv2d | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// bf16(i16) tests.

// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rocmlir-gen -p -t bf16 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p -t bf16 --operation conv2d | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900

// i8 tests

// RUN: rocmlir-gen -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32"
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S
// RUN: rocmlir-gen -p -t i8 | rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx900 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen -p -t i8 --operation conv2d | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
