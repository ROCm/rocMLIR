// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.

// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rocmlir-translate -gpu-module-to-rocdlir

// fp16 tests.

// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=on | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rocmlir-translate -gpu-module-to-rocdlir

// bf16(i16) tests.

// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=on | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rocmlir-translate -gpu-module-to-rocdlir

// i8 tests.

// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-opt
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-loops-to-cf -convert-rock-to-gpu "-convert-gpu-to-rocdl=chipset=gfx908 index-bitwidth=32" | rocmlir-translate -gpu-module-to-rocdlir
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=on | rocmlir-driver -kernel-pipeline=gpu,rocdl -target=gfx908 | rocmlir-translate -gpu-module-to-rocdlir
