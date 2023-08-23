// Sanity test to ensure every step of the XDLOPS lowering process gets valid MLIR
// and LLVM IR.

// fp32 tests.
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx908
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx908 | rocmlir-opt

// fp16 tests.
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx908
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx908 | rocmlir-opt

// bf16 tests.
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t bf16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx908
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx908 | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx908 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx908
// RUN: rocmlir-gen --arch gfx908 -mfma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx908 | rocmlir-opt
