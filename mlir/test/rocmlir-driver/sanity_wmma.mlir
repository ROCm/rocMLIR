// Sanity test to ensure every step of the WMMA lowering process gets valid MLIR
// and LLVM IR.

// fp16 tests.
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx1100 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx1100 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx1100
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx1100 | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx1100 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx1100 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx1100
// RUN: rocmlir-gen --arch gfx1100 -wmma=on -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx1100 | rocmlir-opt
