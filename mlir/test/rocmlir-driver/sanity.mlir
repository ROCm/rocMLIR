// Sanity test to ensure every step of the lowering process gets valid MLIR,
// LLVM IR, and AMD GCN ISA.

// fp32 tests.
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx900 | rocmlir-opt

// fp16 tests.
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t f16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t f16 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx900 | rocmlir-opt

// bf16 tests.
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t bf16 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t bf16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx900 | rocmlir-opt

// i8 tests
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t i8 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t i8 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,rocdl --verify-passes --arch=gfx900 | rocmlir-translate -gpu-module-to-rocdlir | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t i8 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx900 | rocmlir-opt

// Check for MLIRContext options

// RUN: rocmlir-gen --arch gfx900 -mfma=off -atomic_add=off -dot=off -p -t i8 --operation conv --mlir-disable-threading | rocmlir-driver -kernel-pipeline=gpu,rocdl --arch=gfx900 --mlir-disable-threading | rocmlir-translate -gpu-module-to-rocdlir --mlir-disable-threading | opt -passes='default<O3>,strip' -S | llc -mcpu=gfx900
