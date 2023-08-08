// Sanity test to -pv produces valid MLIR for all possible data types and
// lowering paths.

// RUN: rocmlir-gen --arch gfx900 -p -pv | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -p -pv -t f16 | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -p -pv -t bf16 | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -p -pv -t i8 | rocmlir-driver -c --verify-passes | rocmlir-opt
