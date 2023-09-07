// Sanity test to -pv produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: rocmlir-gen --arch gfx908 -p -mfma=on -pv | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -mfma=on -pv -t f16 | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -mfma=on -pv -t bf16 | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx908 -p -mfma=on -pv -t i8 | rocmlir-driver -c --verify-passes | rocmlir-opt
