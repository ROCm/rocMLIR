// Sanity test to -pv_with_cpp produces valid MLIR for all possible data types and
// lowering paths.

// RUN: rocmlir-gen --arch gfx900 -p -pv_with_cpp | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -p -pv_with_cpp -t f16 | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -p -pv_with_cpp -t bf16 | rocmlir-driver -c --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx900 -p -pv_with_cpp -t i8 | rocmlir-driver -c --verify-passes | rocmlir-opt
