// Sanity test to -pv produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: rocmlir-gen -p -x2 -pv | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen -p -x2 -pv -t f16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen -p -x2 -pv -t bf16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen -p -x2 -pv -t i8 | rocmlir-driver -c | rocmlir-opt
