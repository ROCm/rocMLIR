// Sanity test to -pv_with_gpu produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: rocmlir-gen -p -feature mfma -pv_with_gpu | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen -p -feature mfma -pv_with_gpu -t f16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen -p -feature mfma -pv_with_gpu -t bf16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen -p -feature mfma -pv_with_gpu -t i8 | rocmlir-driver -c | rocmlir-opt
