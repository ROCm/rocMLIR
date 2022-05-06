// Sanity test to -pv_with_gpu produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: miopen-gen -p -x2 -pv_with_gpu | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv_with_gpu -t f16 | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv_with_gpu -t bf16 | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv_with_gpu -t i8 | mlir-miopen-driver -c | miopen-opt
