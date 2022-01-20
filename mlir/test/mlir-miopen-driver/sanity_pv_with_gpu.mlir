// Sanity test to -pv_with_gpu produces valid MLIR for all possible data types and
// lowering paths.

// RUN: miopen-gen -p -pv_with_gpu | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -pv_with_gpu -t f16 | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -pv_with_gpu -t bf16 | mlir-miopen-driver -c | miopen-opt
