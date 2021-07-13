// Sanity test to -pv_with_gpu produces valid MLIR for all possible data types and
// lowering paths.

// RUN: mlir-miopen-driver -p -pv_with_gpu -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv_with_gpu -t f16 -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv_with_gpu -t bf16 -c | mlir-opt
