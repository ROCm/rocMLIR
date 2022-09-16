// Sanity test to -pv_with_gpu produces valid MLIR for all possible data types and
// lowering paths.

// RUN: rock-gen -p -pv_with_gpu | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -pv_with_gpu -t f16 | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -pv_with_gpu -t bf16 | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -pv_with_gpu -t i8 | mlir-rock-driver -c | rock-opt
