// Sanity test to -pv_with_cpp produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: rock-gen -p -x2 -pv_with_cpp | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -x2 -pv_with_cpp -t f16 | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -x2 -pv_with_cpp -t bf16 | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -x2 -pv_with_cpp -t i8 | mlir-rock-driver -c | rock-opt
