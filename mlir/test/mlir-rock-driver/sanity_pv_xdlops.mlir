// Sanity test to -pv produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: rock-gen -p -x2 -pv | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -x2 -pv -t f16 | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -x2 -pv -t bf16 | mlir-rock-driver -c | rock-opt
// RUN: rock-gen -p -x2 -pv -t i8 | mlir-rock-driver -c | rock-opt
