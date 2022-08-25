// Sanity test to -pv_with_cpp produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: miopen-gen -p -x2 -pv_with_cpp | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv_with_cpp -t f16 | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv_with_cpp -t bf16 | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv_with_cpp -t i8 | mlir-miopen-driver -c | miopen-opt
