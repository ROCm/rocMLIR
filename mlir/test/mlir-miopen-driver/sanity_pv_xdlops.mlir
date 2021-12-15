// Sanity test to -pv produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: miopen-gen -p -x2 -pv | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv -t f16 | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -x2 -pv -t bf16 | mlir-miopen-driver -c | miopen-opt
