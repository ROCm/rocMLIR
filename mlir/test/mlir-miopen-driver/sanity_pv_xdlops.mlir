// UNSUPPORTED: native
// Sanity test to -pv produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: mlir-miopen-driver -p -x2 -pv -c | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -pv -t f16 -c | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -pv -t bf16 -c | mlir-opt
