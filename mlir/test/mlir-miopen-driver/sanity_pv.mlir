// Sanity test to -pv produces valid MLIR for all possible data types and
// lowering paths.

// RUN: mlir-miopen-driver -p -pv -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv -t f16 -c | mlir-opt
// FIXME: mlir-miopen-driver -p -pv -t bf16 -c | mlir-opt
