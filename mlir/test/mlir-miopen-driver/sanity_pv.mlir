// Sanity test to -pv produces valid MLIR for all possible data types and
// lowering paths.

// RUN: mlir-miopen-driver -p -pv -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv -x2 -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv -t f16 -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv -t f16 -x2 -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv -t bf16 -c | mlir-opt
// RUN: mlir-miopen-driver -p -pv -t bf16 -x2 -c | mlir-opt
