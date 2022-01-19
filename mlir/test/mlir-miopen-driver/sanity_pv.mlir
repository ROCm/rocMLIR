// Sanity test to -pv produces valid MLIR for all possible data types and
// lowering paths.

// RUN: miopen-gen -p -pv | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -pv -t f16 | mlir-miopen-driver -c | miopen-opt
// RUN: miopen-gen -p -pv -t bf16 | mlir-miopen-driver -c | miopen-opt
