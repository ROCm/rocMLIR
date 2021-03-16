// UNSUPPORTED: native
// Sanity test to -pv produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// FIXME: investigate why these commands fail on non-gfx908 CI nodes.
// RUN: mlir-miopen-driver -p -x2 -pv -c | mlir-opt
// RUN: mlir-miopen-driver -p -x2 -pv -t f16 -c | mlir-opt
// FIXME: mlir-miopen-driver -p -x2 -pv -t bf16 -c | mlir-opt
