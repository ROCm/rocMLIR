// Sanity test to -pv produces valid MLIR for all possible data types and
// lowering paths.

// RUN: rocmlir-gen --arch %targetChip -p -pv | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch %targetChip -p -pv -t f16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch %targetChip -p -pv -t bf16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch %targetChip -p -pv -t i8 | rocmlir-driver -c | rocmlir-opt
