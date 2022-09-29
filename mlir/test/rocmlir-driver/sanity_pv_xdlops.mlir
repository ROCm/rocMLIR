// Sanity test to -pv produces valid MLIR for all possible data types and
// XDLOPS lowering paths.

// RUN: rocmlir-gen --arch %targetChip -p -feature mfma -pv | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch %targetChip -p -feature mfma -pv -t f16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch %targetChip -p -feature mfma -pv -t bf16 | rocmlir-driver -c | rocmlir-opt
// RUN: rocmlir-gen --arch %targetChip -p -feature mfma -pv -t i8 | rocmlir-driver -c | rocmlir-opt
