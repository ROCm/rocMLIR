// RUN: mlir-miopen-driver -p -miopen-lowering -miopen-affix-params -debug %s 2>&1 | FileCheck %s --check-prefix=CHECK1
// RUN: mlir-miopen-driver -p -t f16 -miopen-lowering -miopen-affix-params -debug %s 2>&1 | FileCheck %s --check-prefix=CHECK2

// CHECK1: Successfully opened connection to PerfDb
// Perfdb does not have an entry for default configurations
// CHECK1: DB load failed
// CHECK2: Successfully opened connection to PerfDb
// Perfdb does not have an entry for default configurations
// CHECK2: DB load failed
