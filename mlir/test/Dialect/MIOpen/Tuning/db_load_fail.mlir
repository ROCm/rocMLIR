// RUN: mlir-miopen-driver -p -miopen-affix-params -miopen-lowering -debug %s 2>&1 | FileCheck %s

// CHECK: Successfully opened connection to PerfDb
// Perfdb does not have an entry for default configurations
// CHECK: DB load failed
