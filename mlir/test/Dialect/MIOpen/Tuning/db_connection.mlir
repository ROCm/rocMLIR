// RUN: mlir-miopen-driver -p -miopen-lowering -miopen-affix-params -debug %s 2>&1 | FileCheck %s

// CHECK: Successfully opened connection to PerfDb
