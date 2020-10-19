// RUN: mlir-miopen-driver -miopen-lowering -miopen-affix-params -debug %s 2>&1 | FileCheck %s

// CHECK: Successfully opened connection to PerfDb
