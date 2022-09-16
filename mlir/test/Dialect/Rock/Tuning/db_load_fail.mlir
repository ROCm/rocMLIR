// RUN: rock-gen -p %s | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -debug 2>&1 | FileCheck %s

// CHECK: Successfully opened connection to PerfDb
// Perfdb does not have an entry for default configurations
// CHECK: DB load failed
