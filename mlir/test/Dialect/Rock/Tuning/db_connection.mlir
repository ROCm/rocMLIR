// RUN: rock-gen -p %s | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm -debug 2>&1 | FileCheck %s

// CHECK: Successfully opened connection to PerfDb
