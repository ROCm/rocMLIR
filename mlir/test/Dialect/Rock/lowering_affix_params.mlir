// REQUIRES: rock-driver
// RUN: rock-gen -p | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s
// RUN: rock-gen -p --operation=conv2d | mlir-rock-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func.func @{{.*}}(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {block_size = {{.*}} : i32, grid_size = {{.*}} : i32, kernel = 0 : i32}
