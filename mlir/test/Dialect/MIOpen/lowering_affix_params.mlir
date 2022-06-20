// REQUIRES: miopen-driver
// RUN: miopen-gen -p | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm | FileCheck %s
// RUN: miopen-gen -p --operation=conv2d | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func @{{.*}}(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {block_size = {{.*}} : i32, grid_size = {{.*}} : i32, kernel = 0 : i32}
