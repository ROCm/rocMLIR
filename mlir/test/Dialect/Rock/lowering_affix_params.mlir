// REQUIRES: rock-driver
// RUN: rocmlir-gen --arch %targetChip -p | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s
// RUN: rocmlir-gen --arch %targetChip -p --operation=conv2d | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func.func @{{.*}}(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {block_size = {{.*}} : i32, grid_size = {{.*}} : i32, kernel = 0 : i32}
