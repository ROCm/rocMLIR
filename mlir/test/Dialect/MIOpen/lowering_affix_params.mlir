// REQUIRES: miopen-driver
// RUN: mlir-miopen-driver -p -miopen-lowering -miopen-affine-transform -miopen-affix-params | FileCheck %s
// RUN: mlir-miopen-driver -p --operation=conv2d_dummy -miopen-lowering -miopen-affine-transform -miopen-affix-params | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func @{{.*}}(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {block_size = {{.*}} : i32, grid_size = {{.*}} : i32, kernel = 0 : i32}
