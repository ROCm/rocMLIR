// RUN: mlir-miopen-driver -p -miopen-lowering -miopen-affine-transform -miopen-affix-params | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {block_size = {{.*}} : i32, grid_size = {{.*}} : i32, kernel}
