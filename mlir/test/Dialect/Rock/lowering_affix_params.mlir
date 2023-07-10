// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s
// RUN: rocmlir-gen --arch gfx908 -p --operation=conv2d | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func.func @{{.*}}(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {block_size = {{.*}} : i32, grid_size = {{.*}} : i32, kernel = 0 : i32, mhal.arch = "{{.*}}", wave_size = {{.*}} : i32}
