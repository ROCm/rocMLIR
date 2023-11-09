// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s
// RUN: rocmlir-gen --arch gfx908 -p --operation=conv2d | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func.func @{{.*}}(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}", rock.block_size = {{.*}} : i32, rock.grid_size = {{.*}} : i32, rock.wave_size = {{.*}} : i32}
