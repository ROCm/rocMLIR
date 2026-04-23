// RUN: rocmlir-gen --arch gfx908 -p | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s
// RUN: rocmlir-gen --arch gfx908 -p --operation=conv | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s

// CHECK: module {{.*}}
// CHECK-NEXT: func.func @{{.*}}(%{{.*}}: memref<{{.*}}>, %{{.*}}: memref<{{.*}}>, %arg2: memref<{{.*}}>) attributes {block_size = {{.*}} : i32, features = {{.*}}, mhal.arch = "{{.*}}", rock.enable_splitk_for_tuning, rock.kernel = 0 : i32, rock.num_chiplets = {{.*}}, rock.num_cu = {{.*}}}
