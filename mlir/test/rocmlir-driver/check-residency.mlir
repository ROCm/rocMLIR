// RUN: rocmlir-gen --arch gfx90a -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes | FileCheck %s 

// CHECK: llvm.func @{{.*}}({{.*}}) attributes {{.*}} rock.blocks_per_cu = 3 :
