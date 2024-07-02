// RUN: rocmlir-gen --arch gfx90a -p -t f16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes | FileCheck %s 

// CHECK: gpu.binary {{.*}} rock.blocks_per_cu = {{.*}} : i32
