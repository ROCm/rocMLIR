// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,5,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,1,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_dot_add(%arg1: memref<1x2x1280xi8>, %arg2: memref<1x1280x320xi8>, %arg3: memref<1x2x320xi8>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x320xi8>
    rock.gemm %alloc = %arg1 * %arg2 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} : memref<1x2x320xi8> = memref<1x2x1280xi8> * memref<1x1280x320xi8>
    memref.copy %alloc, %arg3 : memref<1x2x320xi8> to memref<1x2x320xi8>
    return
  }
}
