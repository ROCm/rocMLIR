// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s

#transform_map0 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (2*d0 + d1)> by [<AddDim{1} ["i"] at [2] -> [] at []>, <AddDim{1} ["j"] at [3] -> [] at []>, <Unmerge{2, 2} ["ci", "cj"] at [0, 1] -> ["offset"] at [0]>] bounds = [2, 2, 1, 1] -> [4]>
// CHECK: rock_accel_gemm_wmma
func.func @rock_accel_gemm_wmma(%matrixA : memref<1x4xvector<16xf16>, 5>,
                                %matrixB : memref<1x4xvector<16xf16>, 5>,
                                %matrixC : memref<1x1xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 4]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<1x1xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<1x1xvector<8xf32>, 5>
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 4,
       mPerWave = 16,
       nPerWave = 16,
       kpack = 16,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x4xvector<16xf16>, 5> * memref<1x4xvector<16xf16>, 5>
  return
}

// CHECK: rock_accel_gemm_wmma_repeats
func.func @rock_accel_gemm_wmma_repeats(%matrixA : memref<1x4xvector<16xf16>, 5>,
                                        %matrixB : memref<1x4xvector<16xf16>, 5>,
                                        %matrixC : memref<4xvector<8xf32>, 5>) {
  %c1 = arith.constant 1 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1, 1, 4]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xf32>, 5>
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<8xf32>, 5> to memref<2x2x1x1xvector<8xf32>, 5>
  rock.threadwise_accel_gemm %matrixCView[%c1, %c1] += %matrixA * %matrixB features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kpackPerBlock = 4,
       mPerWave = 32,
       nPerWave = 32,
       kpack = 16,
       forceUnroll = true>
     } : memref<2x2x1x1xvector<8xf32>, 5> += memref<1x4xvector<16xf16>, 5> * memref<1x4xvector<16xf16>, 5>
  return
}

// CHECK: rock_accel_gemm_wmma_repeats_int8
func.func @rock_accel_gemm_wmma_repeats_int8(%matrixA : memref<1x4xvector<16xi8>, 5>,
                                             %matrixB : memref<1x4xvector<16xi8>, 5>,
                                             %matrixC : memref<4xvector<8xi32>, 5>) {
  %c1 = arith.constant 1 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1, 1, 4]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x4xvector<16xi8>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x4xvector<16xi8>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xi32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xi32>, 5>
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<8xi32>, 5> to memref<2x2x1x1xvector<8xi32>, 5>
  rock.threadwise_accel_gemm %matrixCView[%c1, %c1] += %matrixA * %matrixB features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kpackPerBlock = 4,
       mPerWave = 32,
       nPerWave = 32,
       kpack = 16,
       forceUnroll = true>
     } : memref<2x2x1x1xvector<8xi32>, 5> += memref<1x4xvector<16xi8>, 5> * memref<1x4xvector<16xi8>, 5>
  return
}

// CHECK: rock_accel_gemm_wmma_partial_repeats_int8
func.func @rock_accel_gemm_wmma_partial_repeats_int8(%matrixA : memref<1x2xvector<16xi8>, 5>,
                                                     %matrixB : memref<1x2xvector<16xi8>, 5>,
                                                     %matrixC : memref<4xvector<8xi32>, 5>) {
  %c1 = arith.constant 1 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1, 1, 2]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x2xvector<16xi8>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x2xvector<16xi8>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xi32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xi32>, 5>
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<8xi32>, 5> to memref<2x2x1x1xvector<8xi32>, 5>
  rock.threadwise_accel_gemm %matrixCView[%c1, %c1] += %matrixA * %matrixB features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kpackPerBlock = 2,
       mPerWave = 32,
       nPerWave = 16,
       kpack = 16,
       forceUnroll = true>
     } : memref<2x2x1x1xvector<8xi32>, 5> += memref<1x2xvector<16xi8>, 5> * memref<1x2xvector<16xi8>, 5>
  return
}
