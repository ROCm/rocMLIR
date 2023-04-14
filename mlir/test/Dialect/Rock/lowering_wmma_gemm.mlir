// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s

// CHECK: rock_xdlops_gemm_v2_wmma
func.func @rock_xdlops_gemm_v2_wmma(%matrixA : memref<1xvector<16xf16>, 5>,
                                    %matrixB : memref<1xvector<16xf16>, 5>,
                                    %matrixC : memref<1xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1xvector<16xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1xvector<16xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<1xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<1xvector<8xf32>, 5>
  rock.xdlops_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.xdlops_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kPerBlock = 4,
       mPerWave = 16,
       nPerWave = 16,
       kpack = 16,
       forceUnroll = true>
     } : memref<1xvector<8xf32>, 5> += memref<1xvector<16xf16>, 5> * memref<1xvector<16xf16>, 5>
  return
}

// CHECK: rock_xdlops_gemm_v2_wmma_repeats
func.func @rock_xdlops_gemm_v2_wmma_repeats(%matrixA : memref<4xvector<16xf16>, 5>,
                                    %matrixB : memref<4xvector<16xf16>, 5>,
                                    %matrixC : memref<4xvector<8xf32>, 5>) {
  %c1 = arith.constant 1 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [2]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4xvector<16xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4xvector<16xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xf32>, 5>
  rock.xdlops_gemm_v2 %matrixC += %matrixA[%c1] * %matrixB[%c1] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.xdlops_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kPerBlock = 2,
       mPerWave = 32,
       nPerWave = 32,
       kpack = 16,
       forceUnroll = true>
     } : memref<4xvector<8xf32>, 5> += memref<4xvector<16xf16>, 5> * memref<4xvector<16xf16>, 5>
  return
}

// CHECK: rock_xdlops_gemm_v2_wmma_repeats_int8
func.func @rock_xdlops_gemm_v2_wmma_repeats_int8(%matrixA : memref<4xvector<16xi8>, 5>,
                                    %matrixB : memref<4xvector<16xi8>, 5>,
                                    %matrixC : memref<4xvector<8xi32>, 5>) {
  %c1 = arith.constant 1 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4xvector<16xi8>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4xvector<16xi8>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xi32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xi32>, 5>
  rock.xdlops_gemm_v2 %matrixC += %matrixA[%c1] * %matrixB[%c1] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.xdlops_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kPerBlock = 4,
       mPerWave = 32,
       nPerWave = 32,
       kpack = 16,
       forceUnroll = true>
     } : memref<4xvector<8xi32>, 5> += memref<4xvector<16xi8>, 5> * memref<4xvector<16xi8>, 5>
  return
}

// CHECK: rock_xdlops_gemm_v2_wmma_partial_repeats_int8
func.func @rock_xdlops_gemm_v2_wmma_partial_repeats_int8(%matrixA : memref<2xvector<16xi8>, 5>,
                                    %matrixB : memref<2xvector<16xi8>, 5>,
                                    %matrixC : memref<2xvector<8xi32>, 5>) {
  %c1 = arith.constant 1 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [2]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<2xvector<16xi8>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<2xvector<16xi8>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<2xvector<8xi32>, 5>
  // CHECK: amdgpu.wmma %[[a]] * %[[b]] + %[[c]]
  // CHECK: memref.store {{.*}}, {{.*}} : memref<2xvector<8xi32>, 5>
  rock.xdlops_gemm_v2 %matrixC += %matrixA[%c1] * %matrixB[%c1] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.xdlops_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kPerBlock = 2,
       mPerWave = 32,
       nPerWave = 16,
       kpack = 16,
       forceUnroll = true>
     } : memref<2xvector<8xi32>, 5> += memref<2xvector<16xi8>, 5> * memref<2xvector<16xi8>, 5>
  return
}
