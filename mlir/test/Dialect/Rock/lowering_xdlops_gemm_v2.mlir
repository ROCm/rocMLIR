// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s

func.func @rock_xdlops_gemm_v2_reduction_nokpack(%matrixA : memref<2xf32, 5>,
                                                 %matrixB : memref<2xf32, 5>,
                                                 %matrixC : memref<2xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @rock_xdlops_gemm_v2_reduction_nokpack
  // CHECK-SAME: ([[ABuf:%.+]]: memref<2xf32, 5>, [[BBuf:%.+]]: memref<2xf32, 5>, [[CBuf:%.+]]: memref<2xvector<16xf32>, 5>)
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [2]
  // CHECK: [[a:%.+]] = memref.load [[ABuf]]
  // CHECK: [[b:%.+]] = memref.load [[BBuf]]
  // CHECK: [[c:%.+]] = memref.load [[CBuf]]
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : f32, vector<16xf32>
  %c0 = arith.constant 0 : index
  rock.xdlops_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    params = #rock.xdlops_gemm_params<
       kPerBlock = 4,
       kpack = 1,
       mPerBlock = 128,
       mPerWave = 64,
       nPerBlock = 64,
       nPerWave = 32,
       forceUnroll = true>
     } : memref<2xvector<16xf32>, 5> += memref<2xf32, 5> * memref<2xf32, 5>
  return
}

func.func @rock_xdlops_gemm_v2_reduction_kpack_f32(%matrixA : memref<2xf32, 5>,
                                                   %matrixB : memref<2xf32, 5>,
                                                   %matrixC : memref<4xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @rock_xdlops_gemm_v2_reduction_kpack_f32
  // CHECK-SAME: ([[ABuf:%.+]]: memref<2xf32, 5>, [[BBuf:%.+]]: memref<2xf32, 5>, [[CBuf:%.+]]: memref<4xvector<16xf32>, 5>)
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [2]
  // CHECK: [[a:%.+]] = memref.load [[ABuf]]
  // CHECK: [[b:%.+]] = memref.load [[BBuf]]
  // CHECK: [[c:%.+]] = memref.load [[CBuf]]
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : f32, vector<16xf32>
  %c0 = arith.constant 0 : index
  rock.xdlops_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    params = #rock.xdlops_gemm_params<
      kPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, 5> += memref<2xf32, 5> * memref<2xf32, 5>
  return
}

func.func @rock_xdlops_gemm_v2_reduction_kpack_i8(%matrixA : memref<4xvector<4xi8>, 5>,
                                                 %matrixB : memref<4xvector<4xi8>, 5>,
                                                 %matrixC : memref<1xvector<16xi32>, 5>) {
  // CHECK-LABEL: func.func @rock_xdlops_gemm_v2_reduction_kpack_i8
  // CHECK-SAME: ([[ABuf:%.+]]: memref<4xvector<4xi8>, 5>, [[BBuf:%.+]]: memref<4xvector<4xi8>, 5>, [[CBuf:%.+]]: memref<1xvector<16xi32>, 5>)
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: [[a:%.+]] = memref.load [[ABuf]]
  // CHECK: [[b:%.+]] = memref.load [[BBuf]]
  // CHECK: [[c:%.+]] = memref.load [[CBuf]]
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : vector<4xi8>, vector<16xi32>
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.xdlops_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    params = #rock.xdlops_gemm_params<
      kPerBlock = 4,
      kpack = 8,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      forceUnroll = true>
  } : memref<1xvector<16xi32>, 5> += memref<4xvector<4xi8>, 5> * memref<4xvector<4xi8>, 5>
  return
}
