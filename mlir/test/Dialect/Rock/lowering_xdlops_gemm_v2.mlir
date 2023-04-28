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
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : f32, f32, vector<16xf32>
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
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
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : f32, f32, vector<16xf32>
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
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
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : vector<4xi8>, vector<4xi8>, vector<16xi32>
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
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

/// Tests for navigating the differences between the available MFMA instructions
/// on different CDNA generations.

func.func @xdlops_gemm_gfx90a_i8(%matrixA : memref<4xvector<4xi8>, 5>,
                                                 %matrixB : memref<4xvector<4xi8>, 5>,
                                                 %matrixC : memref<1xvector<16xi32>, 5>) {
  // CHECK-LABEL: func.func @xdlops_gemm_gfx90a_i8
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
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

func.func @xdlops_gemm_gfx940_i8(%matrixA : memref<4xvector<8xi8>, 5>,
                                                 %matrixB : memref<4xvector<8xi8>, 5>,
                                                 %matrixC : memref<1xvector<16xi32>, 5>) {
  // CHECK-LABEL: func.func @xdlops_gemm_gfx940_i8
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx940",
    params = #rock.xdlops_gemm_params<
      kPerBlock = 4,
      kpack = 16,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      forceUnroll = true>
  } : memref<1xvector<16xi32>, 5> += memref<4xvector<8xi8>, 5> * memref<4xvector<8xi8>, 5>
  return
}

func.func @xdlops_gemm_gfx908_bf16(%matrixA : memref<4xvector<2xbf16>, 5>,
                                                 %matrixB : memref<4xvector<2xbf16>, 5>,
                                                 %matrixC : memref<1xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @xdlops_gemm_gfx908_bf16
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 4 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx908",
    params = #rock.xdlops_gemm_params<
      kPerBlock = 4,
      kpack = 4,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      forceUnroll = true>
  } : memref<1xvector<16xf32>, 5> += memref<4xvector<2xbf16>, 5> * memref<4xvector<2xbf16>, 5>
  return
}

func.func @xdlops_gemm_gfx90a_bf16(%matrixA : memref<4xvector<4xbf16>, 5>,
                                                 %matrixB : memref<4xvector<4xbf16>, 5>,
                                                 %matrixC : memref<1xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @xdlops_gemm_gfx90a_bf16
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      kPerBlock = 4,
      kpack = 8,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      forceUnroll = true>
  } : memref<1xvector<16xf32>, 5> += memref<4xvector<4xbf16>, 5> * memref<4xvector<4xbf16>, 5>
  return
}

func.func @xdlops_gemm_fp8_bf8(%matrixA : memref<4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>>,
                               %matrixB : memref<4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>>,
                               %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  // CHECK-LABEL: func.func @xdlops_gemm_fp8_bf8
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [4]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-SAME: : vector<8xf8E4M3FNUZ>, vector<8xf8E5M2FNUZ>, vector<16xf32>
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.accel_gemm_v2 %matrixC += %matrixA[%c0] * %matrixB[%c0] {
    arch = "amdgcn-amd-amdhsa:gfx940",
    params = #rock.xdlops_gemm_params<
      kPerBlock = 8,
      mPerBlock = 128,
      nPerBlock = 128,
      kpack = 8,
      mPerWave = 64,
      nPerWave = 64,
      forceUnroll = true>
  } : memref<4xvector<16xf32>, #gpu.address_space<private>> += memref<4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>> * memref<4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>>
  return
}
