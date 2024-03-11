// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (2*d0 + d1)> by [<Unmerge{1, 2} ["ci", "cj"] at [0, 1] -> ["offset"] at [0]>] bounds = [1, 2] -> [2]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1) -> (2*d0 + d1)> by [<Unmerge{2, 2} ["ci", "cj"] at [0, 1] -> ["offset"] at [0]>] bounds = [2, 2] -> [4]>

func.func @rock_accel_gemm_reduction_nokpack(%matrixA : memref<1x2xf32, 5>,
                                                 %matrixB : memref<1x2xf32, 5>,
                                                 %matrixC : memref<2xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @rock_accel_gemm_reduction_nokpack
  // CHECK-SAME: ([[ABuf:%.+]]: memref<1x2xf32, 5>, [[BBuf:%.+]]: memref<1x2xf32, 5>, [[CBuf:%.+]]: memref<2xvector<16xf32>, 5>)
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: [[a:%.+]] = memref.load [[ABuf]]
  // CHECK: [[b:%.+]] = memref.load [[BBuf]]
  // CHECK: [[c:%.+]] = memref.load [[CBuf]]
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : f32, f32, vector<16xf32>
  %c0 = arith.constant 0 : index
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<2xvector<16xf32>, 5> to memref<1x2xvector<16xf32>, 5>
  rock.threadwise_accel_gemm %matrixCView += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
       kpackPerBlock = 4,
       kpack = 1,
       mPerBlock = 128,
       mPerWave = 64,
       nPerBlock = 64,
       nPerWave = 32,
       splitKFactor = 1,
       forceUnroll = true>
     } : memref<1x2xvector<16xf32>, 5> += memref<1x2xf32, 5> * memref<1x2xf32, 5>
  return
}

func.func @rock_accel_gemm_reduction_kpack_f32(%matrixA : memref<1x2xf32, 5>,
                                                   %matrixB : memref<1x2xf32, 5>,
                                                   %matrixC : memref<4xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @rock_accel_gemm_reduction_kpack_f32
  // CHECK-SAME: ([[ABuf:%.+]]: memref<1x2xf32, 5>, [[BBuf:%.+]]: memref<1x2xf32, 5>, [[CBuf:%.+]]: memref<4xvector<16xf32>, 5>)
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: [[a:%.+]] = memref.load [[ABuf]]
  // CHECK: [[b:%.+]] = memref.load [[BBuf]]
  // CHECK: [[c:%.+]] = memref.load [[CBuf]]
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : f32, f32, vector<16xf32>
  %c0 = arith.constant 0 : index
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<16xf32>, 5> to memref<2x2xvector<16xf32>, 5>
  rock.threadwise_accel_gemm %matrixCView += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 2,
      kpack = 2,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<2x2xvector<16xf32>, 5> += memref<1x2xf32, 5> * memref<1x2xf32, 5>
  return
}

func.func @rock_accel_gemm_reduction_kpack_i8(%matrixA : memref<1x4xvector<4xi8>, 5>,
                                                 %matrixB : memref<1x4xvector<4xi8>, 5>,
                                                 %matrixC : memref<1x1xvector<16xi32>, 5>) {
  // CHECK-LABEL: func.func @rock_accel_gemm_reduction_kpack_i8
  // CHECK-SAME: ([[ABuf:%.+]]: memref<1x4xvector<4xi8>, 5>, [[BBuf:%.+]]: memref<1x4xvector<4xi8>, 5>, [[CBuf:%.+]]: memref<1x1xvector<16xi32>, 5>)
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: [[a:%.+]] = memref.load [[ABuf]]
  // CHECK: [[b:%.+]] = memref.load [[BBuf]]
  // CHECK: [[c:%.+]] = memref.load [[CBuf]]
  // CHECK: amdgpu.mfma [[a]] * [[b]] + [[c]] {{.*}} : vector<4xi8>, vector<4xi8>, vector<16xi32>
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 4,
      kpack = 8,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1x1xvector<16xi32>, 5> += memref<1x4xvector<4xi8>, 5> * memref<1x4xvector<4xi8>, 5>
  return
}

// Tests for navigating the differences between the available MFMA instructions
// on different CDNA generations.

func.func @accel_gemm_gfx90a_i8(%matrixA : memref<1x4xvector<4xi8>, 5>,
                                                 %matrixB : memref<1x4xvector<4xi8>, 5>,
                                                 %matrixC : memref<1x1xvector<16xi32>, 5>) {
  // CHECK-LABEL  func.func @accel_gemm_gfx90a_i8
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 4,
      kpack = 8,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1x1xvector<16xi32>, 5> += memref<1x4xvector<4xi8>, 5> * memref<1x4xvector<4xi8>, 5>
  return
}

func.func @accel_gemm_gfx940_i8(%matrixA : memref<1x4xvector<8xi8>, 5>,
                                                 %matrixB : memref<1x4xvector<8xi8>, 5>,
                                                 %matrixC : memref<1x1xvector<16xi32>, 5>) {
  // CHECK-LABEL: func.func @accel_gemm_gfx940_i8
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: amdgpu.mfma
  // CHECK-SAME  blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT  amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx940",
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 4,
      kpack = 16,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1x1xvector<16xi32>, 5> += memref<1x4xvector<8xi8>, 5> * memref<1x4xvector<8xi8>, 5>
  return
}

func.func @accel_gemm_gfx908_bf16(%matrixA : memref<1x4xvector<2xbf16>, 5>,
                                                 %matrixB : memref<1x4xvector<2xbf16>, 5>,
                                                 %matrixC : memref<1x1xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @accel_gemm_gfx908_bf16
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 4 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx908",
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1x1xvector<16xf32>, 5> += memref<1x4xvector<2xbf16>, 5> * memref<1x4xvector<2xbf16>, 5>
  return
}

func.func @accel_gemm_gfx90a_bf16(%matrixA : memref<1x4xvector<4xbf16>, 5>,
                                                 %matrixB : memref<1x4xvector<4xbf16>, 5>,
                                                 %matrixC : memref<1x1xvector<16xf32>, 5>) {
  // CHECK-LABEL: func.func @accel_gemm_gfx90a_bf16
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 4,
      kpack = 8,
      mPerWave = 32,
      nPerWave = 32,
      mPerBlock = 64,
      nPerBlock = 64,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<1x1xvector<16xf32>, 5> += memref<1x4xvector<4xbf16>, 5> * memref<1x4xvector<4xbf16>, 5>
  return
}

func.func @accel_gemm_fp8_bf8(%matrixA : memref<1x4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>>,
                               %matrixB : memref<1x4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>>,
                               %matrixC : memref<4xvector<16xf32>, #gpu.address_space<private>>) {
  // CHECK-LABEL: func.func @accel_gemm_fp8_bf8
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: amdgpu.mfma
  // CHECK-SAME: blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32
  // CHECK-SAME:   vector<8xf8E4M3FNUZ>, vector<8xf8E5M2FNUZ>, vector<16xf32>
  // CHECK-NOT: amdgpu.mfma
  %c0 = arith.constant 0 : index
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<16xf32>, #gpu.address_space<private>> to memref<2x2xvector<16xf32>, #gpu.address_space<private>>
  rock.threadwise_accel_gemm %matrixCView += %matrixA * %matrixB at [%c0, %c0, %c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx940",
    params = #rock.xdlops_gemm_params<
      kpackPerBlock = 8,
      mPerBlock = 128,
      nPerBlock = 128,
      kpack = 8,
      mPerWave = 64,
      nPerWave = 64,
      splitKFactor = 1,
      forceUnroll = true>
  } : memref<2x2xvector<16xf32>, #gpu.address_space<private>> += memref<1x4xvector<8xf8E4M3FNUZ>, #gpu.address_space<private>> * memref<1x4xvector<8xf8E5M2FNUZ>, #gpu.address_space<private>>
  return
}
