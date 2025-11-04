// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (2*d0 + d1)> by [<Unmerge{2, 2} ["i", "j"] at [0, 1] -> ["offset"] at [0]>] bounds = [2, 2] -> [4]>
// CHECK: rock_accel_gemm_wmma
func.func @rock_accel_gemm_wmma(%matrixA : memref<1x4xvector<16xf16>, 5>,
                                %matrixB : memref<1x4xvector<16xf16>, 5>,
                                %matrixC : memref<1x1xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<1x1xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x16 %[[a]] * %[[b]] + %[[c]] : vector<16xf16>, vector<16xf16>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<1x1xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 4,
       mPerWave = 16,
       nPerWave = 16,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 3, 
       scheduleVersion = 1, 
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x4xvector<16xf16>, 5> * memref<1x4xvector<16xf16>, 5>
  return
}

// CHECK: rock_accel_gemm_wmma_gfx12
func.func @rock_accel_gemm_wmma_gfx12(%matrixA : memref<1x4xvector<8xf16>, 5>,
                                      %matrixB : memref<1x4xvector<8xf16>, 5>,
                                      %matrixC : memref<1x1xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME  bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x4xvector<8xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x4xvector<8xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<1x1xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x16 %[[a]] * %[[b]] + %[[c]] : vector<8xf16>, vector<8xf16>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<1x1xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1200",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 4,
       mPerWave = 16,
       nPerWave = 16,
       mnPerXdl = 16,
       kpack = 8,
       splitKFactor = 3, 
       scheduleVersion = 1, 
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x4xvector<8xf16>, 5> * memref<1x4xvector<8xf16>, 5>
  return
}

// CHECK: rock_accel_gemm_wmma_repeats
func.func @rock_accel_gemm_wmma_repeats(%matrixA : memref<1x4xvector<16xf16>, 5>,
                                        %matrixB : memref<1x4xvector<16xf16>, 5>,
                                        %matrixC : memref<4xvector<8xf32>, 5>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x4xvector<16xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x16 %[[a]] * %[[b]] + %[[c]] : vector<16xf16>, vector<16xf16>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xf32>, 5>
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<8xf32>, 5> to memref<2x2xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixCView += %matrixA * %matrixB at [%c1, %c1, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kpackPerBlock = 4,
       mPerWave = 32,
       nPerWave = 32,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 3, 
       scheduleVersion = 1, 
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<2x2xvector<8xf32>, 5> += memref<1x4xvector<16xf16>, 5> * memref<1x4xvector<16xf16>, 5>
  return
}

// CHECK: rock_accel_gemm_wmma_repeats_int8
func.func @rock_accel_gemm_wmma_repeats_int8(%matrixA : memref<1x4xvector<16xi8>, 5>,
                                             %matrixB : memref<1x4xvector<16xi8>, 5>,
                                             %matrixC : memref<4xvector<8xi32>, 5>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x4xvector<16xi8>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x4xvector<16xi8>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xi32>, 5>
  // CHECK: amdgpu.wmma 16x16x16 %[[a]] * %[[b]] + %[[c]] {clamp} : vector<16xi8>, vector<16xi8>, vector<8xi32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xi32>, 5>
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<8xi32>, 5> to memref<2x2xvector<8xi32>, 5>
  rock.threadwise_gemm_accel %matrixCView += %matrixA * %matrixB at [%c1, %c1, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kpackPerBlock = 4,
       mPerWave = 32,
       nPerWave = 32,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 3, 
       scheduleVersion = 1, 
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<2x2xvector<8xi32>, 5> += memref<1x4xvector<16xi8>, 5> * memref<1x4xvector<16xi8>, 5>
  return
}

// CHECK: rock_accel_gemm_wmma_partial_repeats_int8
func.func @rock_accel_gemm_wmma_partial_repeats_int8(%matrixA : memref<1x2xvector<16xi8>, 5>,
                                                     %matrixB : memref<1x2xvector<16xi8>, 5>,
                                                     %matrixC : memref<4xvector<8xi32>, 5>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 1 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x2xvector<16xi8>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x2xvector<16xi8>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4xvector<8xi32>, 5>
  // CHECK: amdgpu.wmma 16x16x16 %[[a]] * %[[b]] + %[[c]] {clamp} : vector<16xi8>, vector<16xi8>, vector<8xi32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4xvector<8xi32>, 5>
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<8xi32>, 5> to memref<2x2xvector<8xi32>, 5>
  rock.threadwise_gemm_accel %matrixCView += %matrixA * %matrixB at [%c1, %c1, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1100",
    params = #rock.wmma_gemm_params<
       mPerBlock = 32,
       nPerBlock = 32,
       kpackPerBlock = 2,
       mPerWave = 32,
       nPerWave = 16,
       mnPerXdl = 16,
       kpack = 16,
       splitKFactor = 3, 
       scheduleVersion = 1, 
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<2x2xvector<8xi32>, 5> += memref<1x2xvector<16xi8>, 5> * memref<1x2xvector<16xi8>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_f32
func.func @rock_accel_gemm_wmma_gfx1250_f32(%matrixA : memref<1x1xvector<2xf32>, 5>,
                                             %matrixB : memref<1x1xvector<2xf32>, 5>,
                                             %matrixC : memref<1x1xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x1xvector<2xf32>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x1xvector<2xf32>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<1x1xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x4 %[[a]] * %[[b]] + %[[c]] : vector<2xf32>, vector<2xf32>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<1x1xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 1,
       mPerWave = 16,
       nPerWave = 16,
       kpack = 4,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x1xvector<2xf32>, 5> * memref<1x1xvector<2xf32>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_f16_k32
func.func @rock_accel_gemm_wmma_gfx1250_f16_k32(%matrixA : memref<1x2xvector<16xf16>, 5>,
                                                 %matrixB : memref<1x2xvector<16xf16>, 5>,
                                                 %matrixC : memref<1x1xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x2xvector<16xf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x2xvector<16xf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<1x1xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x32 %[[a]] * %[[b]] + %[[c]] : vector<16xf16>, vector<16xf16>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<1x1xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 2,
       mPerWave = 16,
       nPerWave = 16,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x2xvector<16xf16>, 5> * memref<1x2xvector<16xf16>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_bf16_k32
func.func @rock_accel_gemm_wmma_gfx1250_bf16_k32(%matrixA : memref<1x2xvector<16xbf16>, 5>,
                                                  %matrixB : memref<1x2xvector<16xbf16>, 5>,
                                                  %matrixC : memref<1x1xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<1x2xvector<16xbf16>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<1x2xvector<16xbf16>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<1x1xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x32 %[[a]] * %[[b]] + %[[c]] : vector<16xbf16>, vector<16xbf16>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<1x1xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 16,
       nPerBlock = 16,
       kpackPerBlock = 2,
       mPerWave = 16,
       nPerWave = 16,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<1x1xvector<8xf32>, 5> += memref<1x2xvector<16xbf16>, 5> * memref<1x2xvector<16xbf16>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_fp8_fp8_k64
func.func @rock_accel_gemm_wmma_gfx1250_fp8_fp8_k64(%matrixA : memref<4x4xvector<64xf8E4M3FN>, 5>,
                                                    %matrixB : memref<4x4xvector<64xf8E4M3FN>, 5>,
                                                    %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 4,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x4xvector<64xf8E4M3FN>, 5> * memref<4x4xvector<64xf8E4M3FN>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_fp8_bf8_k64
func.func @rock_accel_gemm_wmma_gfx1250_fp8_bf8_k64(%matrixA : memref<4x4xvector<64xf8E4M3FN>, 5>,
                                                    %matrixB : memref<4x4xvector<64xf8E5M2>, 5>,
                                                    %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E5M2>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 4,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x4xvector<64xf8E4M3FN>, 5> * memref<4x4xvector<64xf8E5M2>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_bf8_fp8_k64
func.func @rock_accel_gemm_wmma_gfx1250_bf8_fp8_k64(%matrixA : memref<4x4xvector<64xf8E5M2>, 5>,
                                                    %matrixB : memref<4x4xvector<64xf8E4M3FN>, 5>,
                                                    %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E5M2>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 4,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x4xvector<64xf8E5M2>, 5> * memref<4x4xvector<64xf8E4M3FN>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_bf8_bf8_k64
func.func @rock_accel_gemm_wmma_gfx1250_bf8_bf8_k64(%matrixA : memref<4x4xvector<64xf8E5M2>, 5>,
                                                    %matrixB : memref<4x4xvector<64xf8E5M2>, 5>,
                                                    %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E5M2>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x4xvector<64xf8E5M2>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E5M2>, vector<64xf8E5M2>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 4,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x4xvector<64xf8E5M2>, 5> * memref<4x4xvector<64xf8E5M2>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_fp8_fp8_k128
func.func @rock_accel_gemm_wmma_gfx1250_fp8_fp8_k128(%matrixA : memref<4x8xvector<64xf8E4M3FN>, 5>,
                                                     %matrixB : memref<4x8xvector<64xf8E4M3FN>, 5>,
                                                     %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E4M3FN>, vector<64xf8E4M3FN>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf8E4M3FN>, 5> * memref<4x8xvector<64xf8E4M3FN>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_fp8_bf8_k128
func.func @rock_accel_gemm_wmma_gfx1250_fp8_bf8_k128(%matrixA : memref<4x8xvector<64xf8E4M3FN>, 5>,
                                                     %matrixB : memref<4x8xvector<64xf8E5M2>, 5>,
                                                     %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E5M2>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf8E4M3FN>, 5> * memref<4x8xvector<64xf8E5M2>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_bf8_fp8_k128
func.func @rock_accel_gemm_wmma_gfx1250_bf8_fp8_k128(%matrixA : memref<4x8xvector<64xf8E5M2>, 5>,
                                                     %matrixB : memref<4x8xvector<64xf8E4M3FN>, 5>,
                                                     %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E5M2>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E5M2>, vector<64xf8E4M3FN>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf8E5M2>, 5> * memref<4x8xvector<64xf8E4M3FN>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_bf8_bf8_k128
func.func @rock_accel_gemm_wmma_gfx1250_bf8_bf8_k128(%matrixA : memref<4x8xvector<64xf8E5M2>, 5>,
                                                     %matrixB : memref<4x8xvector<64xf8E5M2>, 5>,
                                                     %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E5M2>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E5M2>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E5M2>, vector<64xf8E5M2>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf8E5M2>, 5> * memref<4x8xvector<64xf8E5M2>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_i8_k64
func.func @rock_accel_gemm_wmma_gfx1250_i8_k64(%matrixA : memref<4x4xvector<32xi8>, 5>,
                                               %matrixB : memref<4x4xvector<32xi8>, 5>,
                                               %matrixC : memref<4x4xvector<8xi32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x4xvector<32xi8>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x4xvector<32xi8>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xi32>, 5>
  // CHECK: amdgpu.wmma 16x16x64 %[[a]] * %[[b]] + %[[c]] {clamp} : vector<32xi8>, vector<32xi8>, vector<8xi32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xi32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 4,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xi32>, 5> += memref<4x4xvector<32xi8>, 5> * memref<4x4xvector<32xi8>, 5>
  return
}

// CHECK-LABEL: @rock_accel_gemm_wmma_gfx1250_k_selection
func.func @rock_accel_gemm_wmma_gfx1250_k_selection(%matrixA : memref<4x8xvector<64xf8E4M3FN>, 5>,
                                                    %matrixB : memref<4x8xvector<64xf8E5M2>, 5>,
                                                    %matrixC : memref<4x4xvector<8xf32>, 5>) {
  %c0 = arith.constant 0 : index
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK: %[[a:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E4M3FN>, 5>
  // CHECK: %[[b:.*]] = memref.load {{.*}} : memref<4x8xvector<64xf8E5M2>, 5>
  // CHECK: %[[c:.*]] = memref.load {{.*}} : memref<4x4xvector<8xf32>, 5>
  // CHECK: amdgpu.wmma 16x16x128 %[[a]] * %[[b]] + %[[c]] : vector<64xf8E4M3FN>, vector<64xf8E5M2>, vector<8xf32>
  // CHECK: memref.store {{.*}}, {{.*}} : memref<4x4xvector<8xf32>, 5>
  rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] features = wmma {
    arch = "amdgcn-amd-amdhsa:gfx1250",
    params = #rock.wmma_gemm_params<
       mPerBlock = 64,
       nPerBlock = 64,
       kpackPerBlock = 8,
       mPerWave = 64,
       nPerWave = 64,
       kpack = 16,
       splitKFactor = 1,
       scheduleVersion = 1,
       outputSwizzle = 2,
       forceUnroll = true>
     } : memref<4x4xvector<8xf32>, 5> += memref<4x8xvector<64xf8E4M3FN>, 5> * memref<4x8xvector<64xf8E5M2>, 5>
  return
}
