// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s

// CHECK: rock_accel_gemm_fma
func.func @rock_accel_gemm_fma(%dummy: memref<1x2xvector<32xf16>>) attributes {block_size = 64 : i32} {
  %c0 = arith.constant 0 : index
  // CHECK: %[[cReg:.*]] = rock.alloc() : memref<1xvector<1xf32>, #gpu.address_space<private>>
  // CHECK: rock.transforming_for
  // CHECK-SAME: bounds [1, 1, 1]
  // CHECK-DAG: %[[a:.*]] = memref.load {{.*}} : memref<2xvector<32xf16>, #gpu.address_space<private>>
  // CHECK-DAG: %[[b:.*]] = memref.load {{.*}} : memref<2xvector<32xf16>, #gpu.address_space<private>>
  // CHECK-DAG: %[[c:.*]] = memref.load {{.*}} : memref<1xvector<1xf32>, #gpu.address_space<private>>
  // CHECK-DAG: %[[aF32:.*]] = arith.extf %[[a]] : vector<32xf16> to vector<32xf32>
  // CHECK-DAG: %[[bF32:.*]] = arith.extf %[[b]] : vector<32xf16> to vector<32xf32>
  // CHECK-DAG: %[[mul:.*]] = arith.mulf %[[aF32]], %[[bF32]] : vector<32xf32>
  // CHECK-DAG: %[[localDot:.*]] = vector.reduction <add>, %[[mul]] : vector<32xf32> into f32
  // CHECK-DAG: %[[numThreads:.*]] = arith.constant 64 : i32
  // CHECK-DAG: %[[firstShuffleVar:.*]] = arith.constant 1 : i32
  // CHECK: %[[firstShuffleRes:.*]], %{{.*}} = gpu.shuffle  xor %[[localDot]], %[[firstShuffleVar]], %[[numThreads]] : f32
  // CHECK: %[[firstRes:.*]] = arith.addf %[[localDot]], %[[firstShuffleRes]] : f32
  // CHECK-DAG: %[[secondShuffleVar:.*]] = arith.constant 2 : i32
  // CHECK: %[[secondShuffleRes:.*]], %{{.*}} = gpu.shuffle  xor %[[firstRes]], %[[secondShuffleVar]], %[[numThreads]] : f32
  // CHECK: %[[secondRes:.*]] = arith.addf %[[firstRes]], %[[secondShuffleRes]] : f32
  // CHECK-DAG: %[[thirdShuffleVar:.*]] = arith.constant 4 : i32
  // CHECK: %[[thirdShuffleRes:.*]], %{{.*}} = gpu.shuffle  xor %[[secondRes]], %[[thirdShuffleVar]], %[[numThreads]] : f32
  // CHECK: %[[thirdRes:.*]] = arith.addf %[[secondRes]], %[[thirdShuffleRes]] : f32
  // CHECK-DAG: %[[fourthShuffleVar:.*]] = arith.constant 8 : i32
  // CHECK: %[[fourthShuffleRes:.*]], %{{.*}} = gpu.shuffle  xor %[[thirdRes]], %[[fourthShuffleVar]], %[[numThreads]] : f32
  // CHECK: %[[fourthRes:.*]] = arith.addf %[[thirdRes]], %[[fourthShuffleRes]] : f32
  // CHECK-DAG: %[[fifthShuffleVar:.*]] = arith.constant 16 : i32
  // CHECK: %[[fifthShuffleRes:.*]], %{{.*}} = gpu.shuffle  xor %[[fourthRes]], %[[fifthShuffleVar]], %[[numThreads]] : f32
  // CHECK: %[[fifthRes:.*]] = arith.addf %[[fourthRes]], %[[fifthShuffleRes]] : f32
  // CHECK-DAG: %[[sixthShuffleVar:.*]] = arith.constant 32 : i32
  // CHECK: %[[sixthShuffleRes:.*]], %{{.*}} = gpu.shuffle  xor %[[fifthRes]], %[[sixthShuffleVar]], %[[numThreads]] : f32
  // CHECK: %[[finalShuffleRes:.*]] = arith.addf %[[fifthRes]], %[[sixthShuffleRes]] : f32
  // CHECK: scf.if %{{.*}}
  // CHECK-NEXT: %[[finalShuffleResBroadcasted:.*]] = vector.broadcast %[[finalShuffleRes]] : f32 to vector<1xf32>
  // CHECK-NEXT: %[[finalRes:.*]] = arith.addf %[[c]], %[[finalShuffleResBroadcasted]] : vector<1xf32>
  // CHECK-NEXT: memref.store %[[finalRes]], %[[cReg]][%{{.*}}] : memref<1xvector<1xf32>, #gpu.address_space<private>>
  %40 = rock.alloc() : memref<2xvector<32xf16>, #gpu.address_space<private>>
  %41 = rock.alloc() : memref<2xvector<32xf16>, #gpu.address_space<private>>
  %42 = rock.alloc() : memref<1xvector<1xf32>, #gpu.address_space<private>>
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  affine.for %arg3 = 0 to 1 {
    memref.store %cst, %42[%arg3] : memref<1xvector<1xf32>, #gpu.address_space<private>>
  }
  
  %74 = rock.transform %40 by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["i"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 2] -> [2]> : memref<2xvector<32xf16>, #gpu.address_space<private>> to memref<1x2xvector<32xf16>, #gpu.address_space<private>>
  %75 = rock.transform %41 by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["j"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 2] -> [2]> : memref<2xvector<32xf16>, #gpu.address_space<private>> to memref<1x2xvector<32xf16>, #gpu.address_space<private>>
  %76 = rock.transform %42 by <affine_map<(d0, d1) -> (d0 + d1)> by [<Unmerge{1, 1} ["i", "j"] at [0, 1] -> ["offset"] at [0]>] bounds = [1, 1] -> [1]> : memref<1xvector<1xf32>, #gpu.address_space<private>> to memref<1x1xvector<1xf32>, #gpu.address_space<private>>
  rock.threadwise_accel_gemm %76 += %74 * %75 at[%c0, %c0, %c0] features =  dot|atomic_add|atomic_add_f16 {arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", params = #rock.fma_gemm_params<blockSize = 64, mPerBlock = 1, nPerBlock = 1, kpackPerBlock = 2, kpack = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x1xvector<1xf32>, #gpu.address_space<private>> += memref<1x2xvector<32xf16>, #gpu.address_space<private>> * memref<1x2xvector<32xf16>, #gpu.address_space<private>>
  return
}
