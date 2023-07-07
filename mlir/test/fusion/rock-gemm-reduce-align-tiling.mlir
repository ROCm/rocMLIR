// RUN: rocmlir-opt --rock-view-to-transform -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-linalg-align %s | FileCheck %s

// CHECK: [[MAP0:.*]] = #rock.transform_map<{{.*}} by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 128, 256] -> [1, 128, 1]>
// CHECK: [[MAP1:.*]] = #rock.transform_map<{{.*}} by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 128, 256] -> [1, 1, 256]>

// CHECK: test_gemm_reduce_last_axis_fusion
func.func @test_gemm_reduce_last_axis_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: %[[trOut:.*]] = rock.transform %arg2 by [[MAP0]] : memref<1x128x1xf32> to memref<1x128x256xf32>
  // CHECK: rock.threadwise_write_all {{.*}}(%[[trOut]]){{.*}} by  atomic_add : {{.*}} -> memref<1x128x256xf32>
  rock.reduce sum %0 into %arg2 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}


// CHECK: test_gemm_reduce_middle_axis_fusion
func.func @test_gemm_reduce_middle_axis_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x1x256xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: %[[trOut:.*]] = rock.transform %arg2 by [[MAP1]] : memref<1x1x256xf32> to memref<1x128x256xf32>
  // CHECK: rock.threadwise_write_all {{.*}}(%[[trOut]]){{.*}} by  atomic_add : {{.*}} -> memref<1x128x256xf32>
  rock.reduce sum %0 into %arg2 features = mfma|dot|atomic_add {axis = 1 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x1x256xf32>
  return
}

// CHECK: test_gemm_add_reduce_fusion
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @test_gemm_add_reduce_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x256xf32>, %arg3: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  %1 = memref.alloc() : memref<1x128x256xf32>
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg2 : memref<1x128x256xf32>, memref<1x128x256xf32>) outs(%1 : memref<1x128x256xf32>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %4 = arith.addf %arg4, %arg5 : f32
    linalg.yield %4 : f32
  }
  // CHECK: %[[trOut:.*]] = rock.transform %arg3 by [[MAP0]] : memref<1x128x1xf32> to memref<1x128x256xf32>
  // CHECK: rock.threadwise_write_all {{.*}}(%[[trOut]]){{.*}} by  atomic_add : {{.*}} -> memref<1x128x256xf32>
  rock.reduce sum %1 into %arg3 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}

// CHECK: test_gemm_reduce_max
func.func @test_gemm_reduce_max(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: %[[trOut:.*]] = rock.transform %arg2 by [[MAP0]] : memref<1x128x1xf32> to memref<1x128x256xf32>
  // CHECK: rock.threadwise_write_all {{.*}}(%[[trOut]]){{.*}} by  atomic_max : {{.*}} -> memref<1x128x256xf32>
  rock.reduce max %0 into %arg2 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}
