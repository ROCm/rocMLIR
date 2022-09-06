// Ensures that the padding application, group application, etc. in gemm-to-gridwise
// function as expected.

// Note: numCu values are made up

// RUN: miopen-opt -miopen-gemm-to-gridwise %s | FileCheck %s

#general_gemm_params0 = #miopen.general_gemm_params<kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, mThreadsPerCuwave = 4, nThreadsPerCuwave = 4, mCuwavesPerBlock = 4, nCuwavesPerBlock = 4>
#general_gemm_params1 = #miopen.general_gemm_params<kPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, mThreadsPerCuwave = 4, nThreadsPerCuwave = 4, mCuwavesPerBlock = 2, nCuwavesPerBlock = 2>
#xdlops_gemm_params0 = #miopen.xdlops_gemm_params<kPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32>
#xdlops_gemm_params1 = #miopen.xdlops_gemm_params<kPerBlock = 4, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 64, nPerWave = 64>

// CHECK-LABEL: func.func @gemm_easy_case_from_conv
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
func.func @gemm_easy_case_from_conv(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK-NEXT: miopen.gridwise_gemm %[[c]] = %[[a]] * %[[b]]
  miopen.gemm %c = tr %a * %b features = none storeMethod = set {
    arch = "gfx906",
    blockSize = 256 : i32,
    gridSize = 4 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_easy_case_from_conv_xdlops
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
func.func @gemm_easy_case_from_conv_xdlops(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK-NEXT: miopen.gridwise_gemm_v2(%[[a]], %[[b]], %[[c]])
  miopen.gemm %c = tr %a * %b features = mfma|dot|atomic_add storeMethod = set {
    arch = "gfx908",
    blockSize = 256 : i32,
    gridSize = 4 : i32,
    numCu = 64 : i32,
    params = #xdlops_gemm_params0
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_most_general_padding_case
// CHECK-SAME: (%[[a:.*]]: memref<1x1x1xf32>, %[[b:.*]]: memref<1x1x1xf32>, %[[c:.*]]: memref<1x1x1xf32>)
func.func @gemm_most_general_padding_case(%a: memref<1x1x1xf32>, %b: memref<1x1x1xf32>, %c: memref<1x1x1xf32>) {
  // CHECK-DAG: %[[padA:.*]] = miopen.transform %[[a]] by {{.*}} : memref<1x1x1xf32> to memref<1x16x64xf32{{.*}}>
  // CHECK-DAG: %[[padB:.*]] = miopen.transform %[[b]] by {{.*}} : memref<1x1x1xf32> to memref<1x16x64xf32{{.*}}>
  // CHECK-DAG: %[[padC:.*]] = miopen.transform %[[c]] by {{.*}} : memref<1x1x1xf32> to memref<1x64x64xf32{{.*}}>
  // CHECK: miopen.gridwise_gemm %[[padC]] = %[[padA]] * %[[padB]]
  miopen.gemm %c = tr %a * %b features = none storeMethod = set {
    arch = "gfx906",
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params1
  } : memref<1x1x1xf32> = memref<1x1x1xf32> * memref<1x1x1xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_in_standard_form
// CHECK-SAME: (%[[a:.*]]: memref<128x72xf32>, %[[b:.*]]: memref<72x512xf32>, %[[c:.*]]: memref<128x512xf32>)
func.func @gemm_in_standard_form(%a: memref<128x72xf32>, %b: memref<72x512xf32>, %c: memref<128x512xf32>) {
  // CHECK-DAG: %[[normalizeA:.*]] = miopen.transform %[[a]] by {{.*}} : memref<128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = miopen.transform %[[b]] by {{.*}} : memref<72x512xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = miopen.transform %[[c]] by {{.*}} : memref<128x512xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: miopen.gridwise_gemm %[[normalizeC]] = %[[normalizeA]] * %[[normalizeB]]
  miopen.gemm %c = %a * %b features = none storeMethod = set {
    arch = "gfx906",
    blockSize = 256 : i32,
    gridSize = 4 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0
  } : memref<128x512xf32> = memref<128x72xf32> * memref<72x512xf32>
  func.return
}

// CHECK-LABEL: func.func @gemm_transposed_from_gridwise
// CHECK-SAME: (%[[a:.*]]: memref<1x128x72xf32>, %[[b:.*]]: memref<1x512x72xf32>, %[[c:.*]]: memref<1x512x128xf32>)
func.func @gemm_transposed_from_gridwise(%a: memref<1x128x72xf32>, %b: memref<1x512x72xf32>, %c: memref<1x512x128xf32>) {
  // CHECK-DAG: %[[normalizeA:.*]] = miopen.transform %[[a]] {{.*}} : memref<1x128x72xf32> to memref<1x72x128xf32{{.*}}>
  // CHECK-DAG: %[[normalizeB:.*]] = miopen.transform %[[b]] {{.*}} : memref<1x512x72xf32> to memref<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = miopen.transform %[[c]] {{.*}} : memref<1x512x128xf32> to memref<1x128x512xf32{{.*}}>
  // CHECK: miopen.gridwise_gemm %[[normalizeC]] = %[[normalizeA]] * %[[normalizeB]]
  miopen.gemm tr %c = %a * tr %b features = none storeMethod = set {
    arch = "gfx906",
    blockSize = 256 : i32,
    gridSize = 4 : i32,
    numCu = 64 : i32,
    params = #general_gemm_params0
  } : memref<1x512x128xf32> = memref<1x128x72xf32> * memref<1x512x72xf32>
  func.return
}

// kPack shouldn't be applied this early, but the xdlops gemm implementation currently
// expects that. Remove this test in the future
// CHECK-LABEL: func.func @gemm_kpack
// CHECK-SAME: (%[[a:.*]]: memref<1x72x128xf32>, %[[b:.*]]: memref<1x72x512xf32>, %[[c:.*]]: memref<1x128x512xf32>)
func.func @gemm_kpack(%a: memref<1x72x128xf32>, %b: memref<1x72x512xf32>, %c: memref<1x128x512xf32>) {
  // CHECK-DAG: %[[kpackA:.*]] = miopen.transform %[[a]] by {{.*}} : memref<1x72x128xf32> to memref<1x18x128x4xf32{{.*}}>
  // CHECK-DAG: %[[kpackB:.*]] = miopen.transform %[[b]] by {{.*}} : memref<1x72x512xf32> to memref<1x18x512x4xf32{{.*}}>
  // CHECK-NEXT: miopen.gridwise_gemm_v2(%[[kpackA]], %[[kpackB]], %[[c]])
  miopen.gemm %c = tr %a * %b features = mfma|dot|atomic_add storeMethod = set {
    arch = "gfx908",
    blockSize = 256 : i32,
    gridSize = 4 : i32,
    numCu = 64 : i32,
    params = #xdlops_gemm_params1
  } : memref<1x128x512xf32> = memref<1x72x128xf32> * memref<1x72x512xf32>
  func.return
}
