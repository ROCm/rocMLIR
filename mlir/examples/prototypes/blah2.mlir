#map0 = (d_gemmK, d_gemmM) -> (d_gemmM, d_gemmK floordiv 9, (d_gemmK mod 9) floordiv 3, (d_gemmK mod 9) mod 3)


module {
  // coordinates are all known at compile-time
  func @affine_load_constant(%arg0: memref<128x8x3x3xf32>) {
    %d_gemmK = constant 3 : index
    %d_gemmM = constant 4 : index
    %0 = affine.load %arg0[%d_gemmM, %d_gemmK floordiv 9, (%d_gemmK mod 9) floordiv 3, (%d_gemmK mod 9) mod 3] : memref<128x8x3x3xf32>
    return
  }

  // 1 coordinate is given at run-time
  func @affine_load_dynamic_gemmK(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index) {
    %d_gemmM = constant 0 : index
    %0 = affine.load %arg0[%d_gemmM, %d_gemmK floordiv 9, (%d_gemmK mod 9) floordiv 3, (%d_gemmK mod 9) mod 3] : memref<128x8x3x3xf32>
    return
  }

  // 1 coordinate is given at run-time
  func @affine_load_dynamic_gemmM(%arg0: memref<128x8x3x3xf32>, %d_gemmM : index) {
    %d_gemmK = constant 3 : index
    %0 = affine.load %arg0[%d_gemmM, %d_gemmK floordiv 9, (%d_gemmK mod 9) floordiv 3, (%d_gemmK mod 9) mod 3] : memref<128x8x3x3xf32>
    return
  }

  // both coordinates are given at run-time
  func @affine_load_dynamic_gemmK_gemmM(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %0 = affine.load %arg0[%d_gemmM, %d_gemmK floordiv 9, (%d_gemmK mod 9) floordiv 3, (%d_gemmK mod 9) mod 3] : memref<128x8x3x3xf32>
    return
  }
}
