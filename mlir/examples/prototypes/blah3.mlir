#map0 = (d_gemmK, d_gemmM) -> (d_gemmM, d_gemmK floordiv 9, (d_gemmK mod 9) floordiv 3, (d_gemmK mod 9) mod 3)

// diff map on gemmK dimension
#map_gemmK_delta1_carry_000 = (d2, d3, d4) -> (d2, d3, d4 + 1)
#map_gemmK_delta1_carry_001 = (d2, d3, d4) -> (d2, d3 + 1, d4 + 1 - 3)
#map_gemmK_delta1_carry_011 = (d2, d3, d4) -> (d2 + 1, d3 + 1 - 3, d4 + 1 - 3)
#map_gemmK_delta1_carry_010 = (d2, d3, d4) -> (d2 + 1, d3 + 1 - 3, d4 + 1)
#map_gemmK_delta1_carry_generic = (d2, d3, d4) -> (
                                                   d2 + (d3 + (d4 + 1) floordiv 3) floordiv 3,
                                                   (d3 + (d4 + 1) floordiv 3) mod 3,
                                                   (d4 + 1) mod 3
                                                  )

module {
  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_geneic map
  func @affine_load_dynamic_gemmK_gemmM_with_carry_generic(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    // d2 = (%d_gemmK floordiv 9)
    // d3 = (%d_gemmK mod 9) floordiv 3
    // d4 = (%d_gemmK mod 9) mod 3

    %c1 = constant 1.0 : f32
    %0 = affine.load %arg0[%d_gemmM,
                           %d_gemmK floordiv 9 + ((%d_gemmK mod 9) floordiv 3 + ((%d_gemmK mod 9) mod 3 + 1) floordiv 3) floordiv 3,
                           ((%d_gemmK mod 9) floordiv 3 + ((%d_gemmK mod 9) mod 3 + 1) floordiv 3) mod 3,
                           ((%d_gemmK mod 9) mod 3 + 1) mod 3] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d_gemmK floordiv 9 + ((%d_gemmK mod 9) floordiv 3 + ((%d_gemmK mod 9) mod 3 + 1) floordiv 3) floordiv 3,
                           ((%d_gemmK mod 9) floordiv 3 + ((%d_gemmK mod 9) mod 3 + 1) floordiv 3) mod 3,
                           ((%d_gemmK mod 9) mod 3 + 1) mod 3] : memref<128x8x3x3xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_generic map
  func @affine_load_dynamic_gemmK_gemmM_with_carry_generic_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %d2 = divi_signed %d_gemmK, %c9 : index
    %d_tmp = remi_signed %d_gemmK, %c9 : index
    %d3 = divi_signed %d_tmp, %c3 : index
    %d4 = remi_signed %d_tmp, %c3 : index

    %0 = affine.load %arg0[%d_gemmM,
                           %d2 + (%d3 + (%d4 + 1) floordiv 3) floordiv 3,
                           (%d3 + (%d4 + 1) floordiv 3) mod 3,
                           (%d4 + 1) mod 3] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d2 + (%d3 + (%d4 + 1) floordiv 3) floordiv 3,
                           (%d3 + (%d4 + 1) floordiv 3) mod 3,
                           (%d4 + 1) mod 3] : memref<128x8x3x3xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_000 map
  func @affine_load_dynamic_gemmK_gemmM_with_carry_000(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    // d2 = (%d_gemmK floordiv 9)
    // d3 = (%d_gemmK mod 9) floordiv 3
    // d4 = (%d_gemmK mod 9) mod 3

    %c1 = constant 1.0 : f32
    %0 = affine.load %arg0[%d_gemmM,
                           %d_gemmK floordiv 9,
                           (%d_gemmK mod 9) floordiv 3,
                           (%d_gemmK mod 9) mod 3 + 1] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d_gemmK floordiv 9,
                           (%d_gemmK mod 9) floordiv 3,
                           (%d_gemmK mod 9) mod 3 + 1] : memref<128x8x3x3xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_000 map
  func @affine_load_dynamic_gemmK_gemmM_with_carry_000_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %d2 = divi_signed %d_gemmK, %c9 : index
    %d_tmp = remi_signed %d_gemmK, %c9 : index
    %d3 = divi_signed %d_tmp, %c3 : index
    %d4 = remi_signed %d_tmp, %c3 : index 

    %0 = affine.load %arg0[%d_gemmM,
                           %d2,
                           %d3,
                           %d4 + 1] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d2,
                           %d3,
                           %d4 + 1] : memref<128x8x3x3xf32>
    return
  }
}
