// map from pad + embed
#map1 = (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)

// map from pad + embed + merge
#map2 = (d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)

// n : (%d_gemmK, %d_gemmN) -> (%d_gemmN floordiv 900)
// c : (%d_gemmK, %d_gemmN) -> (%d_gemmK floordiv 9)
// h : (%d_gemmK, %d_gemmN) -> (%d_gemmk mod 9) floordiv 3 * 2 + (%d_gemmN mod 900) floordiv 30
// w : (%d_gemmK, %d_gemmN) -> (%d_gemmK mod 9) mod 3 * 2 + (%d_gemmN mod 900) mod 30

// diff maps for %d_gemmK = 8
#map_gemmK_delta8 = (d1, d2, d3) -> (d1, d2 + 4, d3 + 4)

module {
  // both coordinates are known at run-time
  // offsets (0, 0) are known at compile-time
  func @affine_load_store_dynamic_gemmK_gemmM_delta0(%arg0: memref<128x8x32x32xf32>, %d_gemmK : index, %d_gemmN : index) {
    %c1 = constant 1.0 : f32
    %0 = affine.load %arg0[%d_gemmN floordiv 900,
                           %d_gemmK floordiv 9,
                           (%d_gemmK mod 9) floordiv 3 * 2 + (%d_gemmN mod 900) floordiv 30,
                           (%d_gemmK mod 9) mod 3 * 2 + (%d_gemmN mod 900) mod 30] : memref<128x8x32x32xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmN floordiv 900,
                           %d_gemmK floordiv 9,
                           (%d_gemmK mod 9) floordiv 3 * 2 + (%d_gemmN mod 900) floordiv 30,
                           (%d_gemmK mod 9) mod 3 * 2 + (%d_gemmN mod 900) mod 30] : memref<128x8x32x32xf32>
    return
  }



  // both coordinates are known at run-time
  // offsets (8, 0) are known at compile-time
  // apply #map_gemmK_delta8
  func @affine_load_store_dynamic_gemmK_gemmN_delta8(%arg0: memref<128x8x32x32xf32>, %d_gemmK : index, %d_gemmN : index) {
    %c1 = constant 1.0 : f32
    %0 = affine.load %arg0[%d_gemmN floordiv 900,
                           %d_gemmK floordiv 9,
                           (%d_gemmK mod 9) floordiv 3 * 2 + (%d_gemmN mod 900) floordiv 30 + 4,
                           (%d_gemmK mod 9) mod 3 * 2 + (%d_gemmN mod 900) mod 30 + 4] : memref<128x8x32x32xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmN floordiv 900,
                           %d_gemmK floordiv 9,
                           (%d_gemmK mod 9) floordiv 3 * 2 + (%d_gemmN mod 900) floordiv 30 + 4,
                           (%d_gemmK mod 9) mod 3 * 2 + (%d_gemmN mod 900) mod 30 + 4] : memref<128x8x32x32xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (8, 0) are known at compile-time
  // apply #map_gemmK_delta8
  func @affine_load_store_dynamic_gemmK_gemmN_delta8_keep_lower_level_index(%arg0: memref<128x8x32x32xf32>, %d_gemmK : index, %d_gemmN : index) {
    %c1 = constant 1.0 : f32

    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %c30 = constant 30 : index
    %c900 = constant 900 : index

    // n : (%d_gemmK, %d_gemmN) -> (%d_gemmN floordiv 900)
    // c : (%d_gemmK, %d_gemmN) -> (%d_gemmK floordiv 9)
    // h : (%d_gemmK, %d_gemmN) -> (%d_gemmK mod 9) floordiv 3 * 2 + (%d_gemmN mod 900) floordiv 30
    // w : (%d_gemmK, %d_gemmN) -> (%d_gemmK mod 9) mod 3 * 2 + (%d_gemmN mod 900) mod 30

    %n = divi_signed %d_gemmN, %c900 : index
    %c = divi_signed %d_gemmK, %c9 : index
    %tmp0 = remi_signed %d_gemmK, %c9 : index
    %tmp1 = divi_signed %tmp0, %c3 : index
    %tmp2 = muli %tmp1, %c2 : index
    %tmp3 = remi_signed %d_gemmN, %c900 : index
    %tmp4 = divi_signed %tmp3, %c30 : index
    %h = addi %tmp2, %tmp4 : index
   
    %tmp5 = remi_signed %tmp0, %c3 : index
    %tmp6 = muli %tmp5, %c2 : index
    %tmp7 = remi_signed %tmp3, %c30 : index
    %w = addi %tmp6, %tmp7 : index
    
    %0 = affine.load %arg0[%n,
                           %c,
                           %h + 4,
                           %w + 4] : memref<128x8x32x32xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%n,
                           %c,
                           %h + 4,
                           %w + 4] : memref<128x8x32x32xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (8, 0) are known at compile-time
  // apply #map_gemmK_delta8
  func @affine_load_store_dynamic_gemmK_gemmN_delta8_keep_lower_level_index_with_i32(%arg0: memref<128x8x32x32xf32>, %d_gemmK : i32, %d_gemmN : i32) {
    %c1 = constant 1.0 : f32

    %c0 = constant 0 : i32
    %c2 = constant 2 : i32
    %c3 = constant 3 : i32
    %c9 = constant 9 : i32
    %c30 = constant 30 : i32
    %c900 = constant 900 : i32

    // n : (%d_gemmK, %d_gemmN) -> (%d_gemmN floordiv 900)
    // c : (%d_gemmK, %d_gemmN) -> (%d_gemmK floordiv 9)
    // h : (%d_gemmK, %d_gemmN) -> (%d_gemmK mod 9) floordiv 3 * 2 + (%d_gemmN mod 900) floordiv 30
    // w : (%d_gemmK, %d_gemmN) -> (%d_gemmK mod 9) mod 3 * 2 + (%d_gemmN mod 900) mod 30

    %n = divi_signed %d_gemmN, %c900 : i32
    %c = divi_signed %d_gemmK, %c9 : i32
    %tmp0 = remi_signed %d_gemmK, %c9 : i32
    %tmp1 = divi_signed %tmp0, %c3 : i32
    %tmp2 = muli %tmp1, %c2 : i32
    %tmp3 = remi_signed %d_gemmN, %c900 : i32
    %tmp4 = divi_signed %tmp3, %c30 : i32
    %h = addi %tmp2, %tmp4 : i32

    %tmp5 = remi_signed %tmp0, %c3 : i32
    %tmp6 = muli %tmp5, %c2 : i32
    %tmp7 = remi_signed %tmp3, %c30 : i32
    %w = addi %tmp6, %tmp7 : i32

    %n_index = index_cast %n : i32 to index
    %c_index = index_cast %c : i32 to index
    %h_index = index_cast %h : i32 to index
    %w_index = index_cast %w : i32 to index

    %0 = affine.load %arg0[%n_index,
                           %c_index,
                           %h_index + 4,
                           %w_index + 4] : memref<128x8x32x32xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%n_index,
                           %c_index,
                           %h_index + 4,
                           %w_index + 4] : memref<128x8x32x32xf32>
    return
  }
}
