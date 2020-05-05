#map0 = affine_map<(d_gemmK, d_gemmM) -> (d_gemmM, d_gemmK floordiv 9, (d_gemmK mod 9) floordiv 3, (d_gemmK mod 9) mod 3)>

// diff map on gemmK dimension with delta 1
#map_gemmK_delta1_carry_000 = affine_map<(d2, d3, d4) -> (d2, d3, d4 + 1)>
#map_gemmK_delta1_carry_001 = affine_map<(d2, d3, d4) -> (d2, d3 + 1, d4 + 1 - 3)>
#map_gemmK_delta1_carry_011 = affine_map<(d2, d3, d4) -> (d2 + 1, d3 + 1 - 3, d4 + 1 - 3)>
#map_gemmK_delta1_carry_generic = affine_map<(d2, d3, d4) -> (
                                                   d2 + (d3 + (d4 + 1) floordiv 3) floordiv 3,
                                                   (d3 + (d4 + 1) floordiv 3) mod 3,
                                                   (d4 + 1) mod 3
                                                  )>

// NOTE: this map is actually invalid as it's NOT possible for d4 to not carry while d3 carries.
#map_gemmK_delta1_carry_010 = affine_map<(d2, d3, d4) -> (d2 + 1, d3 + 1 - 3, d4 + 1)>



// diff map on gemmK dimension with delta 8 -> (0, 2, 2)
#map_gemmK_delta8_carry_000 = affine_map<(d2, d3, d4) -> (d2, d3 + 2, d4 + 2)>
#map_gemmK_delta8_carry_011 = affine_map<(d2, d3, d4) -> (d2 + 1, d3, d4 - 1)>
#map_gemmK_delta8_carry_010 = affine_map<(d2, d3, d4) -> (d2 + 1, d3 - 1, d4 + 2)>

// NOTE: this map is actually invalid as it's NOT possible for d4 to carry while d4 NOT carry.
#map_gemmK_delta8_carry_001 = affine_map<(d2, d3, d4) -> (d2, d3 + 3, d4 - 1)>


module {
  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_geneic map
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_generic(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
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
  // apply #map_gemmK_delta1_carry_geneic map
  // use i32 type
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_generic_with_i32(%arg0: memref<128x8x3x3xf32>, %d_gemmK_i32 : i32, %d_gemmM_i32 : i32) {
    // d2 = (%d_gemmK floordiv 9)
    // d3 = (%d_gemmK mod 9) floordiv 3
    // d4 = (%d_gemmK mod 9) mod 3

    %c1 = constant 1.0 : f32
    %d_gemmM = index_cast %d_gemmM_i32 : i32 to index
    %d_gemmK = index_cast %d_gemmK_i32 : i32 to index
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
  // use i32 type
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_generic_keep_lower_level_index_with_i32(%arg0: memref<128x8x3x3xf32>, %d_gemmK_i32 : i32, %d_gemmM_i32 : i32) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : i32
    %c9 = constant 9 : i32
    %d2_i32 = divi_signed %d_gemmK_i32, %c9 : i32
    %d_tmp = remi_signed %d_gemmK_i32, %c9 : i32
    %d3_i32 = divi_signed %d_tmp, %c3 : i32
    %d4_i32 = remi_signed %d_tmp, %c3 : i32

    %d_gemmM = index_cast %d_gemmM_i32 : i32 to index
    %d2 = index_cast %d2_i32 : i32 to index
    %d3 = index_cast %d3_i32 : i32 to index
    %d4 = index_cast %d4_i32 : i32 to index

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
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_000(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
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
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_000_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
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

  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_000 map
  // use i32 type
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_000_keep_lower_level_index_with_i32(%arg0: memref<128x8x3x3xf32>, %d_gemmK_i32 : i32, %d_gemmM_i32 : i32) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : i32
    %c9 = constant 9 : i32
    %d2_i32 = divi_signed %d_gemmK_i32, %c9 : i32
    %d_tmp = remi_signed %d_gemmK_i32, %c9 : i32
    %d3_i32 = divi_signed %d_tmp, %c3 : i32
    %d4_i32 = remi_signed %d_tmp, %c3 : i32

    %d_gemmM = index_cast %d_gemmM_i32 : i32 to index
    %d_gemmK = index_cast %d_gemmK_i32 : i32 to index
    %d2 = index_cast %d2_i32 : i32 to index
    %d3 = index_cast %d3_i32 : i32 to index
    %d4 = index_cast %d4_i32 : i32 to index

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

  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_001 map
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_001_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %d2 = divi_signed %d_gemmK, %c9 : index
    %d_tmp = remi_signed %d_gemmK, %c9 : index
    %d3 = divi_signed %d_tmp, %c3 : index
    %d4 = remi_signed %d_tmp, %c3 : index

    %0 = affine.load %arg0[%d_gemmM,
                           %d2,
                           %d3 + 1,
                           %d4 + 1 - 3] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d2,
                           %d3 + 1,
                           %d4 + 1 - 3] : memref<128x8x3x3xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (1, 0) are known at compile-time
  // apply #map_gemmK_delta1_carry_011 map
  func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_011_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %d2 = divi_signed %d_gemmK, %c9 : index
    %d_tmp = remi_signed %d_gemmK, %c9 : index
    %d3 = divi_signed %d_tmp, %c3 : index
    %d4 = remi_signed %d_tmp, %c3 : index

    %0 = affine.load %arg0[%d_gemmM,
                           %d2 + 1,
                           %d3 + 1 - 3,
                           %d4 + 1 - 3] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d2 + 1,
                           %d3 + 1 - 3,
                           %d4 + 1 - 3] : memref<128x8x3x3xf32>
    return
  }

  // NOTE: carry_010_map is actually invalid.
  //// both coordinates are known at run-time
  //// offsets (1, 0) are known at compile-time
  //// apply #map_gemmK_delta1_carry_010 map
  //func @affine_load_store_dynamic_gemmK_gemmM_delta1_with_carry_010_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
  //  %c1 = constant 1.0 : f32

  //  %c3 = constant 3 : index
  //  %c9 = constant 9 : index
  //  %d2 = divi_signed %d_gemmK, %c9 : index
  //  %d_tmp = remi_signed %d_gemmK, %c9 : index
  //  %d3 = divi_signed %d_tmp, %c3 : index
  //  %d4 = remi_signed %d_tmp, %c3 : index

  //  %0 = affine.load %arg0[%d_gemmM,
  //                         %d2 + 1,
  //                         %d3 + 1 - 3,
  //                         %d4 + 1] : memref<128x8x3x3xf32>
  //  %1 = addf %0, %c1 : f32
  //  affine.store %1, %arg0[%d_gemmM,
  //                         %d2 + 1,
  //                         %d3 + 1 - 3,
  //                         %d4 + 1] : memref<128x8x3x3xf32>
  //  return
  //}




  // both coordinates are known at run-time
  // offsets (8, 0) are known at compile-time
  // apply #map_gemmK_delta8_carry_000 map
  func @affine_load_store_dynamic_gemmK_gemmM_delta8_with_carry_000_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %d2 = divi_signed %d_gemmK, %c9 : index
    %d_tmp = remi_signed %d_gemmK, %c9 : index
    %d3 = divi_signed %d_tmp, %c3 : index
    %d4 = remi_signed %d_tmp, %c3 : index

    %0 = affine.load %arg0[%d_gemmM,
                           %d2,
                           %d3 + 2,
                           %d4 + 2] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d2,
                           %d3 + 2,
                           %d4 + 2] : memref<128x8x3x3xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (8, 0) are known at compile-time
  // apply #map_gemmK_delta8_carry_011 map
  func @affine_load_store_dynamic_gemmK_gemmM_delta8_with_carry_011_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %d2 = divi_signed %d_gemmK, %c9 : index
    %d_tmp = remi_signed %d_gemmK, %c9 : index
    %d3 = divi_signed %d_tmp, %c3 : index
    %d4 = remi_signed %d_tmp, %c3 : index

    %0 = affine.load %arg0[%d_gemmM,
                           %d2 + 1,
                           %d3,
                           %d4 - 1] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d2 + 1,
                           %d3,
                           %d4 - 1] : memref<128x8x3x3xf32>
    return
  }

  // both coordinates are known at run-time
  // offsets (8, 0) are known at compile-time
  // apply #map_gemmK_delta8_carry_010 map
  func @affine_load_store_dynamic_gemmK_gemmM_delta8_with_carry_010_keep_lower_level_index(%arg0: memref<128x8x3x3xf32>, %d_gemmK : index, %d_gemmM : index) {
    %c1 = constant 1.0 : f32

    %c3 = constant 3 : index
    %c9 = constant 9 : index
    %d2 = divi_signed %d_gemmK, %c9 : index
    %d_tmp = remi_signed %d_gemmK, %c9 : index
    %d3 = divi_signed %d_tmp, %c3 : index
    %d4 = remi_signed %d_tmp, %c3 : index

    %0 = affine.load %arg0[%d_gemmM,
                           %d2 + 1,
                           %d3 - 1,
                           %d4 + 2] : memref<128x8x3x3xf32>
    %1 = addf %0, %c1 : f32
    affine.store %1, %arg0[%d_gemmM,
                           %d2 + 1,
                           %d3 - 1,
                           %d4 + 2] : memref<128x8x3x3xf32>
    return
  }
}
