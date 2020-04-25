
// upper index -> lower index maps
#map_filter_0 = (d0, d1) -> (d1)
#map_filter_1 = (d0, d1) -> (d0 floordiv 9)
#map_filter_2 = (d0, d1) -> ((d0 floordiv 9) floordiv 3)
#map_filter_3 = (d0, d1) -> ((d0 floordiv 9) mod 3)


module {
  func @test(%arg0 : memref<?xi32>) {

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c0_i32 = constant 0 : i32

    %dim0_memref = dim %arg0, 0 : memref<?xi32>

    loop.for %i = %c0 to %dim0_memref step %c1 {
      %i_i32 = index_cast %i : index to i32
      store %i_i32, %arg0[%i] : memref<?xi32>
    }

    return
  }


  func @test(%pos_src : memref<2xi32>, %memref_src : memref<128x8x3x3xf32>, %memref_dst : memref<8x128xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index

    %c1f = constant 1.0 : f32

    // retrieve (gemmK, gemmM)
    %gemmK = load %arg0[%c0] : memref<4xi32>
    %gemmM = load %arg0[%c1] : memref<4xi32>

    %k = affine.apply %map_filter_0 (%gemmK, %gemmM)
    %c = affine.apply %map_filter_0 (%gemmK, %gemmM)
    %y = affine.apply %map_filter_0 (%gemmK, %gemmM)
    %x = affine.apply %map_filter_0 (%gemmK, %gemmM)

    %data = affine.load %memref_src[%k, %c, %y, %x] : memref<128x8x3x3xf32>
    %1 = addf %data, %c1f : f32

    affine.store 

  }
}
