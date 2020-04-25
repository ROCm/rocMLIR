module {
  func @test_vector(%arg0 : vector<4xf32>, %arg1 : memref<128xf32>) {
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c32 = constant 32 : index
    %c128 = constant 128 : index

    %c0i = constant 0 : i32
    %c1i = constant 1 : i32
    %c2i = constant 2 : i32
    %c3i = constant 3 : i32

    %c0f = constant 0.0 : f32
    %c1f = constant 1.0 : f32
    %c2f = constant 2.0 : f32
    %c3f = constant 3.0 : f32

    %0 = vector.extractelement %arg0[%c0i : i32] : vector<4xf32>
    %1 = vector.extractelement %arg0[%c1i : i32] : vector<4xf32>
    %2 = vector.extractelement %arg0[%c2i : i32] : vector<4xf32>
    %3 = vector.extractelement %arg0[%c3i : i32] : vector<4xf32>

    %4 = vector.extract %arg0[0] : vector<4xf32>
    %5 = vector.extract %arg0[1] : vector<4xf32>
    %6 = vector.extract %arg0[2] : vector<4xf32>
    %7 = vector.extract %arg0[3] : vector<4xf32>

    %8 = vector.insert %c0f, %arg0[0] : f32 into vector<4xf32>
    %9 = vector.insert %c1f, %arg0[0] : f32 into vector<4xf32>
    %10 = vector.insert %c2f, %arg0[0] : f32 into vector<4xf32>
    %11 = vector.insert %c3f, %arg0[0] : f32 into vector<4xf32>
    
    %12 = vector.insertelement %c0f, %arg0[%c0i : i32] : vector<4xf32>
    %13 = vector.insertelement %c1f, %arg0[%c1i : i32] : vector<4xf32>
    %14 = vector.insertelement %c2f, %arg0[%c2i : i32] : vector<4xf32>
    %15 = vector.insertelement %c3f, %arg0[%c3i : i32] : vector<4xf32>


    %tmp = alloc() : memref<4xf32>
    loop.for %iter = %c0 to %c128 step %c4 {
      // vector read
      %v = vector.transfer_read %arg1[%c0], %c0f {permutation_map = (d0)->(d0)} : memref<128xf32>, vector<4xf32>

      // vectore write
      vector.transfer_write %v, %tmp[%c0] {permutation_map = (d0) -> (d0)} : vector<4xf32>, memref<4xf32>
    }
 
    return
  }
}
