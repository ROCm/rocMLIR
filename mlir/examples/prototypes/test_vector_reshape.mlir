module {
  func @test_vector_load_store_1(%input_1d : memref<128xf32>, %output_1d : memref<4xf32>) {
    %c0 = constant 0 : index
    %c124 = constant 124 : index

    %f0 = constant 0.0 : f32

    // load from a constant offset with vector.transfer_read
    %input_vector = vector.transfer_read %input_1d[%c124], %f0 {permutation_map = affine_map<(d0) -> (d0)>} : memref<128xf32>, vector<4xf32>

    // store to a memref with 0 offset
    %output_1d_vector = vector.type_cast %output_1d : memref<4xf32> to memref<vector<4xf32>>
    store %input_vector, %output_1d_vector[] : memref<vector<4xf32>>

    return
  }

  func @test_vector_load_store_2(%input_1d : memref<128xf32>, %output_1d : memref<128xf32>) {
    %c0 = constant 0 : index
    %c124 = constant 124 : index

    %f0 = constant 0.0 : f32

    // load from a constant offset with vector.transfer_read
    %input_vector = vector.transfer_read %input_1d[%c124], %f0 {permutation_map = affine_map<(d0) -> (d0)>} : memref<128xf32>, vector<4xf32>

    // store to a memref with constant offset with vector.transfer_write
    vector.transfer_write %input_vector, %output_1d[%c124] {permutation_map = affine_map<(d0) -> (d0)>} : vector<4xf32>, memref<128xf32>

    return
  }

  func @test_vector_load_store_3(%input_1d : memref<128xf32>, %output_1d : memref<128xf32>) {
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c128 = constant 128 : index

    %f0 = constant 0.0 : f32

    loop.for %iter = %c0 to %c128 step %c4 {
      // load from a constant offset with vector.transfer_read
      %input_vector = vector.transfer_read %input_1d[%iter], %f0 {permutation_map = affine_map<(d0) -> (d0)>} : memref<128xf32>, vector<4xf32>

      // store to a memref with constant offset with vector.transfer_write
      vector.transfer_write %input_vector, %output_1d[%iter] {permutation_map = affine_map<(d0) -> (d0)>} : vector<4xf32>, memref<128xf32>
    }

    return
  }

  // ---------

  func @test_vector_load_store_4(%input_2d : memref<72x128xf32>, %output_1d : memref<4xf32>) {
    %c0 = constant 0 : index
    %c8 = constant 8 : index
    %c124 = constant 124 : index

    %f0 = constant 0.0 : f32

    // load from a constant offset with vector.transfer_read
    %input_vector = vector.transfer_read %input_2d[%c8, %c124], %f0 {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<72x128xf32>, vector<4xf32>

    // store to a memref with 0 offset
    %output_1d_vector = vector.type_cast %output_1d : memref<4xf32> to memref<vector<4xf32>>
    store %input_vector, %output_1d_vector[] : memref<vector<4xf32>>

    return
  }

  func @test_vector_load_store_5(%input_4d : memref<128x8x32x32xf32>, %output_1d : memref<128xf32>) {
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c7 = constant 7 : index
    %c8 = constant 8 : index
    %c24 = constant 24 : index
    %c28 = constant 28 : index
    %c124 = constant 124 : index

    %f0 = constant 0.0 : f32

    // load from a constant offset with vector.transfer_read
    %input_vector = vector.transfer_read %input_4d[%c124, %c7, %c28, %c24], %f0 {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3)>} : memref<128x8x32x32xf32>, vector<4xf32>

    // store to a memref with constant offset with vector.transfer_write
    vector.transfer_write %input_vector, %output_1d[%c124] {permutation_map = affine_map<(d0) -> (d0)>} : vector<4xf32>, memref<128xf32>

    return
  }

  func @test_vector_load_store_6(%input_4d : memref<128x8x32x32xf32>, %output_1d : memref<128xf32>) {
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c7 = constant 7 : index
    %c8 = constant 8 : index
    %c24 = constant 24 : index
    %c28 = constant 28 : index
    %c32 = constant 32 : index
    %c124 = constant 124 : index

    %f0 = constant 0.0 : f32

    // loop over the fastest changing dimension
    loop.for %iter = %c0 to %c32 step %c4 {
      // load from a constant offset with vector.transfer_read
      %input_vector = vector.transfer_read %input_4d[%c124, %c7, %c28, %iter], %f0 {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3)>} : memref<128x8x32x32xf32>, vector<4xf32>

      // store to a memref with constant offset with vector.transfer_write
      vector.transfer_write %input_vector, %output_1d[%iter] {permutation_map = affine_map<(d0) -> (d0)>} : vector<4xf32>, memref<128xf32>
    }

    return
  }
}
