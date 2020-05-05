#map0 = affine_map<(d0) -> (d0)>

module @prototype_module {
  func @prototype_copy_loop_global_to_vgpr(%memref_src : memref<128xf32>, %memref_dst : memref<4xf32>, %pos_src : memref<2xi32>, %pos_dst : memref<2xi32>) {
    %src_long_vector = alloca() : memref<4xf32, 5>
    %dst_long_vector = alloca() : memref<4xf32, 5>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    %c0_f32 = constant 0.0 : f32

    %src_coord_0 = load %pos_src[%c0] : memref<2xi32>
    %src_coord_0_index = index_cast %src_coord_0 : i32 to index

    // %src_long_vector_memref = vector.type_cast %src_long_vector : memref<4xf32, 5> to memref<vector<4xf32>, 5>
    // %dst_long_vector_memref = vector.type_cast %dst_long_vector : memref<4xf32, 5> to memref<vector<4xf32>, 5>

    // %long_vector = vector.transfer_read %memref_src[%src_coord_0_index], %c0_f32 {permutation_map = affine_map<(d0) -> (d0)>} : memref<128xf32>, vector<4xf32>

    // vector.transfer_write %long_vector, %src_long_vector[%c0] {permutation_map = affine_map<(d0) -> (d0)>} : vector<4xf32>, memref<4xf32, 5>


    %memref_src_offset_0 = subview %memref_src[][][] : memref<128xf32> to memref<4xf32>

    %memref_src_offset_0_vector = vector.type_cast %memref_src_offset_0 : memref<4xf32> to memref<vector<4xf32>>

    %long_vector = load %memref_src_offset_0_vector[] : memref<vector<4xf32>>

    %memref_dst_vector = vector.type_cast %memref_dst : memref<4xf32> to memref<vector<4xf32>>

    store %long_vector, %memref_dst_vector[] : memref<vector<4xf32>>

    return
  }
}
