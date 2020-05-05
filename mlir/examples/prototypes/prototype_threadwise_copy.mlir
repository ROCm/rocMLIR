gpu.module @prototype_module {
  gpu.func @prototype_threadwise_copy_filter_tensor_load(%memref_src : memref<72x128xf32>, %memref_dst : memref<4xf32>, %pos_src : memref<2xi32>, %pos_dst : memref<2xi32>)
    workgroup()
    private(
      %src_coord : memref<2xi32, 5>,
      %dst_coord : memref<2xi32, 5>,

      // p_src_long_vector[long_vector_size]
      %p_src_long_vector : memref<4xf32, 5>,

      // p_dst_long_vector[long_vector_size]
      %p_dst_long_vector : memref<4xf32, 5>
    ) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index

    %c0_i32 = constant 0 : i32
    %c0_f32 = constant 0.0 : f32

    %slice_lengths_0 = constant 1 : i32
    %slice_lengths_1 = constant 4 : i32
    // src_dst_dim_access_order = <1, 0>
    %src_dst_dim_access_vector_read_write_dim = constant 1 : i32
    %src_data_per_read = constant 4 : i32
    %dst_data_per_write = constant 1 : i32
    //%src_addrspace = constant 1 : i32
    //%dst_addrspace = constant 5 : i32

    %vector_access_dim = addi %src_dst_dim_access_vector_read_write_dim, %c0_i32 : i32
    %src_data_per_access = addi %src_data_per_read, %c0_i32 : i32
    %dst_data_per_access = addi %dst_data_per_write, %c0_i32 : i32

    // long_vector_size = lcm(src_data_per_read, dst_data_per_write)
    %long_vector_size = addi %src_data_per_read, %c0_i32 : i32

    // long_vector_access_lengths = SliceLengths::Modify(vector_access_dim, SliceLengths::Get(vector_access_dim) / long_vector_size
    %long_vector_access_lengths_0_tmp = addi %slice_lengths_0, %c0_i32 : i32
    %long_vector_access_lengths_0 = index_cast %long_vector_access_lengths_0_tmp : i32 to index
    %long_vector_access_lengths_1_tmp = divi_signed %slice_lengths_1, %src_data_per_read : i32
    %long_vector_access_lengths_1 = index_cast %long_vector_access_lengths_1_tmp : i32 to index

    loop.for %iter_level_0 = %c0 to %long_vector_access_lengths_0 step %c1 {
      loop.for %iter_level_1 = %c0 to %long_vector_access_lengths_1 step %c1 {
        %long_vector_data_begin_id_0 = index_cast %iter_level_0 : index to i32
        %long_vector_data_begin_id_1 = index_cast %iter_level_1 : index to i32

        %long_vector_size_index = index_cast %long_vector_size : i32 to index
        loop.for %i = %c0 to %long_vector_size_index step %c1 {
          store %c0_f32, %p_src_long_vector[%i] : memref<4xf32, 5>
        }

        %read_loop_iterations = divi_signed %long_vector_size, %src_data_per_access : i32
        %read_loop_iterations_index = index_cast %read_loop_iterations : i32 to index
        loop.for %i = %c0 to %read_loop_iterations_index step %c1 {
          %scalar_id_0 = constant 0 : i32
          %i_i32 = index_cast %i : index to i32

          // scalar_id(vector_access_dim) = i * src_data_per_access
          %scalar_id_1 = muli %i_i32, %src_data_per_access : i32
          %buffer_offset = muli %i_i32, %src_data_per_access : i32

          // src_coord = pos_src + (long_vector_data_begin_id_0, long_vector_data_being_id_1) + (scalar_id_0, scalar_id_1)
          %move_0 = addi %long_vector_data_begin_id_0, %scalar_id_0 : i32
          %move_1 = addi %long_vector_data_begin_id_1, %scalar_id_1 : i32

          %pos_src_0 = load %pos_src[%c0] : memref<2xi32>
          %pos_src_1 = load %pos_src[%c1] : memref<2xi32>
          %src_coord_0 = addi %pos_src_0, %move_0 : i32
          %src_coord_1 = addi %pos_src_1, %move_1 : i32
          store %src_coord_0, %src_coord[%c0] : memref<2xi32, 5>
          store %src_coord_1, %src_coord[%c1] : memref<2xi32, 5>

          // this check can be skipped for filter tensor as there is no Pad
          // if (src_coord.IsOffsetValidAssumingUpperIndexIsValid())

          // TBD
          //   transferData<f32, %src_data_per_read, Global, Vgpr, Set>
          //   (memref_src, src_coord.GetOffset(), p_src_long_vector, buffer_offset)
          %src_coord_0_index = index_cast %src_coord_0 : i32 to index
          %src_coord_1_index = index_cast %src_coord_1 : i32 to index

          %long_vector = vector.transfer_read %memref_src[%src_coord_0_index, %src_coord_1_index], %c0_f32 {permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<72x128xf32>, vector<4xf32>
          vector.transfer_write %long_vector, %p_src_long_vector[%c0] {permutation_map = affine_map<(d0) -> (d0)>} : vector<4xf32>, memref<4xf32, 5>
        }

        loop.for %i = %c0 to %long_vector_size_index step %c1 {
          // p_dst_long_vector[i] = type_convert<DstData>{}(p_src_long_vector[i])
          %v = load %p_src_long_vector[%i] : memref<4xf32, 5>
          store %v, %p_dst_long_vector[%i] : memref<4xf32, 5>
        }

        %write_loop_iterations = divi_signed %long_vector_size, %dst_data_per_access : i32
        %write_loop_iterations_index = index_cast %write_loop_iterations : i32 to index
        loop.for %i = %c0 to %write_loop_iterations_index step %c1 {
          %scalar_id_0 = constant 0 : i32
          %i_i32 = index_cast %i : index to i32

          // scalar_id(vector_access_dim) = i * dst_data_per_access
          %scalar_id_1 = muli %i_i32, %dst_data_per_access : i32
          %buffer_offset = muli %i_i32, %dst_data_per_access : i32
          
          // dst_coord = pos_dst + (long_vector_data_begin_id_0, long_vector_data_being_id_1) + (scalar_id_0, scalar_id_1)
          %move_0 = addi %long_vector_data_begin_id_0, %scalar_id_0 : i32
          %move_1 = addi %long_vector_data_begin_id_1, %scalar_id_1 : i32

          %pos_dst_0 = load %pos_dst[%c0] : memref<2xi32>
          %pos_dst_1 = load %pos_dst[%c1] : memref<2xi32>
          %dst_coord_0 = addi %pos_dst_0, %move_0 : i32
          %dst_coord_1 = addi %pos_dst_1, %move_1 : i32
          store %dst_coord_0, %dst_coord[%c0] : memref<2xi32, 5>
          store %dst_coord_1, %dst_coord[%c1] : memref<2xi32, 5>

          // this check can be skipped for filter tensor as there is no Pad
          // if (dst_coord.IsOffsetValidAssumingUpperIndexIsValid())

          // TBD
          //   transferData<f32, %dst_data_per_write, Vgpr, Vgpr, Set>
          //   (memref_dst, dst_coord.GetOffset(), p_src_long_vector, buffer_offset)
          %dst_coord_0_index = index_cast %dst_coord_0 : i32 to index
          %dst_coord_1_index = index_cast %dst_coord_1 : i32 to index

          %long_vector = vector.transfer_read %p_dst_long_vector[%c0], %c0_f32 {permutation_map = affine_map<(d0) -> (d0)>} : memref<4xf32, 5>, vector<4xf32>

          vector.transfer_write %long_vector, %memref_dst[%dst_coord_0_index] {permutation_map = affine_map<(d0) -> (d0)>} : vector<4xf32>, memref<4xf32>
        }

      }
    }

    gpu.return
  }

  // move 2D position
  func @move_pos(%pos : memref<2xi32>, %d0 : i32, %d1 : i32) -> memref<2xi32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %coord_0 = load %pos[%c0] : memref<2xi32>
    %coord_1 = load %pos[%c1] : memref<2xi32>
    %new_coord_0 = addi %coord_0, %d0 : i32
    %new_coord_1 = addi %coord_1, %d1 : i32
    store %new_coord_0, %pos[%c0] : memref<2xi32>
    store %new_coord_1, %pos[%c1] : memref<2xi32>
    return %pos : memref<2xi32>
  }

  // test harness.
  // func @main(%memref_src : memref<72x128xf32>, %memref_dst : memref<4xf32>, %pos_src : memref<2xi32>, %pos_dst : memref<2xi32>) {
  //   %c0 = constant 0 : index
  //   %c1 = constant 1 : index
  //   %c2 = constant 2 : index

  //   %c0_i32 = constant 0 : i32
  //   %c8_i32 = constant 8 : i32

  //   // initialize pos_src, pos_dst.
  //   loop.for %i = %c0 to %c2 step %c1 {
  //     store %c0_i32, %pos_src[%i] : memref<2xi32>
  //     store %c0_i32, %pos_dst[%i] : memref<2xi32>
  //   }

  //   %pos_new_src = call @move_pos(%pos_src, %c8_i32, %c0_i32) : (memref<2xi32>, i32, i32) -> memref<2xi32>

  //   %tid = "gpu.thread_id"() {dimension = "x"} : () -> index
  //   %bid = "gpu.block_id"() {dimension = "x"} : () -> index

  //   return
  // }
}
