

module {
  func @miopen_conv2d_kcyx_nchw_nkhw(%arg0: memref<1024x1024x1x1xf32>, %arg1: memref<128x1024x14x14xf32>, %arg2: memref<128x1024x14x14xf32>) {
    %a00 = memref_cast %arg0 :memref<1024x1024x1x1xf32> to memref<*xf32>
    %a0 = memref_cast %a00 :memref<*xf32> to memref<1x1024x1024x1x1xf32>
    %a11 = memref_cast %arg1 :memref<128x1024x14x14xf32> to memref<*xf32>
    %a1 = memref_cast %a11 :memref<*xf32> to memref<128x1x1024x14x14xf32>
    %a22 = memref_cast %arg2 :memref<128x1024x14x14xf32> to memref<*xf32>
    %a2 = memref_cast %a22 :memref<*xf32> to memref<128x1x1024x14x14xf32>
    %0 = miopen.transform(%a0) {gridwise_gemm_argument_position = 0 : i32, layout = [ {dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["g"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmK"], source_dimensions = [2 : i32, 3 : i32, 4 : i32], source_names = ["cgroup", "y", "x"], transformation = "Unfold"}, {dimensions = [2 : i32], names = ["gemmM"], source_dimensions = [1 : i32], source_names = ["kgroup"], transformation = "PassThrough"}], output_layout = ["gemmG", "gemmK", "gemmM"], source_layout = ["g","kgroup","cgroup", "y", "x"]} : memref<1x1024x1024x1x1xf32> to memref<1x1024x1024xf32>
    %1 = miopen.transform(%a1) {layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [1 : i32], source_names = ["g"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["ni"], source_dimensions = [0 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [2 : i32], names = ["cgroup"], source_dimensions = [2 : i32], source_names = ["ci"], transformation = "PassThrough"}, {dimensions = [3 : i32, 4 : i32], names = ["hipad", "wipad"], parameters = [0 : i32, 0 : i32], source_dimensions = [3 : i32, 4 : i32], source_names = ["hi", "wi"], transformation = "Pad"}], output_layout = ["gemmG","ni", "cgroup", "hipad", "wipad"], source_layout = ["ni","g", "cgroup", "hi", "wi"]} : memref<128x1x1024x14x14xf32> to memref<1x128x1024x14x14xf32>

    %2 = miopen.transform(%1) {intermediate_layout = ["gemmG", "ni", "cgroup", "hipad", "wipad"], layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["ni"], source_dimensions = [1 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [2 : i32], names = ["cgroup"], source_dimensions = [2 : i32], source_names = ["cgroup"], transformation = "PassThrough"}, {dimensions = [3 : i32, 4 : i32], names = ["y", "ho"], parameters = [1 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [3 : i32], source_names = ["hipad"], transformation = "Embed"}, {dimensions = [5 : i32, 6 : i32], names = ["x", "wo"], parameters = [1 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [4 : i32], source_names = ["wipad"], transformation = "Embed"}], output_layout = ["gemmG","ni", "cgroup", "y", "ho", "x", "wo"]} : memref<1x128x1024x14x14xf32> to memref<1x128x1024x1x14x1x14xf32>

    %3 = miopen.transform(%2) {gridwise_gemm_argument_position = 1 : i32, intermediate_layout = ["gemmG", "ni", "cgroup", "y", "ho", "x", "wo"], layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmK"], source_dimensions = [2 : i32, 3 : i32, 5 : i32], source_names = ["cgroup", "y", "x"], transformation = "Merge"}, {dimensions = [2 : i32], names = ["gemmN"], source_dimensions = [1 : i32, 4 : i32, 6 : i32], source_names = ["ni", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmG","gemmK", "gemmN"]} : memref<1x128x1024x1x14x1x14xf32> to memref<1x1024x25088xf32>

    %4 = miopen.transform(%a2) {gridwise_gemm_argument_position = 2 : i32, layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [1 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmM"], source_dimensions = [2 : i32], source_names = ["kgroup"], transformation = "PassThrough"}, {dimensions = [2 : i32], names = ["gemmN"], source_dimensions = [0 : i32, 3 : i32, 4 : i32], source_names = ["no", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmG", "gemmM", "gemmN"], source_layout = ["no", "g", "kgroup", "ho", "wo"]} : memref<128x1x1024x14x14xf32> to memref<1x1024x25088xf32>
    miopen.gridwise_gemm_v2(%0, %3, %4) {arch = "gfx908", dilations = [1 : i32, 1 : i32], filter_dimension = [1024, 1024, 1, 1], filter_layout = ["k", "c", "y", "x"], input_dimension = [128, 1024, 14, 14], input_layout = ["ni", "ci", "hi", "wi"], kernel_algorithm = "v4r4", num_cu = 120 : i32, output_dimension = [128, 1024, 14, 14], output_layout = ["no", "ko", "ho", "wo"], padding = [[0 : i32, 0 : i32], [0 : i32, 0 : i32]], strides = [1 : i32, 1 : i32], xdlopsV2 = true, num_group = 1:i32} : memref<1x1024x1024xf32>, memref<1x1024x25088xf32>, memref<1x1024x25088xf32>   
  return
  }
  func @main() {
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 0.000000e+00 : f32
    %0 = alloc() : memref<1024x1024x1x1xf32>
    %1 = alloc() : memref<128x1024x14x14xf32>
    %2 = alloc() : memref<128x1024x14x14xf32>
    %3 = memref_cast %0 : memref<1024x1024x1x1xf32> to memref<?x?x?x?xf32>
    %4 = memref_cast %1 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    %5 = memref_cast %2 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    call @mcpuMemset4DFloat(%3, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
    call @mcpuMemset4DFloat(%4, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
    call @mcpuMemset4DFloat(%5, %cst_0) : (memref<?x?x?x?xf32>, f32) -> ()
    call @gpu_conv(%0, %1, %2) : (memref<1024x1024x1x1xf32>, memref<128x1024x14x14xf32>, memref<128x1024x14x14xf32>) -> ()
    %6 = alloc() : memref<128x1024x14x14xf32>
    %7 = memref_cast %6 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    call @mcpuMemset4DFloat(%7, %cst_0) : (memref<?x?x?x?xf32>, f32) -> ()
    call @conv2d_host(%0, %1, %6) : (memref<1024x1024x1x1xf32>, memref<128x1024x14x14xf32>, memref<128x1024x14x14xf32>) -> ()
    call @verify_results(%6, %2) : (memref<128x1024x14x14xf32>, memref<128x1024x14x14xf32>) -> ()
    dealloc %0 : memref<1024x1024x1x1xf32>
    dealloc %1 : memref<128x1024x14x14xf32>
    dealloc %2 : memref<128x1024x14x14xf32>
    dealloc %6 : memref<128x1024x14x14xf32>
    return
  }
  func @mcpuMemset4DFloat(memref<?x?x?x?xf32>, f32)
  func @gpu_conv(%arg0: memref<1024x1024x1x1xf32>, %arg1: memref<128x1024x14x14xf32>, %arg2: memref<128x1024x14x14xf32>) {
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
    %0 = memref_cast %arg0 : memref<1024x1024x1x1xf32> to memref<?x?x?x?xf32>
    %1 = memref_cast %arg1 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    %2 = memref_cast %arg2 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    %3 = call @mgpuMemAlloc4DFloat(%0) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    %4 = call @mgpuMemAlloc4DFloat(%1) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    %5 = call @mgpuMemAlloc4DFloat(%2) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    call @mgpuMemCopy4DFloat(%0, %3, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy4DFloat(%1, %4, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy4DFloat(%2, %5, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    %6 = memref_cast %3 : memref<?x?x?x?xf32> to memref<1024x1024x1x1xf32>
    %7 = memref_cast %4 : memref<?x?x?x?xf32> to memref<128x1024x14x14xf32>
    %8 = memref_cast %5 : memref<?x?x?x?xf32> to memref<128x1024x14x14xf32>
    call @conv2d(%6, %7, %8) : (memref<1024x1024x1x1xf32>, memref<128x1024x14x14xf32>, memref<128x1024x14x14xf32>) -> ()
    call @mgpuMemCopy4DFloat(%5, %2, %c2_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    call @mgpuMemDealloc4DFloat(%0) : (memref<?x?x?x?xf32>) -> ()
    call @mgpuMemDealloc4DFloat(%1) : (memref<?x?x?x?xf32>) -> ()
    call @mgpuMemDealloc4DFloat(%2) : (memref<?x?x?x?xf32>) -> ()
    return
  }
  func @mgpuMemAlloc4DFloat(memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
  func @mgpuMemCopy4DFloat(memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32)
  func @conv2d(%arg0: memref<1024x1024x1x1xf32>, %arg1: memref<128x1024x14x14xf32>, %arg2: memref<128x1024x14x14xf32>) {
    return
  }
  func @mgpuMemDealloc4DFloat(memref<?x?x?x?xf32>)
  func @conv2d_host(%arg0: memref<1024x1024x1x1xf32>, %arg1: memref<128x1024x14x14xf32>, %arg2: memref<128x1024x14x14xf32>) {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c107_i8 = constant 107 : i8
    %c99_i8 = constant 99 : i8
    %c121_i8 = constant 121 : i8
    %c120_i8 = constant 120 : i8
    %c110_i8 = constant 110 : i8
    %c104_i8 = constant 104 : i8
    %c119_i8 = constant 119 : i8
    %0 = memref_cast %arg0 : memref<1024x1024x1x1xf32> to memref<*xf32>
    %1 = memref_cast %arg1 : memref<128x1024x14x14xf32> to memref<*xf32>
    %2 = memref_cast %arg2 : memref<128x1024x14x14xf32> to memref<*xf32>
    %3 = alloca() : memref<4xi8>
    %4 = alloca() : memref<4xi8>
    %5 = alloca() : memref<4xi8>
    store %c107_i8, %3[%c0] : memref<4xi8>
    store %c99_i8, %3[%c1] : memref<4xi8>
    store %c121_i8, %3[%c2] : memref<4xi8>
    store %c120_i8, %3[%c3] : memref<4xi8>
    store %c110_i8, %4[%c0] : memref<4xi8>
    store %c99_i8, %4[%c1] : memref<4xi8>
    store %c104_i8, %4[%c2] : memref<4xi8>
    store %c119_i8, %4[%c3] : memref<4xi8>
    store %c110_i8, %5[%c0] : memref<4xi8>
    store %c107_i8, %5[%c1] : memref<4xi8>
    store %c104_i8, %5[%c2] : memref<4xi8>
    store %c119_i8, %5[%c3] : memref<4xi8>
    %6 = memref_cast %3 : memref<4xi8> to memref<*xi8>
    %7 = memref_cast %4 : memref<4xi8> to memref<*xi8>
    %8 = memref_cast %5 : memref<4xi8> to memref<*xi8>
    call @mcpuConv2d(%0, %1, %2, %6, %7, %8, %c1_i32, %c1_i32, %c0_i32, %c0_i32, %c1_i32, %c1_i32) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32) -> ()
    return
  }
  func @mcpuConv2d(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32)
  func @verify_results(%arg0: memref<128x1024x14x14xf32>, %arg1: memref<128x1024x14x14xf32>) {
    %c0 = constant 0 : index
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c1 = constant 1 : index
    %c128 = constant 128 : index
    %c1024 = constant 1024 : index
    %c14 = constant 14 : index
    %0 = alloca() : memref<1xi32>
    store %c1_i32, %0[%c0] : memref<1xi32>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c1024 step %c1 {
        scf.for %arg4 = %c0 to %c14 step %c1 {
          scf.for %arg5 = %c0 to %c14 step %c1 {
            %2 = load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<128x1024x14x14xf32>
            %3 = load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<128x1024x14x14xf32>
            %4 = cmpf "une", %2, %3 : f32
            scf.if %4 {
              store %c0_i32, %0[%c0] : memref<1xi32>
            }
          }
        }
      }
    }
    %1 = memref_cast %0 : memref<1xi32> to memref<*xi32>
    call @print_memref_i32(%1) : (memref<*xi32>) -> ()
    return
  }
  func @print_memref_i32(memref<*xi32>)
}
