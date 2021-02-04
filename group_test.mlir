

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
    
    %2 = miopen.transform(%1) {intermediate_layout = ["gemmG","ni", "cgroup", "hipad", "wipad"], layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["ni"], source_dimensions = [1 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [2 : i32], names = ["cgroup"], source_dimensions = [1 : i32], source_names = ["cgroup"], transformation = "PassThrough"}, {dimensions = [3 : i32, 4 : i32], names = ["y", "ho"], parameters = [1 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [3 : i32], source_names = ["hipad"], transformation = "Embed"}, {dimensions = [5 : i32, 6 : i32], names = ["x", "wo"], parameters = [1 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [4 : i32], source_names = ["wipad"], transformation = "Embed"}], output_layout = ["gemmG","ni", "cgroup", "y", "ho", "x", "wo"]} : memref<1x128x1024x14x14xf32> to memref<1x128x1024x1x14x1x14xf32>

    %3 = miopen.transform(%2) {gridwise_gemm_argument_position = 1 : i32, intermediate_layout = ["ni", "ci", "y", "ho", "x", "wo"], layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [0 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmK"], source_dimensions = [2 : i32, 3 : i32, 4 : i32], source_names = ["cgroup", "y", "x"], transformation = "Merge"}, {dimensions = [2 : i32], names = ["gemmN"], source_dimensions = [1 : i32, 4 : i32, 6 : i32], source_names = ["ni", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmG","gemmK", "gemmN"]} : memref<1x128x1024x1x14x1x14xf32> to memref<1x1024x25088xf32>

    %4 = miopen.transform(%a2) {gridwise_gemm_argument_position = 2 : i32, layout = [{dimensions = [0 : i32], names = ["gemmG"], source_dimensions = [1 : i32], source_names = ["gemmG"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmM"], source_dimensions = [2 : i32], source_names = ["kgroup"], transformation = "PassThrough"}, {dimensions = [2 : i32], names = ["gemmN"], source_dimensions = [0 : i32, 3 : i32, 4 : i32], source_names = ["no", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmG", "gemmM", "gemmN"], source_layout = ["no", "g", "kgroup", "ho", "wo"]} : memref<128x1x1024x14x14xf32> to memref<1x1024x25088xf32>
    miopen.gridwise_gemm_v2(%0, %3, %4) {arch = "gfx908", dilations = [1 : i32, 1 : i32], filter_dimension = [1024, 1024, 1, 1], filter_layout = ["k", "c", "y", "x"], input_dimension = [128, 1024, 14, 14], input_layout = ["ni", "ci", "hi", "wi"], kernel_algorithm = "v4r4", num_cu = 120 : i32, output_dimension = [128, 1024, 14, 14], output_layout = ["no", "ko", "ho", "wo"], padding = [[0 : i32, 0 : i32], [0 : i32, 0 : i32]], strides = [1 : i32, 1 : i32], xdlopsV2 = true, num_group = 1:i32} : memref<1x1024x1024xf32>, memref<1x1024x25088xf32>, memref<1x1024x25088xf32>
    return
  }
  func @main() {
    %0 = alloc() : memref<1024x1024x1x1xf32>
    %1 = alloc() : memref<128x1024x14x14xf32>
    %2 = alloc() : memref<128x1024x14x14xf32>
    %3 = memref_cast %0 : memref<1024x1024x1x1xf32> to memref<?x?x?x?xf32>
    %4 = memref_cast %1 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    %5 = memref_cast %2 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 0.000000e+00 : f32
    call @mcpuMemset4DFloat(%3, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
    call @mcpuMemset4DFloat(%4, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
    call @mcpuMemset4DFloat(%5, %cst_0) : (memref<?x?x?x?xf32>, f32) -> ()
    call @gpu_conv(%0, %1, %2) : (memref<1024x1024x1x1xf32>, memref<128x1024x14x14xf32>, memref<128x1024x14x14xf32>) -> ()
    dealloc %0 : memref<1024x1024x1x1xf32>
    dealloc %1 : memref<128x1024x14x14xf32>
    dealloc %2 : memref<128x1024x14x14xf32>
    return
  }
  func @mcpuMemset4DFloat(memref<?x?x?x?xf32>, f32)
  func @gpu_conv(%arg0: memref<1024x1024x1x1xf32>, %arg1: memref<128x1024x14x14xf32>, %arg2: memref<128x1024x14x14xf32>) {
    %0 = memref_cast %arg0 : memref<1024x1024x1x1xf32> to memref<?x?x?x?xf32>
    %1 = memref_cast %arg1 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    %2 = memref_cast %arg2 : memref<128x1024x14x14xf32> to memref<?x?x?x?xf32>
    %3 = call @mgpuMemAlloc4DFloat(%0) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    %4 = call @mgpuMemAlloc4DFloat(%1) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    %5 = call @mgpuMemAlloc4DFloat(%2) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
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
}
