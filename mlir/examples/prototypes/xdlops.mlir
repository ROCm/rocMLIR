// compilation process:
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_minimal" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908

func @mfma_minimal(%a : f32, %b : f32, %memref_c : memref<32xf32>) {
  %memref_vector_c = vector.type_cast %memref_c : memref<32xf32> to memref<vector<32xf32>>
  %vector_c = load %memref_vector_c[] : memref<vector<32xf32>>

  %c0 = constant 0 : i32
  %c1 = constant 1 : i32
  %vector_d = miopen.mfma(%a, %b, %vector_c, %c1, %c0, %c0) : vector<32xf32>
  store %vector_d, %memref_vector_c[] : memref<vector<32xf32>>

  return
}
