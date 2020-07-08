// compilation process:
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_64_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_32_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_64_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_32_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_16_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_16_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_64_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_8_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_4_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -mlir-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908

func @mfma_64_64(%a : f32, %b : f32, %memref_c : memref<64xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 64 } : f32, f32, memref<64xf32>
  return
}

func @mfma_32_64(%a : f32, %b : f32, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 64 } : f32, f32, memref<32xf32>
  return
}

func @mfma_64_32(%a : f32, %b : f32, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 32 } : f32, f32, memref<32xf32>
  return
}

func @mfma_32_32(%a : f32, %b : f32, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 32 } : f32, f32, memref<16xf32>
  return
}

func @mfma_16_16(%a : f32, %b : f32, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 16 } : f32, f32, memref<4xf32>
  return
}

func @mfma_16_64(%a : f32, %b : f32, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 64 } : f32, f32, memref<16xf32>
  return
}

func @mfma_64_16(%a : f32, %b : f32, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 16 } : f32, f32, memref<16xf32>
  return
}

func @mfma_8_64(%a : f32, %b : f32, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 8, n_per_wave = 64 } : f32, f32, memref<4xf32>
  return
}

func @mfma_4_64(%a : f32, %b : f32, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 4, n_per_wave = 64 } : f32, f32, memref<4xf32>
  return
}
