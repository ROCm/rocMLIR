// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_64_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_32_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_64_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_32_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_16_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_16_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_64_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_8_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f32_4_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908

func @mfma_f32_64_64(%a : f32, %b : f32, %memref_c : memref<64xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 64 } : f32, memref<64xf32>
  return
}

func @mfma_f32_32_64(%a : f32, %b : f32, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 64 } : f32, memref<32xf32>
  return
}

func @mfma_f32_64_32(%a : f32, %b : f32, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 32 } : f32, memref<32xf32>
  return
}

func @mfma_f32_32_32(%a : f32, %b : f32, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 32 } : f32, memref<16xf32>
  return
}

func @mfma_f32_16_16(%a : f32, %b : f32, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 16 } : f32, memref<4xf32>
  return
}

func @mfma_f32_16_64(%a : f32, %b : f32, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 64 } : f32, memref<16xf32>
  return
}

func @mfma_f32_64_16(%a : f32, %b : f32, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 16 } : f32, memref<16xf32>
  return
}

func @mfma_f32_8_64(%a : f32, %b : f32, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 8, n_per_wave = 64 } : f32, memref<4xf32>
  return
}

func @mfma_f32_4_64(%a : f32, %b : f32, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 4, n_per_wave = 64 } : f32, memref<4xf32>
  return
}

// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_64_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_32_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_64_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_32_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_16_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_16_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_64_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_8_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_f16_4_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908

func @mfma_f16_64_64(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<64xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 64 } : vector<4xf16>, memref<64xf32>
  return
}

func @mfma_f16_32_64(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 64 } : vector<4xf16>, memref<32xf32>
  return
}

func @mfma_f16_64_32(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 32 } : vector<4xf16>, memref<32xf32>
  return
}

func @mfma_f16_32_32(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 32 } : vector<4xf16>, memref<16xf32>
  return
}

func @mfma_f16_16_16(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 16 } : vector<4xf16>, memref<4xf32>
  return
}

func @mfma_f16_16_64(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 64 } : vector<4xf16>, memref<16xf32>
  return
}

func @mfma_f16_64_16(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 16 } : vector<4xf16>, memref<16xf32>
  return
}

func @mfma_f16_8_64(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 8, n_per_wave = 64 } : vector<4xf16>, memref<4xf32>
  return
}

func @mfma_f16_4_64(%a : vector<4xf16>, %b : vector<4xf16>, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 4, n_per_wave = 64 } : vector<4xf16>, memref<4xf32>
  return
}

// TBD: mlir-translate for BF16 needs to be investigated.
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_64_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_32_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_64_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_32_32" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_16_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_16_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_64_16" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_8_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908
// mlir-opt -convert-miopen-to-gpu="kernel-name=mfma_bf16_4_64" -convert-gpu-to-rocdl xdlops.mlir | mlir-translate -gpu-module-to-rocdlir | opt -O3 -S -strip | llc -mcpu=gfx908

func @mfma_bf16_64_64(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<64xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 64 } : vector<2xbf16>, memref<64xf32>
  return
}

func @mfma_bf16_32_64(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 64 } : vector<2xbf16>, memref<32xf32>
  return
}

func @mfma_bf16_64_32(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<32xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 32 } : vector<2xbf16>, memref<32xf32>
  return
}

func @mfma_bf16_32_32(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 32, n_per_wave = 32 } : vector<2xbf16>, memref<16xf32>
  return
}

func @mfma_bf16_16_16(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 16 } : vector<2xbf16>, memref<4xf32>
  return
}

func @mfma_bf16_16_64(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 16, n_per_wave = 64 } : vector<2xbf16>, memref<16xf32>
  return
}

func @mfma_bf16_64_16(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<16xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 64, n_per_wave = 16 } : vector<2xbf16>, memref<16xf32>
  return
}

func @mfma_bf16_8_64(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 8, n_per_wave = 64 } : vector<2xbf16>, memref<4xf32>
  return
}

func @mfma_bf16_4_64(%a : vector<2xbf16>, %b : vector<2xbf16>, %memref_c : memref<4xf32>) {
  miopen.mfma(%a, %b, %memref_c) { m_per_wave = 4, n_per_wave = 64 } : vector<2xbf16>, memref<4xf32>
  return
}

