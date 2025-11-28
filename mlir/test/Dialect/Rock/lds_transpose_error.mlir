// RUN: rocmlir-opt -split-input-file -verify-diagnostics %s

// Error case: Invalid MFMA geometry (16x8 is not valid)
// This tests that LdsTransposeConfigAttr::verify() catches invalid MFMA
// geometry combinations. Valid combinations are: (16,16), (16,32), (32,8), (32,16)
func.func @threadwise_read_into_invalid_mfma_geometry_16x8(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<8xf16, #gpu.address_space<private>>)
    attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{invalid MFMA geometry (16x8) for LDS transpose - valid combinations: (16,16), (16,32), (32,8), (32,16)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      mfmaDDim = 16, mfmaKDim = 8,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: Invalid MFMA geometry (32x32 is not valid)
func.func @threadwise_read_into_invalid_mfma_geometry_32x32(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<8xf16, #gpu.address_space<private>>)
    attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{invalid MFMA geometry (32x32) for LDS transpose - valid combinations: (16,16), (16,32), (32,8), (32,16)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      mfmaDDim = 32, mfmaKDim = 32,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: Invalid MFMA geometry (8x8 is not valid)
func.func @threadwise_read_into_invalid_mfma_geometry_8x8(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<8xf16, #gpu.address_space<private>>)
    attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{invalid MFMA geometry (8x8) for LDS transpose - valid combinations: (16,16), (16,32), (32,8), (32,16)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      mfmaDDim = 8, mfmaKDim = 8,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: kPerBlock not divisible by mfmaKDim
func.func @threadwise_read_into_kperblock_not_divisible(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<8xf16, #gpu.address_space<private>>)
    attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{kPerBlock (30) must be divisible by mfmaKDim (16)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      mfmaDDim = 16, mfmaKDim = 16,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 30,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
  return
}
