// RUN: rocmlir-opt -split-input-file -verify-diagnostics %s

// Error case: Invalid MFMA geometry (16x8 is not valid)
// This tests that LDSTransposeConfigAttr::verify() catches invalid MFMA
// geometry combinations. Valid combinations are: (16,16), (16,32), (16,64),
// (16,128), (32,8), (32,16), (32,32), (32,64)
func.func @threadwise_read_into_invalid_mfma_geometry_16x8(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<8xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{invalid MFMA geometry (16x8) for LDS transpose - valid combinations: (16,16), (16,32), (16,64), (16,128), (32,8), (32,16), (32,32), (32,64)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 8,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: Invalid MFMA geometry (16x256 is not valid)
func.func @threadwise_read_into_invalid_mfma_geometry_16x256(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<8xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{invalid MFMA geometry (16x256) for LDS transpose - valid combinations: (16,16), (16,32), (16,64), (16,128), (32,8), (32,16), (32,32), (32,64)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 256,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 256,
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
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{invalid MFMA geometry (8x8) for LDS transpose - valid combinations: (16,16), (16,32), (16,64), (16,128), (32,8), (32,16), (32,32), (32,64)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 8, kDim = 8,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: kPerBlock not divisible by kDim
func.func @threadwise_read_into_kperblock_not_divisible(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<8xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{kPerBlock (30) must be divisible by kDim (16)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 16,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 30,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<8xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: kPerBlock not divisible by kDim, INT8 double-rate (32x32).
// Exercises the divisibility check on an INT8-only geometry so that the
// path is covered for kDim=32 as well.
func.func @threadwise_read_into_kperblock_not_divisible_int8(
    %source: memref<128xi8, #gpu.address_space<workgroup>>,
    %dest: memref<16xi8, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  rock.threadwise_read_into {
    // expected-error @+1 {{kPerBlock (50) must be divisible by kDim (32)}}
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 32, kDim = 32,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 50,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xi8, #gpu.address_space<workgroup>> -> memref<16xi8, #gpu.address_space<private>>
  return
}

// -----

// Error case: LDS transpose load not supported on gfx942
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx942"} {
  func.func @lds_transpose_load_unsupported_arch(%src: memref<128x256xf16, #gpu.address_space<workgroup>>, %i: index, %j: index) -> vector<4xf16> {
    // expected-error @+1 {{LDS transpose load is not supported on this architecture: amdgcn-amd-amdhsa:gfx942}}
    %v = rock.lds_transpose_load %src[%i, %j] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
    return %v : vector<4xf16>
  }
}

// -----

// Error case: FP8 destination with F16-only geometry (16x16).
// FP8/BF8 only support quad-rate geometries (16x128, 32x64) and the standard
// (16,32) / (32,16) shared with F16/INT8, never the F16-only (16,16) / (32,8).
func.func @threadwise_read_into_fp8_with_f16_only_geometry(
    %source: memref<128xf8E4M3FN, #gpu.address_space<workgroup>>,
    %dest: memref<8xf8E4M3FN, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{MFMA geometry (16x16) is not supported for FP8/BF8/INT8}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 16,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf8E4M3FN, #gpu.address_space<workgroup>> -> memref<8xf8E4M3FN, #gpu.address_space<private>>
  return
}

// -----

// Error case: INT8 destination with F16-only geometry (32x8).
// F16-only geometries (16,16) / (32,8) are not used by any FP8/BF8/INT8 MFMA
// instruction.
func.func @threadwise_read_into_int8_with_f16_only_geometry(
    %source: memref<128xi8, #gpu.address_space<workgroup>>,
    %dest: memref<8xi8, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{MFMA geometry (32x8) is not supported for FP8/BF8/INT8}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 32, kDim = 8,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 16,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xi8, #gpu.address_space<workgroup>> -> memref<8xi8, #gpu.address_space<private>>
  return
}

// -----

// Error case: F16 destination with quad-rate FP8 geometry (16x128).
// Quad-rate geometries (16x128, 32x64) are FP8/BF8 only and must not be used
// with 16-bit or INT8 element types.
func.func @threadwise_read_into_f16_with_quad_rate_geometry(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<4xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{quad-rate MFMA geometry (16x128) is only valid for FP8/BF8}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 128,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 128,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<4xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: INT8 destination with quad-rate FP8 geometry (32x64).
// Quad-rate geometries are FP8/BF8 only.
func.func @threadwise_read_into_int8_with_quad_rate_geometry(
    %source: memref<128xi8, #gpu.address_space<workgroup>>,
    %dest: memref<8xi8, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{quad-rate MFMA geometry (32x64) is only valid for FP8/BF8}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 32, kDim = 64,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 64,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xi8, #gpu.address_space<workgroup>> -> memref<8xi8, #gpu.address_space<private>>
  return
}

// -----

// Error case: F16 destination with INT8-only double-rate geometry (16x64).
// Double-rate geometries (16,64) and (32,32) are INT8 only.
func.func @threadwise_read_into_f16_with_int8_only_geometry(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<4xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{double-rate MFMA geometry (16x64) is only valid for INT8}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 64,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 64,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<4xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: FP8 destination with INT8-only double-rate geometry (32x32).
// Double-rate geometries (16,64) and (32,32) are INT8 only.
func.func @threadwise_read_into_fp8_with_int8_only_geometry(
    %source: memref<128xf8E5M2, #gpu.address_space<workgroup>>,
    %dest: memref<8xf8E5M2, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{double-rate MFMA geometry (32x32) is only valid for INT8}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 32, kDim = 32,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf8E5M2, #gpu.address_space<workgroup>> -> memref<8xf8E5M2, #gpu.address_space<private>>
  return
}

// -----

// Error case: unsupported destination element type (f32).
// ldsTransposeConfig only supports f16, bf16, f8E4M3FN, f8E5M2, or i8.
func.func @threadwise_read_into_unsupported_dest_type(
    %source: memref<128xf32, #gpu.address_space<workgroup>>,
    %dest: memref<4xf32, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ldsTransposeConfig only supports f16, bf16, f8E4M3FN, f8E5M2, or i8 destination element types}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 16,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf32, #gpu.address_space<workgroup>> -> memref<4xf32, #gpu.address_space<private>>
  return
}

// -----

// Error case: destination must be rank-1 with a static shape.
// A rank-2 destination is rejected even when the geometry would otherwise be
// valid.
func.func @threadwise_read_into_rank2_dest(
    %source: memref<128xf16, #gpu.address_space<workgroup>>,
    %dest: memref<2x4xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ldsTransposeConfig requires a rank-1 destination with a static shape}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 16,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16, #gpu.address_space<workgroup>> -> memref<2x4xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: source memref is not in workgroup memory.
// LDS transpose load operates on LDS, so the source must live in workgroup
// address space. A source without an explicit memory space (defaults to
// global) is rejected.
func.func @threadwise_read_into_non_lds_source(
    %source: memref<128xf16>,
    %dest: memref<4xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ldsTransposeConfig requires the source memref to live in workgroup (LDS) memory}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 16,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xf16> -> memref<4xf16, #gpu.address_space<private>>
  return
}

// -----

// Error case: source and dest element types do not match. ds_read_tr*_b64
// reads the LDS source into the destination at the same element type, so a
// type mismatch is rejected here rather than failing later in
// LDSTransposeLoadOp::verify.
func.func @threadwise_read_into_src_dest_type_mismatch(
    %source: memref<128xbf16, #gpu.address_space<workgroup>>,
    %dest: memref<4xf16, #gpu.address_space<private>>)
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{ldsTransposeConfig requires the source and dest element types to match, but got source 'bf16' vs dest 'f16'}}
  rock.threadwise_read_into {
    ldsTransposeConfig = #rock.lds_transpose_config<
      dDim = 16, kDim = 16,
      mPerBlock = 128, nPerBlock = 128, kPerBlock = 32,
      mPerWave = 64, nPerWave = 64,
      doubleBuffering = false, isOperandA = true
    >
  } [] (%source) [] -> %dest : memref<128xbf16, #gpu.address_space<workgroup>> -> memref<4xf16, #gpu.address_space<private>>
  return
}
