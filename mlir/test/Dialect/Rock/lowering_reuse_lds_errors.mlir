// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-reuse-lds -split-input-file -verify-diagnostics

// Test: LDS buffer element type must be i8
#wg = #gpu.address_space<workgroup>
func.func @rock_reuse_lds_non_i8_type() attributes{rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @below {{ReuseLDS: LDS buffer element type must be i8, but it's 'f32'}}
  %0 = rock.alloc() : memref<256xf32, #wg>
  %1 = rock.alloc() : memref<256xf32, #wg>
  rock.live_in %0 : memref<256xf32, #wg>
  rock.live_out %0 : memref<256xf32, #wg>
  rock.live_in %1 : memref<256xf32, #wg>
  rock.live_out %1 : memref<256xf32, #wg>
  return
}

// -----

// Test: Rank should be 1
#wg = #gpu.address_space<workgroup>
func.func @rock_reuse_lds_rank_not_1() attributes{rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // expected-error @below {{ReuseLDS: rank should be 1, but it's 2}}
  %0 = rock.alloc() : memref<32x32xi8, #wg>
  %1 = rock.alloc() : memref<32x32xi8, #wg>
  rock.live_in %0 : memref<32x32xi8, #wg>
  rock.live_out %0 : memref<32x32xi8, #wg>
  rock.live_in %1 : memref<32x32xi8, #wg>
  rock.live_out %1 : memref<32x32xi8, #wg>
  return
}

// -----

// Test: ReuseLDS requires too much LDS memory. In this case, two interfering
// allocations that total 262144 bytes, exceeding gfx950's 163840 byte LDS limit
#wg = #gpu.address_space<workgroup>
// expected-error @below {{ReuseLDS requires too much LDS memory: 262144 bytes (hardware max for amdgcn-amd-amdhsa:gfx950 is 163840 bytes)}}
func.func @rock_reuse_lds_too_much_lds() attributes{rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
  %0 = rock.alloc() : memref<131072xi8, #wg>
  %1 = rock.alloc() : memref<131072xi8, #wg>
  rock.live_in %0 : memref<131072xi8, #wg>
  rock.live_in %1 : memref<131072xi8, #wg>
  rock.live_out %0 : memref<131072xi8, #wg>
  rock.live_out %1 : memref<131072xi8, #wg>
  return
}

