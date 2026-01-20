// RUN: rocmlir-opt --rock-sugar-to-loops %s | FileCheck %s 

// CHECK-LABEL: func @test_load_transpose_fp16
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
  func.func @test_load_transpose_fp16(%src: memref<128x256xf16, #gpu.address_space<workgroup>>, %i: index, %j: index) -> vector<4xf16> {
    // CHECK: amdgpu.transpose_load %arg0[%arg1, %arg2] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16> 
    %v = rock.lds_transpose_load %src[%i, %j] : memref<128x256xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
    return %v : vector<4xf16>
  }

// CHECK-LABEL: func @test_load_transpose_bf16
  func.func @test_load_transpose_bf16(%src: memref<64x128xbf16, #gpu.address_space<workgroup>>, %i: index, %j: index) -> vector<4xbf16> {
    // CHECK: amdgpu.transpose_load %arg0[%arg1, %arg2] : memref<64x128xbf16, #gpu.address_space<workgroup>> -> vector<4xbf16>
    %v = rock.lds_transpose_load %src[%i, %j] : memref<64x128xbf16, #gpu.address_space<workgroup>> -> vector<4xbf16>
    return %v : vector<4xbf16>
  }

// CHECK-LABEL: func @test_load_transpose_fp8_e4m3
  func.func @test_load_transpose_fp8_e4m3(%src: memref<128x256xf8E4M3FN, #gpu.address_space<workgroup>>, %i: index, %j: index) -> vector<8xf8E4M3FN> {
    // CHECK: amdgpu.transpose_load %arg0[%arg1, %arg2] : memref<128x256xf8E4M3FN, #gpu.address_space<workgroup>> -> vector<8xf8E4M3FN>
    %v = rock.lds_transpose_load %src[%i, %j] : memref<128x256xf8E4M3FN, #gpu.address_space<workgroup>> -> vector<8xf8E4M3FN>
    return %v : vector<8xf8E4M3FN>
  }

// CHECK-LABEL: func @test_load_transpose_fp8_e5m2
  func.func @test_load_transpose_fp8_e5m2(%src: memref<64x128xf8E5M2, #gpu.address_space<workgroup>>, %i: index, %j: index) -> vector<8xf8E5M2> {
    // CHECK: amdgpu.transpose_load %arg0[%arg1, %arg2] : memref<64x128xf8E5M2, #gpu.address_space<workgroup>> -> vector<8xf8E5M2>
    %v = rock.lds_transpose_load %src[%i, %j] : memref<64x128xf8E5M2, #gpu.address_space<workgroup>> -> vector<8xf8E5M2>
    return %v : vector<8xf8E5M2>
  }
}
