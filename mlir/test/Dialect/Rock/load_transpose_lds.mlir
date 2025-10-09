// RUN: rocmlir-opt --rock-sugar-to-loops %s | FileCheck %s 

// CHECK-LABEL: func @test_load_transpose_fp16
module {
  func.func @test_load_transpose_fp16(%src: memref<128x256xf16, 3>, %i: index, %j: index) -> vector<4xf16> {
    // CHECK: amdgpu.transpose_load %arg0[%arg1, %arg2] : memref<128x256xf16, 3> -> vector<4xf16> 
    %v = rock.lds_transpose_load %src[%i, %j] : memref<128x256xf16, 3> -> vector<4xf16>
    return %v : vector<4xf16>
  }

// CHECK-LABEL: func @test_load_transpose_bf16
  func.func @test_load_transpose_bf16(%src: memref<64x128xbf16, 3>, %i: index, %j: index) -> vector<4xbf16> {
    // CHECK: amdgpu.transpose_load %arg0[%arg1, %arg2] : memref<64x128xbf16, 3> -> vector<4xbf16>
    %v = rock.lds_transpose_load %src[%i, %j] : memref<64x128xbf16, 3> -> vector<4xbf16>
    return %v : vector<4xbf16>
  }
}
