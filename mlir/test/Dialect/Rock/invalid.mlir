// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @gemm_too_big(%a: memref<2048x2048x2048xf32>,
                        %b: memref<2048x2048x2048xf32>,
                        %c: memref<2048x2048x2048xf32>) {
  // expected-error@+1 {{'rock.gemm' op matrix A cannot potentially be over 2 GB}}
  rock.gemm %c = %a * %b features = dot storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx1030",
    numCu = 64 : i32}
  : memref<2048x2048x2048xf32> = memref<2048x2048x2048xf32> * memref<2048x2048x2048xf32>
  func.return
}

// -----

func.func @gemm_c_big(%a: memref<2048x2048x1xf32>,
                      %b: memref<2048x1x2048xf32>,
                      %c: memref<2048x2048x2048xf32>) {
  // expected-error@+1 {{'rock.gemm' op matrix C cannot potentially be over 2 GB}}
  rock.gemm %c = %a * %b features = dot storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx1030",
    numCu = 64 : i32}
  : memref<2048x2048x2048xf32> = memref<2048x2048x1xf32> * memref<2048x1x2048xf32>
  func.return
}

// -----

func.func @gemm_c_too_big(%a: memref<2048x2048x2048xf32>,
                        %b: memref<2048x2048x2048xf32>,
                        %c: memref<2048x2048x2048xf32>) {
  // expected-error@+1 {{'rock.gemm' op matrix A cannot potentially be over 2 GB}}
  rock.gemm %c = %a * %b features = dot storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx1030",
    numCu = 64 : i32}
  : memref<2048x2048x2048xf32> = memref<2048x2048x2048xf32> * memref<2048x2048x2048xf32>
  func.return
}
