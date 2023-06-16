// RUN: rocmlir-opt -fp8-ext-to-tables -split-input-file %s | FileCheck %s

module {
  func.func @ext_scalar(%arg0: f8E5M2FNUZ) -> f16 {
    // CHECK-LABEL: @ext_scalar
    // CHECK-SAME: ([[ARG0:%.+]]: f8E5M2FNUZ)
    // CHECK: [[TABLE:%.+]] = memref.get_global @__rocmlir_extf_tbl_f8E5M2FNUZ : memref<256xf32>
    // CHECK: [[BYTE:%.+]] = arith.bitcast [[ARG0]] : f8E5M2FNUZ to i8
    // CHECK: [[LONGBYTE:%.+]] = arith.extui [[BYTE]] : i8 to i32
    // CHECK: [[IDX:%.+]] = arith.index_cast [[LONGBYTE]] : i32 to index
    // CHECK: [[EXT:%.+]] = memref.load [[TABLE]]{{\[}}[[IDX]]]
    // CHECK: [[TRUNC:%.+]] = arith.truncf [[EXT]] : f32 to f16
    // CHECK: return [[TRUNC]]
    %ret = arith.extf %arg0 : f8E5M2FNUZ to f16
    return %ret : f16
  }
  // CHECK-LABEL: memref.global "private" constant @__rocmlir_extf_tbl_f8E5M2FNUZ
}

// -----

module {
  func.func @ext_vector(%arg0: vector<2x2xf8E4M3FNUZ>) -> vector<2x2xf32> {
  // CHECK-LABEL: @ext_vector
  // CHECK-SAME: ([[ARG0:%.+]]: vector<2x2xf8E4M3FNUZ>)
  // CHECK: [[TABLE:%.+]] = memref.get_global @__rocmlir_extf_tbl_f8E4M3FNUZ : memref<256xf32>
  // CHECK: [[RET0:%.+]] = arith.constant dense<0.000000e+00> : vector<2x2xf32>
  // CHECK: [[IN0:%.+]] = vector.extract [[ARG0]][0, 0]
  // CHECK: [[BYTE0:%.+]] = arith.bitcast [[IN0]] : f8E4M3FNUZ to i8
  // CHECK: [[LONGBYTE0:%.+]] = arith.extui [[BYTE0]] : i8 to i32
  // CHECK: [[IDX0:%.+]] = arith.index_cast [[LONGBYTE0]] : i32 to index
  // CHECK: [[EXT0:%.+]] = memref.load [[TABLE]]{{\[}}[[IDX0]]]
  // CHECK: [[RET1:%.+]] = vector.insert [[EXT0]], [[RET0]] [0, 0] : f32 into vector<2x2xf32>
  // ...
    %ret = arith.extf %arg0 : vector<2x2xf8E4M3FNUZ> to vector<2x2xf32>
    func.return %ret : vector<2x2xf32>
  }
}

// -----

// Test that the buffer gets inserted inside GPU modules if relevant.
module attributes {gpu.container_module} {
  gpu.module @kernel_mod {
    gpu.func @kernel(%arg0: f8E4M3FNUZ, %arg1: memref<1xf64>) kernel {
      %c0 = arith.constant 0 : index
      %ret = arith.extf %arg0 : f8E4M3FNUZ to f64
      memref.store %ret, %arg1[%c0] : memref<1xf64>
      gpu.return
    }
    // CHECK: gpu.return
    // CHECK-NEXT: }
    // CHECK-NEXT: memref.global "private" constant @__rocmlir_extf_tbl_f8E4M3FNUZ
    // CHECK-NEXT: }
    // CHECK-NEXT: }
  }
}
