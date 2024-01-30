// RUN: rocmlir-opt --rock-sugar-to-loops %s | FileCheck %s
// CHECK-LABEL: rock_extract_multibuffer
// CHECK-SAME: (%[[idx:.*]]: index, %[[buf0:.*]]: memref<1024xf16>, %[[buf1:.*]]: memref<1024xf16>, %[[buf2:.*]]: memref<1024xf16>) -> memref<1024xf16>
func.func @rock_extract_multibuffer(%idx : index, %buf0 : memref<1024xf16>, %buf1 : memref<1024xf16>, %buf2 : memref<1024xf16>) -> memref<1024xf16>{
  // CHECK:  %[[mbIdx:.*]] = arith.remui %[[idx]], %c3
  // CHECK:  %[[cmp0:.*]] = arith.cmpi ugt, %[[mbIdx]], %c1
  // CHECK:  %[[sel0:.*]] = arith.select %[[cmp0]], %[[buf2]], %[[buf1]]
  // CHECK:  %[[cmp1:.*]] = arith.cmpi ugt, %[[mbIdx]], %c0
  // CHECK:  %[[sel1:.*]] = arith.select %[[cmp1]], %[[sel0]], %[[buf0]]
  %s = rock.extract_multibuffer(%buf0, %buf1, %buf2)[%idx] (memref<1024xf16>, memref<1024xf16>, memref<1024xf16>) : memref<1024xf16>
  // CHECK: return %[[sel1]]
  return %s : memref<1024xf16>
}
