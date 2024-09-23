// RUN: rocmlir-opt -rock-linalg-align -canonicalize %s | FileCheck %s

!tile = memref<16xf16, #gpu.address_space<private>>
!in_global = memref<4xf16>
!out_global = memref<16xf16>
#map = affine_map<(d0) -> (d0)>
#transform_map = #rock.transform_map<#map by [<Pad{0, 12} ["xPad"] at [0] -> ["x"] at [0]>] bounds = [16] -> [4]>

// CHECK-LABEL: @must_reapply_padding
// CHECK-SAME: (%[[arg0:.+]]: memref<4xf16>, %[[arg1:.+]]: memref<16xf16>)
// CHECK: %[[padded:.+]] = rock.transform %[[arg0]]
// CHECK: %[[validity:.+]] = rock.threadwise_read_into [](%[[padded]]) -> %[[globalIn:.+]] :
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[globalIn]] : memref<{{[^>]+}}>>)
// CHECK-SAME: outs(%[[unmaskedOut:.+]] : memref<{{[^>]+}}>>)
// CHECK: rock.threadwise_read_into [](%[[unmaskedOut]]) -> %[[maskedOut:.+]] if [%[[validity]]]
// CHECK: rock.threadwise_write_all features = none %[[maskedOut]]
func.func @must_reapply_padding(%arg0: !in_global, %arg1: !out_global) attributes {kernel} {
  %cst = arith.constant 4.0 : f16
  %alloc = memref.alloc() : !in_global
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : !in_global) outs (%alloc : !in_global) {
  ^bb0(%arg2: f16, %arg3: f16):
    %0 = arith.addf %arg2, %cst : f16
    linalg.yield %0 : f16
  }
  %1 = rock.transform %alloc by #transform_map : !in_global to !out_global
  %buf = rock.alloc() : !tile
  %2 = rock.threadwise_read_into [](%1) -> %buf : !out_global -> !tile, vector<16xi1>
  rock.threadwise_write_all features = none %buf -> [](%arg1) by set : !tile -> !out_global
  return
}

// CHECK-LABEL: @doesnt_reapply_padding
// CHECK-SAME: (%[[arg0:.+]]: memref<4xf16>, %[[arg1:.+]]: memref<16xf16>)
// CHECK: %[[padded:.+]] = rock.transform %[[arg0]]
// CHECK: rock.threadwise_read_into [](%[[padded]]) -> %[[globalIn:.+]] :
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[globalIn]] : memref<{{[^>]+}}>>)
// CHECK-SAME: outs(%[[unmaskedOut:.+]] : memref<{{[^>]+}}>>)
// CHECK: rock.threadwise_write_all features = none %[[unmaskedOut]]
func.func @doesnt_reapply_padding(%arg0: !in_global, %arg1: !out_global) attributes {kernel} {
  %cst = arith.constant 4.0 : f16
  %alloc = memref.alloc() : !in_global
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : !in_global) outs (%alloc : !in_global) {
  ^bb0(%arg2: f16, %arg3: f16):
    %0 = arith.mulf %arg2, %cst : f16
    linalg.yield %0 : f16
  }
  %1 = rock.transform %alloc by #transform_map : !in_global to !out_global
  %buf = rock.alloc() : !tile
  %2 = rock.threadwise_read_into [](%1) -> %buf : !out_global -> !tile, vector<16xi1>
  rock.threadwise_write_all features = none %buf -> [](%arg1) by set : !tile -> !out_global
  return
}
