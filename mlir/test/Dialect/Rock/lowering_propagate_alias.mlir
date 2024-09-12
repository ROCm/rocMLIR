// RUN: rocmlir-opt -rock-propagate-alias %s | FileCheck %s

// CHECK-LABEL: func.func @rock_alias
func.func @rock_alias() attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  %0 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>

  %c0 = arith.constant 0 : index
  // CHECK: %[[NOALIASVIEW1:.*]] = memref.view %0[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
  %1 = rock.noalias_view %0[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[NOALIASVIEW1]][%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  %view_1 = memref.view %1[%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<8xf16>, #gpu.address_space<workgroup>>

  %c4096 = arith.constant 4096 : index
  // CHECK: %[[NOALIASVIEW2:.*]] = memref.view %0[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
  %2 = rock.noalias_view %0[%c4096][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xi8, #gpu.address_space<workgroup>>
  // CHECK-NEXT: %[[VIEW2:.*]] = memref.view %[[NOALIASVIEW2]][%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  %view_2 = memref.view %2[%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  
  %reg = rock.alloc() : memref<8xf16, #gpu.address_space<private>>
  // CHECK: %[[REGVAR:.*]] = vector.load %[[REGALLOC:.*]][%c0] : memref<8xf16, #gpu.address_space<private>>, vector<8xf16>
  %reg_load = vector.load %reg[%c0] : memref<8xf16, #gpu.address_space<private>>, vector<8xf16>
  // CHECK: memref.store %[[REGVAR]], %[[VIEW1]][%c0] {alias_scopes = [#[[ALIAS0:.*]]], noalias_scopes = [#[[ALIAS1:.*]]]} : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  memref.store %reg_load, %view_1[%c0] : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  
  // CHECK: memref.load %[[VIEW1]][%c0] {alias_scopes = [#[[ALIAS0]]], noalias_scopes = [#[[ALIAS1]]]} : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  %load_view1 = memref.load %view_1[%c0] : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>

  // CHECK: memref.store %[[REGVAR]], %[[VIEW2]][%c0] {alias_scopes = [#[[ALIAS1]]], noalias_scopes = [#[[ALIAS0]]]} : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  memref.store %reg_load, %view_2[%c0] : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  
  // CHECK: memref.load %[[VIEW2]][%c0] {alias_scopes = [#[[ALIAS1]]], noalias_scopes = [#[[ALIAS0]]]} : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
  %load_view2 = memref.load %view_2[%c0] : memref<256xvector<8xf16>, #gpu.address_space<workgroup>>

  rock.dealloc %0 : memref<8192xi8, #gpu.address_space<workgroup>>

  return
}
