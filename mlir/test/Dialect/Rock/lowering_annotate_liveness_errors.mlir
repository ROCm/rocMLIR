// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-annotate-liveness -verify-diagnostics

#wg = #gpu.address_space<workgroup>

func.func @non_closed_read_write_pattern() attributes{arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  %0 = rock.alloc() : memref<1024xi8, #wg>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // expected-error @+1 {{Found a non closed read-write pattern}}
  %2 = rock.alloc() : memref<2048xi8, #wg>
  %c0 = arith.constant 0 : index
  %view0 = memref.view %0[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view1 = memref.view %1[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view2 = memref.view %2[%c0][] : memref<2048xi8, #wg> to memref<512xf32, #wg>
  %cst_0 = arith.constant 0.000000e+00 : f32

  rock.in_bounds_store %cst_0 -> %view0[%c0] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view1[%c0] : f32 -> memref<256xf32, #wg>, index
  %load0 = rock.in_bounds_load %view0[%c0] : memref<256xf32, #wg>, index -> f32
  %load1 = rock.in_bounds_load %view1[%c0] : memref<256xf32, #wg>, index -> f32

  rock.in_bounds_store %cst_0 -> %view2[%c0] : f32 -> memref<512xf32, #wg>, index
  
  return
}

func.func @read_before_write() attributes{arch = "##TOKEN_ARCH##", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  %0 = rock.alloc() : memref<1024xi8, #wg>
  %1 = rock.alloc() : memref<1024xi8, #wg>
  // expected-error @+1 {{Read before write}}
  %2 = rock.alloc() : memref<2048xi8, #wg>
  %c0 = arith.constant 0 : index
  %view0 = memref.view %0[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view1 = memref.view %1[%c0][] : memref<1024xi8, #wg> to memref<256xf32, #wg>
  %view2 = memref.view %2[%c0][] : memref<2048xi8, #wg> to memref<512xf32, #wg>
  %cst_0 = arith.constant 0.000000e+00 : f32

  rock.in_bounds_store %cst_0 -> %view0[%c0] : f32 -> memref<256xf32, #wg>, index
  rock.in_bounds_store %cst_0 -> %view1[%c0] : f32 -> memref<256xf32, #wg>, index
  %load0 = rock.in_bounds_load %view0[%c0] : memref<256xf32, #wg>, index -> f32
  %load1 = rock.in_bounds_load %view1[%c0] : memref<256xf32, #wg>, index -> f32

  %load2 = rock.in_bounds_load %view2[%c0] : memref<512xf32, #wg>, index -> f32
  rock.in_bounds_store %cst_0 -> %view2[%c0] : f32 -> memref<512xf32, #wg>, index
  
  return
}
