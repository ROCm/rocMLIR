// RUN: rocmlir-opt %s --rock-pipeline="rock-pipeline-remove-stages=false" | FileCheck %s

// Test rock-pipeline behavior with various configurations.
//
// Note: Compute-first scheduling (reordering stages to put MMA first) is
// currently DISABLED due to breaking the prologue generation in the MLIR
// pipelining transformation. When enabled, it would apply when:
// 1. Function has kernel attribute
// 2. Double buffering is enabled (II == 1)
// 3. blockSize == 8 * waveSize (512 = 8 * 64 for gfx90a)
//
// Since it's disabled, all configurations use the standard memory-first
// stage ordering: [GlobalRead, LDSWrite, LDSRead, MMA]

// CHECK-LABEL: pipeline_8wave_double_buffer
// Even with 8 waves + double buffer, stages remain in memory-first order
// because compute-first scheduling is disabled
// CHECK: scf.for
// CHECK: name = "GlobalRead"
// CHECK: name = "LDSWrite"
func.func @pipeline_8wave_double_buffer(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 512 : i32, grid_size = 1 : i32, arch = "gfx90a", kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %regGlobal = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regLds = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regMma = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>

    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %tmp, %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        %tmp = memref.load %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="LDSWrite"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        memref.store %tmp, %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
      rock.stage {
        %tmp = memref.load %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %comp = arith.addf %tmp, %c2 : f16
        memref.store %comp, %regMma[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="MMA"}
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %regMma[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// Test with 4-wave config (blockSize != 8 * waveSize)
// blockSize = 256 = 4 * 64 (4 waves)
// Uses memory-first order: [GlobalRead, LDSWrite, LDSRead, MMA]

// CHECK-LABEL: pipeline_4wave_double_buffer
// Memory-first order is used
// CHECK: scf.for
// CHECK: name = "GlobalRead"
// CHECK: name = "LDSWrite"
func.func @pipeline_4wave_double_buffer(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 256 : i32, grid_size = 1 : i32, arch = "gfx90a", kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %regGlobal = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regLds = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regMma = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>

    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %tmp, %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        %tmp = memref.load %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="LDSWrite"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        memref.store %tmp, %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
      rock.stage {
        %tmp = memref.load %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %comp = arith.addf %tmp, %c2 : f16
        memref.store %comp, %regMma[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="MMA"}
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %regMma[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// Test with II=2 (single buffering)
// blockSize = 512 but II = 2 (single buffering)

// CHECK-LABEL: pipeline_single_buffer
// With II=2, stages stay in original memory-first order
// CHECK: scf.for
// CHECK: name = "GlobalRead"
// CHECK: name = "LDSWrite"
func.func @pipeline_single_buffer(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 512 : i32, grid_size = 1 : i32, arch = "gfx90a", kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %regGlobal = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regLds = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regMma = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>

    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %tmp, %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        %tmp = memref.load %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="LDSWrite"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        memref.store %tmp, %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
      rock.stage {
        %tmp = memref.load %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %comp = arith.addf %tmp, %c2 : f16
        memref.store %comp, %regMma[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="MMA"}
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %regMma[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// Test without kernel attribute
// This verifies behavior for non-kernel functions

// CHECK-LABEL: pipeline_no_kernel_attr
// Without kernel attribute, memory-first order is used
// CHECK: scf.for
// CHECK: name = "GlobalRead"
// CHECK: name = "LDSWrite"
func.func @pipeline_no_kernel_attr(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 512 : i32, grid_size = 1 : i32, arch = "gfx90a"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %regGlobal = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regLds = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %regMma = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>

    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %tmp, %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        %tmp = memref.load %regGlobal[%arg3] : memref<16xf16, #gpu.address_space<private>>
        memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="LDSWrite"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        memref.store %tmp, %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
      rock.stage {
        %tmp = memref.load %regLds[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %comp = arith.addf %tmp, %c2 : f16
        memref.store %comp, %regMma[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="MMA"}
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %regMma[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}
