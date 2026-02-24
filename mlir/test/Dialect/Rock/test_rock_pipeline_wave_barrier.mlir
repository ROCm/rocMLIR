// RUN: rocmlir-opt %s --rock-pipeline="rock-pipeline-remove-stages=true" | FileCheck %s

// This test file verifies the optimization that skips backward LDS barriers
// for single-wave kernels with specific schedule versions.

// Test for single-wave kernel with scheduleVersion=1 (Default)
// When blockSize <= waveSize and scheduleVersion is 1 or 3, backward barriers should be skipped
// For scheduleVersion=1, the loop has 3 stages: GlobalRead, LDSWrite, LDSRead

// CHECK-LABEL: func.func @rock_pipeline_one_wave_schedule_v1
// Prologue stores to LDS:
// CHECK: memref.store {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: scf.for
// Inside loop - only ONE barrier (forward), no backward barrier for single-wave
// CHECK: rock.lds_barrier
// CHECK: memref.load {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: rock.threadwise_gemm_accel
// CHECK-NOT: rock.lds_barrier
// CHECK: memref.store {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: }
// Epilogue barrier and LDS read:
// CHECK: rock.lds_barrier
// CHECK: return
func.func @rock_pipeline_one_wave_schedule_v1(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 64 : i32, arch = "amdgcn-amd-amdhsa:gfx90a"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // 128 f16 elements = 256 bytes
    %rawLds = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %matrixA = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixB = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixC = memref.alloc() : memref<1x1xvector<4xf32>, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<128xf16, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>

    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %a = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %a, %regA[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        %a = memref.load %regA[%arg3] : memref<16xf16, #gpu.address_space<private>>
        memref.store %a, %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="LDSWrite"}
      rock.stage {
        %a = memref.load %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        memref.store %a, %regB[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %tid = rock.workitem_id : index
        rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at[%tid, %tid, %tid] {
          params = #rock.accel_gemm_params<
            kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 16, kpack = 8,
            mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1,
            scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0,
            gridGroupSize = 0, forceUnroll = true>
        } : memref<1x1xvector<4xf32>, #gpu.address_space<private>> += memref<1x2xvector<4xf16>, #gpu.address_space<private>> * memref<1x2xvector<4xf16>, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %regB[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// Test for single-wave kernel with scheduleVersion=3 (DirectToLDSDefault)
// When blockSize <= waveSize and scheduleVersion is 1 or 3, backward barriers should be skipped
// For scheduleVersion=3, the loop has only 2 stages: GlobalRead (writes directly to LDS) and LDSRead

// CHECK-LABEL: func.func @rock_pipeline_one_wave_schedule_v3
// CHECK: scf.for
// Inside loop - only ONE barrier for single-wave with scheduleVersion=3
// CHECK: memref.store {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK-NEXT: rock.lds_barrier
// CHECK: memref.load {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: rock.threadwise_gemm_accel
// No second barrier before end of loop body
// CHECK: }
// No barriers after loop for this test since it doesn't fully pipeline
// CHECK-NOT: rock.lds_barrier
// CHECK: return
func.func @rock_pipeline_one_wave_schedule_v3(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 64 : i32, arch = "amdgcn-amd-amdhsa:gfx90a"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // 128 f16 elements = 256 bytes
    %rawLds = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %matrixA = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixB = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixC = memref.alloc() : memref<1x1xvector<4xf32>, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<128xf16, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>

    // For scheduleVersion=3 (DirectToLDS), there are only 2 stages:
    // Stage 1: GlobalRead - loads from global and writes directly to LDS
    // Stage 2: LDSRead - reads from LDS and performs MFMA
    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        // GlobalRead stage: load from global and write DIRECTLY to LDS (Direct-to-LDS)
        %a = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %a, %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        // LDSRead stage: read from LDS and perform MFMA
        %a = memref.load %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        memref.store %a, %regB[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %tid = rock.workitem_id : index
        rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at[%tid, %tid, %tid] {
          params = #rock.accel_gemm_params<
            kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 16, kpack = 8,
            mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1,
            scheduleVersion = 3, outputSwizzle = 2, wavesPerEU = 0,
            gridGroupSize = 0, forceUnroll = true>
        } : memref<1x1xvector<4xf32>, #gpu.address_space<private>> += memref<1x2xvector<4xf16>, #gpu.address_space<private>> * memref<1x2xvector<4xf16>, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %regB[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// Test for multi-wave kernel with scheduleVersion=1 - should still have backward barrier
// When blockSize > waveSize, backward barriers should NOT be skipped

// CHECK-LABEL: func.func @rock_pipeline_multi_wave_schedule_v1
// Prologue stores to LDS:
// CHECK: memref.store {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: scf.for
// Inside loop - TWO barriers for multi-wave (forward + backward)
// CHECK: rock.lds_barrier
// CHECK: memref.load {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: rock.threadwise_gemm_accel
// CHECK: rock.lds_barrier
// CHECK: memref.store {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: }
// Epilogue barrier and LDS read:
// CHECK: rock.lds_barrier
// CHECK: return
func.func @rock_pipeline_multi_wave_schedule_v1(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 128 : i32, arch = "amdgcn-amd-amdhsa:gfx90a"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // 128 f16 elements = 256 bytes
    %rawLds = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %matrixA = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixB = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixC = memref.alloc() : memref<1x1xvector<4xf32>, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<128xf16, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>

    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %a = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %a, %regA[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        %a = memref.load %regA[%arg3] : memref<16xf16, #gpu.address_space<private>>
        memref.store %a, %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="LDSWrite"}
      rock.stage {
        %a = memref.load %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        memref.store %a, %regB[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %tid = rock.workitem_id : index
        rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at[%tid, %tid, %tid] {
          params = #rock.accel_gemm_params<
            kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 16, kpack = 8,
            mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1,
            scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0,
            gridGroupSize = 0, forceUnroll = true>
        } : memref<1x1xvector<4xf32>, #gpu.address_space<private>> += memref<1x2xvector<4xf16>, #gpu.address_space<private>> * memref<1x2xvector<4xf16>, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %regB[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// Test for single-wave kernel with scheduleVersion=2 (DoubleBuffer) - should still have backward barrier
// scheduleVersion=2 does NOT allow skipping backward barrier even for single-wave

// CHECK-LABEL: func.func @rock_pipeline_one_wave_schedule_v2
// Prologue stores to LDS:
// CHECK: memref.store {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: scf.for
// Inside loop - TWO barriers even for single-wave with scheduleVersion=2
// CHECK: rock.lds_barrier
// CHECK: memref.load {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: rock.threadwise_gemm_accel
// CHECK: rock.lds_barrier
// CHECK: memref.store {{.*}} : memref<128xf16, #gpu.address_space<workgroup>>
// CHECK: }
// Epilogue barrier and LDS read:
// CHECK: rock.lds_barrier
// CHECK: return
func.func @rock_pipeline_one_wave_schedule_v2(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) attributes {block_size = 64 : i32, arch = "amdgcn-amd-amdhsa:gfx90a"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // 128 f16 elements = 256 bytes
    %rawLds = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<32xi8, #gpu.address_space<private>>
    %matrixA = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixB = memref.alloc() : memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %matrixC = memref.alloc() : memref<1x1xvector<4xf32>, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<128xf16, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>

    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %a = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %a, %regA[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.yield
      }{name="GlobalRead"}
      rock.stage {
        %a = memref.load %regA[%arg3] : memref<16xf16, #gpu.address_space<private>>
        memref.store %a, %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        rock.yield
      }{name="LDSWrite"}
      rock.stage {
        %a = memref.load %lds[%arg3] : memref<128xf16, #gpu.address_space<workgroup>>
        memref.store %a, %regB[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %tid = rock.workitem_id : index
        rock.threadwise_gemm_accel %matrixC += %matrixA * %matrixB at[%tid, %tid, %tid] {
          params = #rock.accel_gemm_params<
            kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 16, kpack = 8,
            mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1,
            scheduleVersion = 2, outputSwizzle = 2, wavesPerEU = 0,
            gridGroupSize = 0, forceUnroll = true>
        } : memref<1x1xvector<4xf32>, #gpu.address_space<private>> += memref<1x2xvector<4xf16>, #gpu.address_space<private>> * memref<1x2xvector<4xf16>, #gpu.address_space<private>>
        rock.yield
      }{name="LDSRead"}
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %regB[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}
