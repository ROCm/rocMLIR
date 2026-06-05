// RUN: rocmlir-opt %s --rock-pipeline="rock-pipeline-remove-stages=false" | FileCheck %s
// RUN: rocmlir-opt %s --rock-pipeline="rock-pipeline-remove-stages=true" | FileCheck %s --check-prefix=REMOVE-STAGES

// REMOVE-STAGES-LABEL: rock_nopipeline
// No pipeline attribute - no barriers expected
// REMOVE-STAGES: rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
// REMOVE-STAGES: rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
// REMOVE-STAGES-NOT: rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
// REMOVE-STAGES-NOT: rock.stage
// REMOVE-STAGES-NOT: rock.lds_barrier
// REMOVE-STAGES: scf.for
  // REMOVE-STAGES-NOT: rock.stage
  // REMOVE-STAGES-NOT: rock.lds_barrier
  // REMOVE-STAGES-NOT: rock.extract_multibuffer
  // REMOVE-STAGES: scf.for
    // REMOVE-STAGES-NOT: rock.stage
    // REMOVE-STAGES-NOT: rock.lds_barrier
  // REMOVE-STAGES: scf.for
    // REMOVE-STAGES-NOT: rock.stage
    // REMOVE-STAGES-NOT: rock.lds_barrier
// REMOVE-STAGES: return
func.func @rock_nopipeline(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds0  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds1  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds0 = memref.view %rawLds0[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds1 = memref.view %rawLds1[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>

    scf.for %idx = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds0[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds0[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds1[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds1[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// one loop inside another loop, inner loop has pipeline<1> attribute
// CHECK-LABEL: rock_pipeline_oneloop
// REMOVE-STAGES-LABEL: rock_pipeline_oneloop
// REMOVE-STAGES-NOT: rock.stage
// REMOVE-STAGES: scf.for
  // Prologue: first LDS write has no barrier (iteration 0 of the pipeline)
  // REMOVE-STAGES: memref.store %{{.*}}, %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // Prologue: barrier immediately before second LDS write
  // REMOVE-STAGES: %[[PRO_VAL:.*]] = memref.load %{{.*}} : memref<16xf16, #gpu.address_space<private>>
  // REMOVE-STAGES-NEXT: rock.lds_barrier
  // REMOVE-STAGES-NEXT: %[[PRO_LDS:.*]] = rock.extract_multibuffer
  // REMOVE-STAGES-NEXT: memref.store %[[PRO_VAL]], %[[PRO_LDS]]{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: scf.for
    // Loop body: barrier immediately before LDS write
    // REMOVE-STAGES: %[[LOOP_VAL:.*]] = memref.load %{{.*}} : memref<16xf16, #gpu.address_space<private>>
    // REMOVE-STAGES-NEXT: rock.lds_barrier
    // REMOVE-STAGES-NEXT: %[[LOOP_LDS:.*]] = rock.extract_multibuffer
    // REMOVE-STAGES-NEXT: memref.store %[[LOOP_VAL]], %[[LOOP_LDS]]{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: }
  // Epilogue: barrier immediately before LDS write
  // REMOVE-STAGES: %[[EPI_W_VAL:.*]] = memref.load %{{.*}} : memref<16xf16, #gpu.address_space<private>>
  // REMOVE-STAGES-NEXT: rock.lds_barrier
  // REMOVE-STAGES-NEXT: %[[EPI_W_LDS:.*]] = rock.extract_multibuffer
  // REMOVE-STAGES-NEXT: memref.store %[[EPI_W_VAL]], %[[EPI_W_LDS]]{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // Epilogue: barrier immediately before LDS read
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES-NEXT: %[[EPI_R_LDS:.*]] = rock.extract_multibuffer
  // REMOVE-STAGES-NEXT: %[[EPI_R_VAL:.*]] = memref.load %[[EPI_R_LDS]]{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
// REMOVE-STAGES: return
func.func @rock_pipeline_oneloop(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds2:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg0:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg1:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    // CHECK: %[[ldsView2:.*]] = memref.view %[[rawLds2]]

    // CHECK: scf.for
    scf.for %idx = %c0 to %c4 step %c1 {
      // CHECK: name = "S0" 
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c0]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: name = "S3" 
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<1>}
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// two loops inside an outer loop, inner loops have pipeline<1> attribute
// CHECK-LABEL: rock_pipeline_twoloops
// REMOVE-STAGES-LABEL: rock_pipeline_twoloops
// REMOVE-STAGES-NOT: rock.stage
// REMOVE-STAGES: scf.for
  // First inner pipelined loop
  // Prologue: barrier before LDS write
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: scf.for
    // Loop body: barrier before LDS write
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: }
  // Epilogue: barrier before LDS write
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // Epilogue: barrier before LDS read
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.load %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // Second inner pipelined loop
  // Prologue: barrier before LDS write
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: scf.for
    // Loop body: barrier before LDS write
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: }
  // Epilogue: barrier before LDS write
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // Epilogue: barrier before LDS read
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.load %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
// REMOVE-STAGES: return
func.func @rock_pipeline_twoloops(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds2  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds2 = memref.view %rawLds2[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds2:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds3:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds4:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg0:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg1:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    // CHECK: %[[ldsView2:.*]] = memref.view %[[rawLds2]]
    // CHECK: %[[ldsView3:.*]] = memref.view %[[rawLds3]]
    // CHECK: %[[ldsView4:.*]] = memref.view %[[rawLds4]]

    // CHECK: scf.for
    scf.for %idx = %c0 to %c4 step %c1 {
      // CHECK: name = "S0" 
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c0]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: name = "S3" 
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<1>}
      
      // CHECK: name = "S0" 
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S2"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c0]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S2"
      // CHECK: name = "S3" 
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<1>}
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// two loops inside an outer loop, inner loops have pipeline<2> attribute
// CHECK-LABEL: rock_pipeline_twoloops_ii2
// REMOVE-STAGES-LABEL: rock_pipeline_twoloops_ii2
// REMOVE-STAGES-NOT: rock.stage
// REMOVE-STAGES: scf.for
  // First inner pipelined loop (ii=2)
  // Prologue: barrier before LDS write
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: scf.for
    // Loop body: barrier before LDS read
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.load %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
    // Loop body: barrier before LDS write
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: }
  // Epilogue: barrier before LDS read
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.load %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // Second inner pipelined loop (ii=2)
  // Prologue: barrier before LDS write
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: scf.for
    // Loop body: barrier before LDS read
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.load %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
    // Loop body: barrier before LDS write
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
  // REMOVE-STAGES: }
  // Epilogue: barrier before LDS read
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: memref.load %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
// REMOVE-STAGES: return
func.func @rock_pipeline_twoloops_ii2(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds2  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds2 = memref.view %rawLds2[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds2:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg0:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg1:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    // CHECK: %[[ldsView2:.*]] = memref.view %[[rawLds2]]

    // CHECK: scf.for
    scf.for %idx = %c0 to %c4 step %c1 {
      // CHECK: name = "S0" 
      // CHECK: name = "__bwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]])
      // CHECK: name = "S1"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c3]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: name = "S0"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]])
        // CHECK: name = "S2"
        // CHECK: name = "__bwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]])
        // CHECK: name = "S1"
        // CHECK: name = "S3"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]])
      // CHECK: name = "S2"
      // CHECK: name = "S3"
      scf.for %arg3 = %c0 to %c4 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<2>}
      
      // CHECK: name = "S0" 
      // CHECK: name = "__bwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c3]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: name = "S0"
        // CHECK: rock.extract_multibuffer(%[[ldsView2]])
        // CHECK: name = "S2"
        // CHECK: name = "__bwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView2]])
        // CHECK: name = "S1"
        // CHECK: name = "S3"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: name = "S3"
      scf.for %arg3 = %c0 to %c4 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<2>}
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// two loops inside an outer loop (which is inside another outer loop), inner loops have pipeline<1> attribute
// CHECK-LABEL: rock_pipeline_twoloops_triplenested
// REMOVE-STAGES-LABEL: rock_pipeline_twoloops_triplenested
// REMOVE-STAGES-NOT: rock.stage
// REMOVE-STAGES: scf.for
  // REMOVE-STAGES: scf.for
    // First inner pipelined loop
    // Prologue: barrier before LDS write
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: scf.for
      // Loop body: barrier before LDS write
      // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: }
    // Epilogue barriers
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: rock.lds_barrier
    // Second inner pipelined loop
    // Prologue: barrier before LDS write
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: scf.for
      // Loop body: barrier before LDS write
      // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: }
    // Epilogue barriers
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: rock.lds_barrier
// REMOVE-STAGES: return
func.func @rock_pipeline_twoloops_triplenested(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds2  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds2 = memref.view %rawLds2[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds2:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds3:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds4:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg0:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg1:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    // CHECK: %[[ldsView2:.*]] = memref.view %[[rawLds2]]
    // CHECK: %[[ldsView3:.*]] = memref.view %[[rawLds3]]
    // CHECK: %[[ldsView4:.*]] = memref.view %[[rawLds4]]

    // CHECK: scf.for
    scf.for %idx = %c0 to %c4 step %c1 {
      scf.for %idx2 = %c0 to %c4 step %c1 {
        // CHECK: name = "S0" 
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S2"
        // CHECK: scf.for 
        // CHECK-SAME: %[[c0]] to %[[c0]]
          // CHECK: name = "__fwd_barrier__"
          // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
          // CHECK: name = "S1"
          // CHECK: name = "S0"
          // CHECK: name = "S3"
          // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
          // CHECK: name = "S2"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S1"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S2"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S2"
        // CHECK: name = "S3" 
        scf.for %arg3 = %c0 to %c3 step %c1 {
          rock.stage {
            %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
            memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
            rock.yield
          }{name="S0"}
          rock.stage {
            %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
            memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
            rock.yield
          }{name="S1"}
          rock.stage {
            %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
            %comp = arith.addf %tmp, %c2 : f16
            memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
            rock.yield
          }{name="S2"}
          rock.stage {
            %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
            %comp = arith.addf %tmp, %c2 : f16
            memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
            rock.yield
          }{name="S3"}
        }{rock.pipeline = #rock.rock.pipeline<1>}
        
        // CHECK: name = "S0" 
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S2"
        // CHECK: scf.for 
        // CHECK-SAME: %[[c0]] to %[[c0]]
          // CHECK: name = "__fwd_barrier__"
          // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
          // CHECK: name = "S1"
          // CHECK: name = "S0"
          // CHECK: name = "S3"
          // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
          // CHECK: name = "S2"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S1"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S2"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S2"
        // CHECK: name = "S3" 
        scf.for %arg3 = %c0 to %c3 step %c1 {
          rock.stage {
            %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
            memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
            rock.yield
          }{name="S0"}
          rock.stage {
            %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
            memref.store %tmp, %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
            rock.yield
          }{name="S1"}
          rock.stage {
            %tmp = memref.load %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
            %comp = arith.addf %tmp, %c2 : f16
            memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
            rock.yield
          }{name="S2"}
          rock.stage {
            %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
            %comp = arith.addf %tmp, %c2 : f16
            memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
            rock.yield
          }{name="S3"}
        }{rock.pipeline = #rock.rock.pipeline<1>}
      }
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// two outer loops that each contain two inner loops, inner loops have pipeline<1> attribute
// CHECK-LABEL: rock_pipeline_twoloops_twoouterloops
// REMOVE-STAGES-LABEL: rock_pipeline_twoloops_twoouterloops
// REMOVE-STAGES-NOT: rock.stage
// First outer loop
// REMOVE-STAGES: scf.for
  // First inner pipelined loop
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: scf.for
    // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: }
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: rock.lds_barrier
  // Second inner pipelined loop
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: scf.for
    // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: }
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: rock.lds_barrier
// Second outer loop
// REMOVE-STAGES: scf.for
  // First inner pipelined loop
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: scf.for
    // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: }
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: rock.lds_barrier
  // Second inner pipelined loop
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: scf.for
    // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: }
  // REMOVE-STAGES: rock.lds_barrier
  // REMOVE-STAGES: rock.lds_barrier
// REMOVE-STAGES: return
func.func @rock_pipeline_twoloops_twoouterloops(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds2  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds3  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds4  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds2 = memref.view %rawLds2[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds3 = memref.view %rawLds3[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds4 = memref.view %rawLds4[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds2:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds3:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds4:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds5:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds6:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds7:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds8:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg0:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg1:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    // CHECK: %[[ldsView2:.*]] = memref.view %[[rawLds2]]
    // CHECK: %[[ldsView3:.*]] = memref.view %[[rawLds3]]
    // CHECK: %[[ldsView4:.*]] = memref.view %[[rawLds4]]
    // CHECK: %[[ldsView5:.*]] = memref.view %[[rawLds5]]
    // CHECK: %[[ldsView6:.*]] = memref.view %[[rawLds6]]
    // CHECK: %[[ldsView7:.*]] = memref.view %[[rawLds7]]
    // CHECK: %[[ldsView8:.*]] = memref.view %[[rawLds8]]

    // CHECK: scf.for
    scf.for %idx = %c0 to %c4 step %c1 {
      // CHECK: name = "S0" 
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c0]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
        // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView1]], %[[ldsView2]])
      // CHECK: name = "S2"
      // CHECK: name = "S3" 
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<1>}
      
      // CHECK: name = "S0" 
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S2"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c0]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
        // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView3]], %[[ldsView4]])
      // CHECK: name = "S2"
      // CHECK: name = "S3" 
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<1>}
    }
    
    // CHECK: scf.for
    scf.for %idx = %c0 to %c4 step %c1 {
      // CHECK: name = "S0" 
      // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
      // CHECK: name = "S2"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c0]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
        // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView5]], %[[ldsView6]])
      // CHECK: name = "S2"
      // CHECK: name = "S3" 
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds3[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds3[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<1>}
      
      // CHECK: name = "S0" 
      // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
      // CHECK: name = "S2"
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c0]]
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
        // CHECK: name = "S1"
        // CHECK: name = "S0"
        // CHECK: name = "S3"
        // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
        // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView7]], %[[ldsView8]])
      // CHECK: name = "S2"
      // CHECK: name = "S3" 
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
          memref.store %tmp, %lds4[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S1"}
        rock.stage {
          %tmp = memref.load %lds4[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{rock.pipeline = #rock.rock.pipeline<1>}
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// one loop inside a loop, inner loops have pipeline<1> attribute and no rock.stages
// CHECK-LABEL: rock_pipeline_nestednostages
// REMOVE-STAGES-LABEL: rock_pipeline_nestednostages
// No stages to pipeline - but barriers are preserved when stages are empty
// REMOVE-STAGES-NOT: rock.stage
// REMOVE-STAGES: scf.for
  // REMOVE-STAGES: scf.for
    // Barriers preserved: barrier before LDS write
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.store %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
    // Barriers preserved: barrier before LDS read
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: memref.load %{{.*}} : memref<16xf16, #gpu.address_space<workgroup>>
// REMOVE-STAGES: return
func.func @rock_pipeline_nestednostages(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
    // CHECK: rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK-NOT: rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>

    // CHECK-NOT: rock.stage
    // CHECK: scf.for %{{.*}} = %[[c0]] to %[[c4]] step %[[c1]]
      // CHECK-NOT: rock.stage
      // CHECK-NOT: rock.extract_multibuffer
    // CHECK: scf.for %{{.*}} = %[[c0]] to %[[c3]] step %[[c1]]
    // CHECK-NOT: rock.stage
    scf.for %idx = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c3 step %c1 {
        %tmp1 = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
        memref.store %tmp1, %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %tmp2 = memref.load %reg0[%arg3] : memref<16xf16, #gpu.address_space<private>>
        rock.lds_barrier
        memref.store %tmp2, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        rock.lds_barrier
        %tmp3 = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
        memref.store %tmp3, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %tmp4 = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
        %comp = arith.addf %tmp4, %c2 : f16
        memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
      }{rock.pipeline = #rock.rock.pipeline<1>}
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// two loops inside an outer loop, inner loops have pipeline<2> attribute and two rock.stages
// CHECK-LABEL: rock_pipeline_twoloops_ii_equal_numstages
// REMOVE-STAGES-LABEL: rock_pipeline_twoloops_ii_equal_numstages
// REMOVE-STAGES-NOT: rock.stage
// REMOVE-STAGES: scf.for
  // First inner pipelined loop (ii=2, 2 stages - no prologue/epilogue)
  // REMOVE-STAGES: scf.for
    // Loop body: barrier before LDS write, barrier before LDS read
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: rock.lds_barrier
  // Second inner pipelined loop (ii=2, 2 stages - no prologue/epilogue)
  // REMOVE-STAGES: scf.for
    // Loop body: barrier before LDS write, barrier before LDS read
    // REMOVE-STAGES: rock.lds_barrier
    // REMOVE-STAGES: rock.lds_barrier
// REMOVE-STAGES: return
func.func @rock_pipeline_twoloops_ii_equal_numstages(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %rawLds2  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    %lds2 = memref.view %rawLds2[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds2:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK-NOT: rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    // CHECK: %[[ldsView2:.*]] = memref.view %[[rawLds2]]

    // CHECK: scf.for
    // CHECK-SAME: %[[c0]] to %[[c4]]
    scf.for %idx = %c0 to %c4 step %c1 {
      // CHECK-NEXT: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c3]]
        // CHECK: name = "__bwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]])
        // CHECK: name = "S0"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView1]])
        // CHECK: name = "S1"
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %lds[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S1"}
      }{rock.pipeline = #rock.rock.pipeline<2>}
      
      // CHECK: scf.for 
      // CHECK-SAME: %[[c0]] to %[[c3]]
        // CHECK: name = "__bwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView2]])
        // CHECK: name = "S0"
        // CHECK: name = "__fwd_barrier__"
        // CHECK: rock.extract_multibuffer(%[[ldsView2]])
        // CHECK: name = "S1"
      scf.for %arg3 = %c0 to %c3 step %c1 {
        rock.stage {
          %tmp = memref.load %input[%arg3] : memref<16xf16, #gpu.address_space<global>>
          memref.store %tmp, %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          rock.yield
        }{name="S0"}
        rock.stage {
          %tmp = memref.load %lds2[%arg3] : memref<16xf16, #gpu.address_space<workgroup>>
          %comp = arith.addf %tmp, %c2 : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S1"}
      }{rock.pipeline = #rock.rock.pipeline<2>}
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}
