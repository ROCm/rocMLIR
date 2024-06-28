// RUN: rocmlir-opt %s --rock-pipeline="rock-pipeline-remove-stages=false" | FileCheck %s


// CHECK-LABEL: rock_pipeline_3_stages_ii_1
func.func @rock_pipeline_3_stages_ii_1(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i8
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[lds0:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[lds1:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawRegA:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawRegB:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    // CHECK: %[[lds0View:.*]] = memref.view {{.*}}
    // CHECK: %[[lds1View:.*]] = memref.view {{.*}}
    // CHECK: %[[rawRegAView:.*]] = memref.view {{.*}}
    // CHECK: %[[rawRegBView:.*]] = memref.view {{.*}}

    // CHECK: name = "S0"
    // CHECK: name = "S1"
    // CHECK: name = "S0"
    // CHECK: scf.for
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[lds0View]], %[[lds1View]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[lds0View]], %[[lds1View]])
      // CHECK: name = "S2"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: name = "S2"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S2"
    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %a = memref.load %input[%arg3] : memref<16xi8, #gpu.address_space<global>>
        memref.store %a, %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S0"}
      rock.stage {
        %a = memref.load %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        memref.store %a, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        rock.yield
      }{name="S1"}
      rock.stage {
        %a = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %b = arith.addi %a, %c2 : i8
        memref.store %b, %regB[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S2"}
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %regB[%c0] : memref<16xi8, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xi8, #gpu.address_space<global>>
    return
}

// CHECK-LABEL: rock_pipeline_3_stages_ii_2
func.func @rock_pipeline_3_stages_ii_2(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i8
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>

    // CHECK: %[[rawLds:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawRegA:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawRegB:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[ldsView:.*]] = memref.view %[[rawLds]]
    // CHECK: %[[regAView:.*]] = memref.view %[[rawRegA]]
    // CHECK: %[[regBView:.*]] = memref.view %[[rawRegB]]

    // CHECK: name = "S0"
    // CHECK: name = "__bwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: scf.for
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[regAView]])
      // CHECK name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView]])
      // CHECK name = "S2"
      // CHECK: name = "__bwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[regAView]])
      // CHECK: rock.extract_multibuffer(%[[ldsView]])
      // CHECK: name = "S1"
    // CHECK: name = "__fwd_barrier__"
    // CHECK name = "S2"
    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %a = memref.load %input[%arg3] : memref<16xi8, #gpu.address_space<global>>
        memref.store %a, %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S0"}
      rock.stage {
        %a = memref.load %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        memref.store %a, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        rock.yield
      }{name="S1"}
      rock.stage {
        %a = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %b = arith.addi %a, %c2 : i8
        memref.store %b, %regB[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S2"}
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %regB[%c0] : memref<16xi8, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xi8, #gpu.address_space<global>>
    return
}

// CHECK-LABEL: rock_pipeline_3_stages_ii_3
func.func @rock_pipeline_3_stages_ii_3(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i8
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawLds:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawRegA:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawRegB:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[ldsView:.*]] = memref.view %[[rawLds]]
    // CHECK: %[[regAView:.*]] = memref.view %[[rawRegA]]
    // CHECK: %[[regBView:.*]] = memref.view %[[rawRegB]]

    // CHECK: scf.for
      // CHECK: name = "__bwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[regAView]])
      // CHECK: name = "S0"
      // CKECK: rock.extract_multibuffer(%[[ldsView]])
      // CHECK: rock.extract_multibuffer(%[[regAView]])
      // CHECK: name = "S1"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView]])
      // CHECK: name = "S2"
    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %a = memref.load %input[%arg3] : memref<16xi8, #gpu.address_space<global>>
        memref.store %a, %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S0"}
      rock.stage {
        %a = memref.load %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        memref.store %a, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        rock.yield
      }{name="S1"}
      rock.stage {
        %a = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %b = arith.addi %a, %c2 : i8
        memref.store %b, %regB[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S2"}
    }{pipeline = #rock.pipeline<3>}

    %out = memref.load %regB[%c0] : memref<16xi8, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xi8, #gpu.address_space<global>>
    return
}

// CHECK-LABEL: rock_pipeline_4_stages_ii_2
func.func @rock_pipeline_4_stages_ii_2(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i8
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    %rawReg = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
    %reg = memref.view %rawReg[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawLds0:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawReg:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[lds0View:.*]] = memref.view %[[rawLds0]]
    // CHECK: %[[lds1View:.*]] = memref.view %[[rawLds1]]
    // CHECK: memref.view %[[rawReg]]

    // CHECK: name = "S0"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S1"
    // CHECK scf.for
      // CHECK: name = "__fwd_barrier__"
      // CHECK:  rock.extract_multibuffer(%[[lds0View]], %[[lds1View]])
      // CHECK: name = "S0"
      // CHECK:  rock.extract_multibuffer(%[[lds0View]], %[[lds1View]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK name = "S1"
      // CHECK:  rock.extract_multibuffer(%[[lds0View]], %[[lds1View]])
      // CHECK name = "S3"
    // CHECK: name = "__fwd_barrier__"
    // CHECK name = "S2"
    // CHECK: name = "__fwd_barrier__"
    // CHECK name = "S3"
    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %tmp = memref.load %input[%arg3] : memref<16xi8, #gpu.address_space<global>>
        memref.store %tmp, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        rock.yield
      }{name="S0"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %b = arith.addi %tmp, %c2 : i8
        memref.store %tmp, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        rock.yield
      }{name="S1"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %b = arith.addi %tmp, %c2 : i8
        memref.store %tmp, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        rock.yield
      }{name="S2"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %b = arith.addi %tmp, %c2 : i8
        memref.store %tmp, %reg[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S3"}
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %reg[%c0] : memref<16xi8, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xi8, #gpu.address_space<global>>
    return
}

// CHECK-LABEL: rock_pipeline_4_stages_ii_1
func.func @rock_pipeline_4_stages_ii_1(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i8
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    %rawReg0 = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %rawReg1 = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %rawReg2 = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
    %reg0 = memref.view %rawReg0[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    %reg1 = memref.view %rawReg1[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    %reg2 = memref.view %rawReg2[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawLds0:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawReg0:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawReg1:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawReg2:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[ldsView0:.*]] = memref.view %[[rawLds0]]
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    // CHECK: %[[regView0:.*]] = memref.view %[[rawReg0]]
    // CHECK: %[[regView1:.*]] = memref.view %[[rawReg1]]
    // CHECK: %[[regView2:.*]] = memref.view %[[rawReg2]]

    // Please note how we swap S0/S1 and S2/S3 to avoid private multi-buffers
    // CHECK: name = "S0"
    // CHECK: name = "S1"
    // CHECK: name = "S0"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: name = "S0"
    // CHECK: name = "S2"
    // CHECK: scf.for
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[regView0]])
      // CHECK: rock.extract_multibuffer(%[[ldsView0]], %[[ldsView1]])
      // CHECK: name = "S1"
      // CHECK: rock.extract_multibuffer(%[[regView0]])
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[regView1]])
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView0]], %[[ldsView1]])
      // CHECK: rock.extract_multibuffer(%[[regView1]])
      // CHECK: name = "S2"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: name = "S3"
    // CHECK: name = "S2"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S3"
    // CHECK: name = "S2"
    // CHECK: name = "S3"
    scf.for %arg3 = %c0 to %c16 step %c1 {
      rock.stage {
        %tmp = memref.load %input[%arg3] : memref<16xi8, #gpu.address_space<global>>
        memref.store %tmp, %reg0[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S0"}
      rock.stage {
        %tmp = memref.load %reg0[%arg3] : memref<16xi8, #gpu.address_space<private>>
        memref.store %tmp, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        rock.yield
      }{name="S1"}
      rock.stage {
        %tmp = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %comp = arith.addi %tmp, %c2 : i8
        memref.store %tmp, %reg1[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S2"}
      rock.stage {
        %tmp = memref.load %reg1[%arg3] : memref<16xi8, #gpu.address_space<private>>
        %comp = arith.addi %tmp, %c2 : i8
        memref.store %comp, %reg2[%arg3] : memref<16xi8, #gpu.address_space<private>>
        rock.yield
      }{name="S3"}
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %reg2[%c0] : memref<16xi8, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xi8, #gpu.address_space<global>>
    return
}
