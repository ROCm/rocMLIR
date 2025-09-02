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

    // CHECK: name = "S0"
    // CHECK: name = "S1"
    // CHECK: name = "S0"
    // CHECK: scf.for
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[lds0]], %[[lds1]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[lds0]], %[[lds1]])
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

    // CHECK: name = "S0"
    // CHECK: name = "__bwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: scf.for
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[rawRegA]])
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[rawLds]])
      // CHECK: name = "S2"
      // CHECK: name = "__bwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[rawRegA]])
      // CHECK: rock.extract_multibuffer(%[[rawLds]])
      // CHECK: name = "S1"
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
    }{pipeline = #rock.pipeline<2>}

    %out = memref.load %regB[%c0] : memref<16xi8, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xi8, #gpu.address_space<global>>
    return
}

// this test shouldn't pipeline loop but it would add barriers and multibuffer by 1
// CHECK-LABEL: rock_pipeline_3_stages_ii_2_less_iterations
func.func @rock_pipeline_3_stages_ii_2_less_iterations(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : i8
    %c1_0 = arith.constant 1 : index

    %rawLds  = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    %rawRegA = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %rawRegB = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<16xi8, #gpu.address_space<workgroup>> to memref<16xi8, #gpu.address_space<workgroup>>
    %regA = memref.view %rawRegA[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>
    %regB = memref.view %rawRegB[%c0][] : memref<16xi8, #gpu.address_space<private>> to memref<16xi8, #gpu.address_space<private>>

    // CHECK: %[[rawLds:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawRegA:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawRegB:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    
    // CHECK: rock.extract_multibuffer(%[[rawRegA]])
    // CHECK: name = "S0"
    // CHECK: rock.extract_multibuffer(%[[rawRegA]])
    // CHECK: rock.extract_multibuffer(%[[rawLds]])
    // CHECK: name = "S1"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: rock.extract_multibuffer(%[[rawLds]])
    // CHECK: name = "S2"
    scf.for %arg3 = %c0 to %c1_0 step %c1 {
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

    // CHECK: scf.for
      // CHECK: name = "__bwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[rawRegA]])
      // CHECK: name = "S0"
      // CKECK: rock.extract_multibuffer(%[[rawLds]])
      // CHECK: rock.extract_multibuffer(%[[rawRegA]])
      // CHECK: name = "S1"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[rawLds]])
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

// This test shouldn't do any pipelining as it doesn't have any stages but it should still multibuffer by 1
// CHECK-LABEL: rock_pipeline_no_stages_ii_1
func.func @rock_pipeline_no_stages_ii_1(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
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
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[c16:.*]] = arith.constant 16 : index
    // CHECK: %[[lds0:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawRegA:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    // CHECK: %[[rawRegB:.*]] = rock.alloc() : memref<16xi8, #gpu.address_space<private>>

    // CHECK: %[[lds0View:.*]] = memref.view {{.*}}
    // CHECK: %[[rawRegAView:.*]] = memref.view {{.*}}
    // CHECK: %[[rawRegBView:.*]] = memref.view {{.*}}

    // CHECK: scf.for
    // CHECK-SAME: %[[c0]] to %[[c16]]
      // CHECK-NOT: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[lds0View]])
      // CHECK: rock.extract_multibuffer(%[[lds0View]])
    scf.for %arg3 = %c0 to %c16 step %c1 {
        %a = memref.load %input[%arg3] : memref<16xi8, #gpu.address_space<global>>
        memref.store %a, %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        %b = memref.load %regA[%arg3] : memref<16xi8, #gpu.address_space<private>>
        memref.store %b, %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %c = memref.load %lds[%arg3] : memref<16xi8, #gpu.address_space<workgroup>>
        %d = arith.addi %c, %c2 : i8
        memref.store %d, %regB[%arg3] : memref<16xi8, #gpu.address_space<private>>
    }{pipeline = #rock.pipeline<1>}
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
    
    // CHECK: name = "__bwd_barrier__"
    // CHECK: name = "S0"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: scf.for
      // CHECK: name = "__bwd_barrier__"
      // CHECK: name = "__fwd_barrier__"
      // CHECK:  rock.extract_multibuffer(%[[rawLds0]], %[[rawLds1]])
      // CHECK: name = "S0"
      // CHECK:  rock.extract_multibuffer(%[[rawLds0]], %[[rawLds1]])
      // CHECK: name = "S2"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S1"
      // CHECK:  rock.extract_multibuffer(%[[rawLds0]], %[[rawLds1]])
      // CHECK: name = "S3"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S2"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S3"
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

// CHECK-LABEL: rock_pipeline_4_stages_ii_1_i8
func.func @rock_pipeline_4_stages_ii_1_i8(%input : memref<16xi8, #gpu.address_space<global>>, %output : memref<16xi8, #gpu.address_space<global>>){
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
      // CHECK: rock.extract_multibuffer(%[[rawReg0]])
      // CHECK: rock.extract_multibuffer(%[[rawLds0]], %[[rawLds1]])
      // CHECK: name = "S1"
      // CHECK: rock.extract_multibuffer(%[[rawReg0]])
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[rawReg1]])
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[rawLds0]], %[[rawLds1]])
      // CHECK: rock.extract_multibuffer(%[[rawReg1]])
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

// CHECK-LABEL: rock_pipeline_4_stages_ii_1_f16
func.func @rock_pipeline_4_stages_ii_1_f16(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c16 = arith.constant 16 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds0:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[rawLds1:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg0:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg1:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView0:.*]] = memref.view %[[rawLds0]]
    // CHECK: %[[ldsView1:.*]] = memref.view %[[rawLds1]]
    
    // No multibuffering on Private buffers
    // CHECK: name = "S0"
    // CHECK: name = "S1"
    // CHECK: name = "S0"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: name = "S0"
    // CHECK: name = "S2"
    // CHECK: scf.for
      // CHECK: name = "__fwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView0]], %[[ldsView1]])
      // CHECK: name = "S1"
      // CHECK: name = "S0"
      // CHECK: name = "S3"
      // CHECK: rock.extract_multibuffer(%[[ldsView0]], %[[ldsView1]])
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
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// This test should adjust II to 2 to enable loop pipelining
// CHECK-LABEL: rock_pipeline_4_stages_ii_1_f16_less_iterations
func.func @rock_pipeline_4_stages_ii_1_f16_less_iterations(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c2_0 = arith.constant 2 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
    // CHECK: %[[c0:.*]] = arith.constant 0 : index
    // CHECK: %[[c1:.*]] = arith.constant 1 : index
    // CHECK: %[[rawLds:.*]] = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[reg0:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg1:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[reg2:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    // CHECK: %[[ldsView:.*]] = memref.view %[[rawLds]]

    // CHECK: name = "S0" 
    // CHECK: name = "__bwd_barrier__"
    // CHECK: name = "S1"
    // CHECK: scf.for
    // CHECK-SAME: %[[c0]] to %[[c1]]
      // CHECK: name = "__fwd_barrier__"
      // CHECK: name = "S0"
      // CHECK: rock.extract_multibuffer(%[[ldsView]])
      // CHECK: name = "S2"
      // CHECK: name = "__bwd_barrier__"
      // CHECK: rock.extract_multibuffer(%[[ldsView]])
      // CHECK: name = "S1"
      // CHECK: name = "S3"
    // CHECK: name = "__fwd_barrier__"
    // CHECK: name = "S2"
    // CHECK: name = "S3"
    scf.for %arg3 = %c0 to %c2_0 step %c1 {
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
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

// this test should do loop pipelining without adjust II but notice that it emits scf.for loop with zero iterations. 
// CHECK-LABEL: rock_pipeline_4_stages_ii_1_f16_less_iterations_2
func.func @rock_pipeline_4_stages_ii_1_f16_less_iterations_2(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index

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
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}
