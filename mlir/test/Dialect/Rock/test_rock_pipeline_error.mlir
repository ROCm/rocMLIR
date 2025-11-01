// RUN: rocmlir-opt --rock-pipeline="rock-pipeline-remove-stages=false" %s -verify-diagnostics

func.func @rock_pipeline_nested_parent_pipeline(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>){
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
  
    scf.for %idx = %c0 to %c4 step %c1 {
      // expected-error @+1 {{Nested pipelining is not supported yet}}
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
    }{pipeline = #rock.pipeline<1>}

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

func.func @rock_pipeline_step_nonone(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>){
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c2f = arith.constant 2.0 : f16
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %rawLds  = rock.alloc() : memref<32xi8, #gpu.address_space<workgroup>>
    %reg0 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg1 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %reg2 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>

    %lds = memref.view %rawLds[%c0][] : memref<32xi8, #gpu.address_space<workgroup>> to memref<16xf16, #gpu.address_space<workgroup>>
  
    scf.for %idx = %c0 to %c4 step %c2 {
      // expected-error @+1 {{Step size other one is not permitted in rock-pipeline}}
      scf.for %arg3 = %c0 to %c4 step %c2 {
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
          %comp = arith.addf %tmp, %c2f : f16
          memref.store %tmp, %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S2"}
        rock.stage {
          %tmp = memref.load %reg1[%arg3] : memref<16xf16, #gpu.address_space<private>>
          %comp = arith.addf %tmp, %c2f : f16
          memref.store %comp, %reg2[%arg3] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        }{name="S3"}
      }{pipeline = #rock.pipeline<1>}
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}

func.func @rock_pipeline_dynamic_count(%input : memref<16xf16, #gpu.address_space<global>>, %output : memref<16xf16, #gpu.address_space<global>>, %dyncount : index){
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
  
    scf.for %idx = %c0 to %c4 step %c1 {
      // expected-error @+1 {{Number of iterations are unknown while doing rock-pipeline}}
      scf.for %arg3 = %c0 to %dyncount step %c1 {
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
    }

    %out = memref.load %reg2[%c0] : memref<16xf16, #gpu.address_space<private>>
    memref.store %out, %output[%c0] : memref<16xf16, #gpu.address_space<global>>
    return
}
