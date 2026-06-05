// RUN: rocmlir-driver --rock-pipeline %s | FileCheck %s
// RUN: rocmlir-driver --rock-pipeline %s | FileCheck %s --check-prefix=COUNT

// COUNT-COUNT-1: rock.lds_barrier

module {
  func.func @pipeline_loop_in_scf_if(%arg0: memref<128xf16>, %arg1: memref<128xf16>, %arg2: memref<128xf16>, %arg3: i32) attributes {block_size = 64 : i32, grid_size = 1 : i32, rock.kernel} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sgt, %arg3, %c0_i32 : i32
    scf.if %0 {
      %1 = rock.alloc() : memref<64xf16, #gpu.address_space<workgroup>>
      %2 = rock.alloc() : memref<64xf16, #gpu.address_space<workgroup>>
      %3 = rock.alloc() : memref<32xf32, #gpu.address_space<private>>
      %cst = arith.constant 0.000000e+00 : f32
      affine.for %arg4 = 0 to 32 {
        memref.store %cst, %3[%arg4] : memref<32xf32, #gpu.address_space<private>>
      }

      // CHECK: scf.for %[[IV:.*]] = %c0 to %c3 step %c1 {
      // CHECK-NEXT: %[[IV_PLUS_1:.*]] = arith.addi %[[IV]], %c1
      // CHECK: %[[ALLOC_E:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
      // CHECK: %[[ALLOC_F:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
      // CHECK: %[[WID3:.*]] = rock.workitem_id : index
      // CHECK: %[[MUL:.*]] = arith.muli %[[IV_PLUS_1]], %c16
      // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[WID3]]
      // CHECK: memref.load %arg0[%[[ADD]]]
      // CHECK: memref.store {{.*}}, %[[ALLOC_E]][%c0]
      // CHECK: memref.load %arg1[%[[ADD]]]
      // CHECK: memref.store {{.*}}, %[[ALLOC_F]][%c0]
      scf.for %arg4 = %c0 to %c4 step %c1 {
        rock.stage {
          %4 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
          %5 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
          %6 = rock.workitem_id : index
          %c16 = arith.constant 16 : index
          %7 = arith.muli %arg4, %c16 : index
          %8 = arith.addi %7, %6 : index
          %9 = memref.load %arg0[%8] : memref<128xf16>
          memref.store %9, %4[%c0] : memref<16xf16, #gpu.address_space<private>>
          %10 = memref.load %arg1[%8] : memref<128xf16>
          memref.store %10, %5[%c0] : memref<16xf16, #gpu.address_space<private>>
          rock.yield
        } {name = "GlobalRead"}
        rock.stage {
          %4 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
          %5 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
          %6 = rock.workitem_id : index
          %7 = memref.load %4[%c0] : memref<16xf16, #gpu.address_space<private>>
          memref.store %7, %1[%6] : memref<64xf16, #gpu.address_space<workgroup>>
          %8 = memref.load %5[%c0] : memref<16xf16, #gpu.address_space<private>>
          memref.store %8, %2[%6] : memref<64xf16, #gpu.address_space<workgroup>>
          rock.yield
        } {name = "LDSWrite"}
        rock.lds_barrier
        rock.stage {
          // CHECK: affine.for %[[INNER_IV:.*]] = 0 to 16 {
          // CHECK: memref.load {{.*}}[%[[INNER_IV]]]
          // CHECK: memref.load {{.*}}[%[[INNER_IV]]]
          // CHECK: arith.extf
          // CHECK: arith.extf
          // CHECK: arith.mulf
          // CHECK: memref.load {{.*}}[%[[INNER_IV]]]
          // CHECK: arith.addf
          // CHECK: memref.store {{.*}}[%[[INNER_IV]]]
          // CHECK: }
          // CHECK: %[[ALLOC_PRIV_A:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
          // CHECK: %[[ALLOC_PRIV_B:.*]] = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
          // CHECK: %[[WID_PRIV:.*]] = rock.workitem_id : index
          // CHECK: memref.load %[[ALLOC_PRIV_A]][%c0]
          // CHECK-NEXT: rock.lds_barrier
          affine.for %arg5 = 0 to 16 {
            %4 = memref.load %1[%arg5] : memref<64xf16, #gpu.address_space<workgroup>>
            %5 = memref.load %2[%arg5] : memref<64xf16, #gpu.address_space<workgroup>>
            %6 = arith.extf %4 : f16 to f32
            %7 = arith.extf %5 : f16 to f32
            %8 = arith.mulf %6, %7 : f32
            %9 = memref.load %3[%arg5] : memref<32xf32, #gpu.address_space<private>>
            %10 = arith.addf %9, %8 : f32
            memref.store %10, %3[%arg5] : memref<32xf32, #gpu.address_space<private>>
          }
          rock.yield
        } {name = "MMA"}
        rock.lds_barrier
      } {rock.pipeline = #rock.rock.pipeline<2>}

      // CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<64xf16, #gpu.address_space<workgroup>>
      // CHECK: }
      // CHECK-NOT: {rock.pipeline = #rock.rock.pipeline<2>}

      // CHECK: affine.for %{{.*}} = 0 to 32 {
      affine.for %arg4 = 0 to 32 {
        %4 = memref.load %3[%arg4] : memref<32xf32, #gpu.address_space<private>>
        %5 = arith.truncf %4 : f32 to f16
        memref.store %5, %arg2[%arg4] : memref<128xf16>
      }
    }
    return
  }
}
