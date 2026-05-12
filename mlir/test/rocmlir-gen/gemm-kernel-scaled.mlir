// Default behaviour now generates scales in the natural form (no broadcast).
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv | FileCheck %s --check-prefix=GEMM-SCALED
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleA --transScaleB -pv | FileCheck %s --check-prefix=GEMM-SCALED-BOTHTRANS
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleA -pv | FileCheck %s --check-prefix=GEMM-SCALED-TRANSA
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleB -pv | FileCheck %s --check-prefix=GEMM-SCALED-TRANSB
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -scale_a_dtype f32 -scale_b_dtype f32 | FileCheck %s --check-prefix=GEMM-SCALED-F32

// One additional RUN line covers the legacy broadcast path with the default
// quantBlockSize (32) so the broadcasted IR shape is still exercised by tests.
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -broadcastScales=true | FileCheck %s --check-prefix=GEMM-SCALED-BCAST

// Negative test: quantBlockSize values other than 32 must be rejected at
// generation time, regardless of broadcastScales, because the lowering
// pipeline only supports the OCP MX block size of 32 (broadcasted-form
// scales would be silently re-grouped to 32, dropping every other scale).
// RUN: not rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -quantBlockSize 16 2>&1 | FileCheck %s --check-prefix=GEMM-SCALED-QB-NAT
// RUN: not rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -quantBlockSize 16 -broadcastScales=true 2>&1 | FileCheck %s --check-prefix=GEMM-SCALED-QB-BCAST

// GEMM-SCALED-QB-NAT: rocmlir-gen: -quantBlockSize=16 is not supported by the lowering pipeline
// GEMM-SCALED-QB-BCAST: rocmlir-gen: -quantBlockSize=16 is not supported by the lowering pipeline

// GEMM-SCALED: func.func @rock_gemm
// GEMM-SCALED-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf8E8M0FNU>, %[[ARG4:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-SAME: memref<128xf8E8M0FNU> to memref<1x16x8xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-SAME: memref<128xf8E8M0FNU> to memref<1x8x16xf8E8M0FNU>
// GEMM-SCALED-NOT: rock.transform {{.*}}AddDim
// GEMM-SCALED-NOT: rock.transform {{.*}}Broadcast
// GEMM-SCALED: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_EXPAND]] * %{{.*}} scaled by %[[SCALEB_EXPAND]]
// GEMM-SCALED-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x8xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x8x16xf8E8M0FNU>

// GEMM-SCALED: func.func @host_naive_gemm
// GEMM-SCALED-SAME: (%[[A:.*]]: memref<4096xf4E2M1FN>, %[[B:.*]]: memref<4096xf4E2M1FN>, %[[C:.*]]: memref<256xf32>, %[[SCALEA:.*]]: memref<128xf8E8M0FNU>, %[[SCALEB:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED: %[[A_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED: call @_memcpy_f4E2M1FN_f32_4096(%[[A]], %[[A_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED: %[[B_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED: call @_memcpy_f4E2M1FN_f32_4096(%[[B]], %[[B_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED: %[[A_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEA]], %[[A_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED: %[[B_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEB]], %[[B_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED: %[[A_EXPAND:.*]] = memref.expand_shape %[[A_ALLOC]]
// GEMM-SCALED-SAME: memref<4096xf32> into memref<16x256xf32>
// GEMM-SCALED: %[[B_EXPAND:.*]] = memref.expand_shape %[[B_ALLOC]] 
// GEMM-SCALED-SAME: memref<4096xf32> into memref<256x16xf32>
// GEMM-SCALED: %[[A_SCALE_EXPAND:.*]] = memref.expand_shape %[[A_SCALE_ALLOC]] 
// GEMM-SCALED-SAME: memref<128xf32> into memref<16x8xf32>
// GEMM-SCALED: %[[B_SCALE_EXPAND:.*]] = memref.expand_shape %[[B_SCALE_ALLOC]] 
// GEMM-SCALED-SAME: memref<128xf32> into memref<8x16xf32>
// GEMM-SCALED: %[[C_EXPAND:.*]] = memref.expand_shape %[[C]] 
// GEMM-SCALED-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED: linalg.generic
// GEMM-SCALED-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]], %[[A_SCALE_EXPAND]], %[[B_SCALE_EXPAND]] : memref<16x256xf32>, memref<256x16xf32>, memref<16x8xf32>, memref<8x16xf32>) outs(%[[C_EXPAND]] : memref<16x16xf32>) {
// GEMM-SCALED: (%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-NEXT: linalg.yield

// GEMM-SCALED-BOTHTRANS: func.func @rock_gemm
// GEMM-SCALED-BOTHTRANS-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf8E8M0FNU>, %[[ARG4:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-BOTHTRANS: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<128xf8E8M0FNU> to memref<1x8x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<128xf8E8M0FNU> to memref<1x16x8xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS-NOT: rock.transform {{.*}}AddDim
// GEMM-SCALED-BOTHTRANS-NOT: rock.transform {{.*}}Broadcast
// GEMM-SCALED-BOTHTRANS: rock.gemm %{{.*}} = %{{.*}} scaled by tr %[[SCALEA_EXPAND]] * %{{.*}} scaled by tr %[[SCALEB_EXPAND]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x8x16xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x16x8xf8E8M0FNU>

// GEMM-SCALED-BOTHTRANS: func.func @host_naive_gemm
// GEMM-SCALED-BOTHTRANS-SAME: (%[[A:.*]]: memref<4096xf4E2M1FN>, %[[B:.*]]: memref<4096xf4E2M1FN>, %[[C:.*]]: memref<256xf32>, %[[SCALEA:.*]]: memref<128xf8E8M0FNU>, %[[SCALEB:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-BOTHTRANS: %[[A_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-BOTHTRANS: call @_memcpy_f4E2M1FN_f32_4096(%[[A]], %[[A_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-BOTHTRANS: %[[B_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-BOTHTRANS: call @_memcpy_f4E2M1FN_f32_4096(%[[B]], %[[B_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-BOTHTRANS: %[[A_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED-BOTHTRANS: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEA]], %[[A_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED-BOTHTRANS: %[[B_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED-BOTHTRANS: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEB]], %[[B_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED-BOTHTRANS: %[[A_EXPAND:.*]] = memref.expand_shape %[[A_ALLOC]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<4096xf32> into memref<16x256xf32>
// GEMM-SCALED-BOTHTRANS: %[[B_EXPAND:.*]] = memref.expand_shape %[[B_ALLOC]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<4096xf32> into memref<256x16xf32>
// GEMM-SCALED-BOTHTRANS: %[[A_SCALE_EXPAND:.*]] = memref.expand_shape %[[A_SCALE_ALLOC]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<128xf32> into memref<8x16xf32>
// GEMM-SCALED-BOTHTRANS: %[[B_SCALE_EXPAND:.*]] = memref.expand_shape %[[B_SCALE_ALLOC]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<128xf32> into memref<16x8xf32>
// GEMM-SCALED-BOTHTRANS: %[[C_EXPAND:.*]] = memref.expand_shape %[[C]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-BOTHTRANS: linalg.generic
// GEMM-SCALED-BOTHTRANS-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]], %[[A_SCALE_EXPAND]], %[[B_SCALE_EXPAND]] : memref<16x256xf32>, memref<256x16xf32>, memref<8x16xf32>, memref<16x8xf32>) outs(%[[C_EXPAND]] : memref<16x16xf32>) {
// GEMM-SCALED-BOTHTRANS: (%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-BOTHTRANS-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-BOTHTRANS-NEXT: linalg.yield


// GEMM-SCALED-TRANSA: func.func @rock_gemm
// GEMM-SCALED-TRANSA-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf8E8M0FNU>, %[[ARG4:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-TRANSA: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-TRANSA-SAME: memref<128xf8E8M0FNU> to memref<1x8x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-TRANSA-SAME: memref<128xf8E8M0FNU> to memref<1x8x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA-NOT: rock.transform {{.*}}AddDim
// GEMM-SCALED-TRANSA-NOT: rock.transform {{.*}}Broadcast
// GEMM-SCALED-TRANSA: rock.gemm %{{.*}} = %{{.*}} scaled by tr %[[SCALEA_EXPAND]] * %{{.*}} scaled by %[[SCALEB_EXPAND]]
// GEMM-SCALED-TRANSA-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x8x16xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x8x16xf8E8M0FNU>

// GEMM-SCALED-TRANSA: func.func @host_naive_gemm
// GEMM-SCALED-TRANSA-SAME: (%[[A:.*]]: memref<4096xf4E2M1FN>, %[[B:.*]]: memref<4096xf4E2M1FN>, %[[C:.*]]: memref<256xf32>, %[[SCALEA:.*]]: memref<128xf8E8M0FNU>, %[[SCALEB:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-TRANSA: %[[A_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-TRANSA: call @_memcpy_f4E2M1FN_f32_4096(%[[A]], %[[A_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-TRANSA: %[[B_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-TRANSA: call @_memcpy_f4E2M1FN_f32_4096(%[[B]], %[[B_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-TRANSA: %[[A_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED-TRANSA: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEA]], %[[A_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED-TRANSA: %[[B_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED-TRANSA: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEB]], %[[B_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED-TRANSA: %[[A_EXPAND:.*]] = memref.expand_shape %[[A_ALLOC]]
// GEMM-SCALED-TRANSA-SAME: memref<4096xf32> into memref<16x256xf32>
// GEMM-SCALED-TRANSA: %[[B_EXPAND:.*]] = memref.expand_shape %[[B_ALLOC]] 
// GEMM-SCALED-TRANSA-SAME: memref<4096xf32> into memref<256x16xf32>
// GEMM-SCALED-TRANSA: %[[A_SCALE_EXPAND:.*]] = memref.expand_shape %[[A_SCALE_ALLOC]] 
// GEMM-SCALED-TRANSA-SAME: memref<128xf32> into memref<8x16xf32>
// GEMM-SCALED-TRANSA: %[[B_SCALE_EXPAND:.*]] = memref.expand_shape %[[B_SCALE_ALLOC]] 
// GEMM-SCALED-TRANSA-SAME: memref<128xf32> into memref<8x16xf32>
// GEMM-SCALED-TRANSA: %[[C_EXPAND:.*]] = memref.expand_shape %[[C]] 
// GEMM-SCALED-TRANSA-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-TRANSA: linalg.generic
// GEMM-SCALED-TRANSA-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]], %[[A_SCALE_EXPAND]], %[[B_SCALE_EXPAND]] : memref<16x256xf32>, memref<256x16xf32>, memref<8x16xf32>, memref<8x16xf32>) outs(%[[C_EXPAND]] : memref<16x16xf32>) {
// GEMM-SCALED-TRANSA: (%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-TRANSA-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-TRANSA-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-TRANSA-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-TRANSA-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-TRANSA-NEXT: linalg.yield

// GEMM-SCALED-TRANSB: func.func @rock_gemm
// GEMM-SCALED-TRANSB-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf8E8M0FNU>, %[[ARG4:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-TRANSB: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-TRANSB-SAME: memref<128xf8E8M0FNU> to memref<1x16x8xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-TRANSB-SAME: memref<128xf8E8M0FNU> to memref<1x16x8xf8E8M0FNU>
// GEMM-SCALED-TRANSB-NOT: rock.transform {{.*}}AddDim
// GEMM-SCALED-TRANSB-NOT: rock.transform {{.*}}Broadcast
// GEMM-SCALED-TRANSB: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_EXPAND]] * %{{.*}} scaled by tr %[[SCALEB_EXPAND]]
// GEMM-SCALED-TRANSB-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x8xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x16x8xf8E8M0FNU>

// GEMM-SCALED-TRANSB: func.func @host_naive_gemm
// GEMM-SCALED-TRANSB-SAME: (%[[A:.*]]: memref<4096xf4E2M1FN>, %[[B:.*]]: memref<4096xf4E2M1FN>, %[[C:.*]]: memref<256xf32>, %[[SCALEA:.*]]: memref<128xf8E8M0FNU>, %[[SCALEB:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-TRANSB: %[[A_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-TRANSB: call @_memcpy_f4E2M1FN_f32_4096(%[[A]], %[[A_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-TRANSB: %[[B_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-TRANSB: call @_memcpy_f4E2M1FN_f32_4096(%[[B]], %[[B_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-TRANSB: %[[A_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED-TRANSB: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEA]], %[[A_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED-TRANSB: %[[B_SCALE_ALLOC:.*]] = memref.alloc() : memref<128xf32>
// GEMM-SCALED-TRANSB: call @_memcpy_f8E8M0FNU_f32_128(%[[SCALEB]], %[[B_SCALE_ALLOC]]) : (memref<128xf8E8M0FNU>, memref<128xf32>) -> ()
// GEMM-SCALED-TRANSB: %[[A_EXPAND:.*]] = memref.expand_shape %[[A_ALLOC]]
// GEMM-SCALED-TRANSB-SAME: memref<4096xf32> into memref<16x256xf32>
// GEMM-SCALED-TRANSB: %[[B_EXPAND:.*]] = memref.expand_shape %[[B_ALLOC]] 
// GEMM-SCALED-TRANSB-SAME: memref<4096xf32> into memref<256x16xf32>
// GEMM-SCALED-TRANSB: %[[A_SCALE_EXPAND:.*]] = memref.expand_shape %[[A_SCALE_ALLOC]] 
// GEMM-SCALED-TRANSB-SAME: memref<128xf32> into memref<16x8xf32>
// GEMM-SCALED-TRANSB: %[[B_SCALE_EXPAND:.*]] = memref.expand_shape %[[B_SCALE_ALLOC]] 
// GEMM-SCALED-TRANSB-SAME: memref<128xf32> into memref<16x8xf32>
// GEMM-SCALED-TRANSB: %[[C_EXPAND:.*]] = memref.expand_shape %[[C]] 
// GEMM-SCALED-TRANSB-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-TRANSB: linalg.generic
// GEMM-SCALED-TRANSB-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]], %[[A_SCALE_EXPAND]], %[[B_SCALE_EXPAND]] : memref<16x256xf32>, memref<256x16xf32>, memref<16x8xf32>, memref<16x8xf32>) outs(%[[C_EXPAND]] : memref<16x16xf32>) {
// GEMM-SCALED-TRANSB: (%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-TRANSB-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-TRANSB-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-TRANSB-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-TRANSB-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-TRANSB-NEXT: linalg.yield

// GEMM-SCALED-F32: func.func @rock_gemm
// GEMM-SCALED-F32-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf32>, %[[ARG4:.*]]: memref<128xf32>)
// GEMM-SCALED-F32: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-F32-SAME: memref<128xf32> to memref<1x16x8xf32>
// GEMM-SCALED-F32: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-F32-SAME: memref<128xf32> to memref<1x8x16xf32>
// GEMM-SCALED-F32-NOT: rock.transform {{.*}}AddDim
// GEMM-SCALED-F32-NOT: rock.transform {{.*}}Broadcast
// GEMM-SCALED-F32: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_EXPAND]] * %{{.*}} scaled by %[[SCALEB_EXPAND]]
// GEMM-SCALED-F32-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x8xf32> * memref<1x256x16xf4E2M1FN> scaled by memref<1x8x16xf32>

// GEMM-SCALED-F32: func.func @host_naive_gemm
// GEMM-SCALED-F32-SAME: (%[[A:.*]]: memref<4096xf4E2M1FN>, %[[B:.*]]: memref<4096xf4E2M1FN>, %[[C:.*]]: memref<256xf32>, %[[SCALEA:.*]]: memref<128xf32>, %[[SCALEB:.*]]: memref<128xf32>)
// GEMM-SCALED-F32: %[[A_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-F32: call @_memcpy_f4E2M1FN_f32_4096(%[[A]], %[[A_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-F32: %[[B_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-F32: call @_memcpy_f4E2M1FN_f32_4096(%[[B]], %[[B_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-F32: %[[A_EXPAND:.*]] = memref.expand_shape %[[A_ALLOC]]
// GEMM-SCALED-F32-SAME: memref<4096xf32> into memref<16x256xf32>
// GEMM-SCALED-F32: %[[B_EXPAND:.*]] = memref.expand_shape %[[B_ALLOC]] 
// GEMM-SCALED-F32-SAME: memref<4096xf32> into memref<256x16xf32>
// GEMM-SCALED-F32: %[[A_SCALE_EXPAND:.*]] = memref.expand_shape %[[SCALEA]] 
// GEMM-SCALED-F32-SAME: memref<128xf32> into memref<16x8xf32>
// GEMM-SCALED-F32: %[[B_SCALE_EXPAND:.*]] = memref.expand_shape %[[SCALEB]] 
// GEMM-SCALED-F32-SAME: memref<128xf32> into memref<8x16xf32>
// GEMM-SCALED-F32: %[[C_EXPAND:.*]] = memref.expand_shape %[[C]] 
// GEMM-SCALED-F32-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-F32: linalg.generic
// GEMM-SCALED-F32-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]], %[[A_SCALE_EXPAND]], %[[B_SCALE_EXPAND]] : memref<16x256xf32>, memref<256x16xf32>, memref<16x8xf32>, memref<8x16xf32>) outs(%[[C_EXPAND]] : memref<16x16xf32>) {
// GEMM-SCALED-F32: (%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-F32-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-F32-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-F32-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-F32-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-F32-NEXT: linalg.yield

// GEMM-SCALED-BCAST: func.func @rock_gemm
// GEMM-SCALED-BCAST-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf8E8M0FNU>, %[[ARG4:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-BCAST: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-BCAST-SAME: memref<128xf8E8M0FNU> to memref<1x16x8xf8E8M0FNU>
// GEMM-SCALED-BCAST: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-BCAST-SAME: memref<128xf8E8M0FNU> to memref<1x8x16xf8E8M0FNU>
// GEMM-SCALED-BCAST: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-BCAST-SAME: memref<1x16x8xf8E8M0FNU> to memref<1x16x8x1xf8E8M0FNU>
// GEMM-SCALED-BCAST: %[[SCALEA_BROADCAST:.*]] = rock.transform %[[SCALEA_ADDDIM]]
// GEMM-SCALED-BCAST-SAME: memref<1x16x8x1xf8E8M0FNU> to memref<1x16x8x32xf8E8M0FNU>
// GEMM-SCALED-BCAST: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST]]
// GEMM-SCALED-BCAST-SAME: memref<1x16x8x32xf8E8M0FNU> to memref<1x16x256xf8E8M0FNU>
// GEMM-SCALED-BCAST: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-BCAST-SAME: memref<1x8x16xf8E8M0FNU> to memref<1x8x1x16xf8E8M0FNU>
// GEMM-SCALED-BCAST: %[[SCALEB_BROADCAST:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-BCAST-SAME: memref<1x8x1x16xf8E8M0FNU> to memref<1x8x32x16xf8E8M0FNU>
// GEMM-SCALED-BCAST: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST]]
// GEMM-SCALED-BCAST-SAME: memref<1x8x32x16xf8E8M0FNU> to memref<1x256x16xf8E8M0FNU>
// GEMM-SCALED-BCAST: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_MERGE]] * %{{.*}} scaled by %[[SCALEB_MERGE]]
// GEMM-SCALED-BCAST-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x256xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x256x16xf8E8M0FNU>
