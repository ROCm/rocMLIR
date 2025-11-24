// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv | FileCheck %s --check-prefix=GEMM-SCALED
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -quantBlockSize 16 | FileCheck %s --check-prefix=GEMM-SCALED-16
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleA --transScaleB -pv | FileCheck %s --check-prefix=GEMM-SCALED-BOTHTRANS
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleA -pv | FileCheck %s --check-prefix=GEMM-SCALED-TRANSA
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm --transScaleB -pv | FileCheck %s --check-prefix=GEMM-SCALED-TRANSB
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -scale_a_dtype f32 -scale_b_dtype f32 | FileCheck %s --check-prefix=GEMM-SCALED-F32
// RUN: rocmlir-gen -t f4E2M1FN -m 16 -n 16 -k 256 -out_dtype f32 --scaledGemm --arch gfx950 --operation gemm -pv -scale_a_dtype f32 -scale_b_dtype f32 -quantBlockSize 16 | FileCheck %s --check-prefix=GEMM-SCALED-F32-16

// GEMM-SCALED: func.func @rock_gemm
// GEMM-SCALED-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf8E8M0FNU>, %[[ARG4:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-SAME: memref<128xf8E8M0FNU> to memref<1x16x8xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-SAME: memref<128xf8E8M0FNU> to memref<1x8x16xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]] 
// GEMM-SCALED-SAME: memref<1x16x8xf8E8M0FNU> to memref<1x16x8x1xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEA_BROADCAST:.*]] = rock.transform %[[SCALEA_ADDDIM]] 
// GEMM-SCALED-SAME: memref<1x16x8x1xf8E8M0FNU> to memref<1x16x8x32xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST]] 
// GEMM-SCALED-SAME: memref<1x16x8x32xf8E8M0FNU> to memref<1x16x256xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]] 
// GEMM-SCALED-SAME: memref<1x8x16xf8E8M0FNU> to memref<1x8x1x16xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEB_BROADCAST:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-SAME: memref<1x8x1x16xf8E8M0FNU> to memref<1x8x32x16xf8E8M0FNU>
// GEMM-SCALED: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST]]
// GEMM-SCALED-SAME: memref<1x8x32x16xf8E8M0FNU> to memref<1x256x16xf8E8M0FNU>
// GEMM-SCALED: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_MERGE]] * %{{.*}} scaled by %[[SCALEB_MERGE]]  
// GEMM-SCALED-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x256xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x256x16xf8E8M0FNU>

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

// GEMM-SCALED-16: func.func @rock_gemm
// GEMM-SCALED-16-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<256xf8E8M0FNU>, %[[ARG4:.*]]: memref<256xf8E8M0FNU>)
// GEMM-SCALED-16: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-16-SAME: memref<256xf8E8M0FNU> to memref<1x16x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-16-SAME: memref<256xf8E8M0FNU> to memref<1x16x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-16-SAME: memref<1x16x16xf8E8M0FNU> to memref<1x16x16x1xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEA_BROADCAST_BLOCK:.*]] = rock.transform %[[SCALEA_ADDDIM]]
// GEMM-SCALED-16-SAME: memref<1x16x16x1xf8E8M0FNU> to memref<1x16x16x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST_BLOCK]]
// GEMM-SCALED-16-SAME: memref<1x16x16x16xf8E8M0FNU> to memref<1x16x256xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-16-SAME: memref<1x16x16xf8E8M0FNU> to memref<1x16x1x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEB_BROADCAST_BLOCK:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-16-SAME: memref<1x16x1x16xf8E8M0FNU> to memref<1x16x16x16xf8E8M0FNU>
// GEMM-SCALED-16: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST_BLOCK]]
// GEMM-SCALED-16-SAME: memref<1x16x16x16xf8E8M0FNU> to memref<1x256x16xf8E8M0FNU>
// GEMM-SCALED-16: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_MERGE]] * %{{.*}} scaled by %[[SCALEB_MERGE]]
// GEMM-SCALED-16-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x256xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x256x16xf8E8M0FNU>

// GEMM-SCALED-16: func.func @host_naive_gemm
// GEMM-SCALED-16-SAME: (%[[A:.*]]: memref<4096xf4E2M1FN>, %[[B:.*]]: memref<4096xf4E2M1FN>, %[[C:.*]]: memref<256xf32>, %[[SCALEA:.*]]: memref<256xf8E8M0FNU>, %[[SCALEB:.*]]: memref<256xf8E8M0FNU>)
// GEMM-SCALED-16: %[[A_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-16: call @_memcpy_f4E2M1FN_f32_4096(%[[A]], %[[A_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-16: %[[B_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-16: call @_memcpy_f4E2M1FN_f32_4096(%[[B]], %[[B_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-16: %[[A_SCALE_ALLOC:.*]] = memref.alloc() : memref<256xf32>
// GEMM-SCALED-16: call @_memcpy_f8E8M0FNU_f32_256(%[[SCALEA]], %[[A_SCALE_ALLOC]]) : (memref<256xf8E8M0FNU>, memref<256xf32>) -> ()
// GEMM-SCALED-16: %[[B_SCALE_ALLOC:.*]] = memref.alloc() : memref<256xf32>
// GEMM-SCALED-16: call @_memcpy_f8E8M0FNU_f32_256(%[[SCALEB]], %[[B_SCALE_ALLOC]]) : (memref<256xf8E8M0FNU>, memref<256xf32>) -> ()
// GEMM-SCALED-16: %[[A_EXPAND:.*]] = memref.expand_shape %[[A_ALLOC]]
// GEMM-SCALED-16-SAME: memref<4096xf32> into memref<16x256xf32>
// GEMM-SCALED-16: %[[B_EXPAND:.*]] = memref.expand_shape %[[B_ALLOC]]
// GEMM-SCALED-16-SAME: memref<4096xf32> into memref<256x16xf32>
// GEMM-SCALED-16: %[[A_SCALE_EXPAND:.*]] = memref.expand_shape %[[A_SCALE_ALLOC]]
// GEMM-SCALED-16-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-16: %[[B_SCALE_EXPAND:.*]] = memref.expand_shape %[[B_SCALE_ALLOC]]
// GEMM-SCALED-16-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-16: %[[C_EXPAND:.*]] = memref.expand_shape %[[C]]
// GEMM-SCALED-16-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-16: linalg.generic
// GEMM-SCALED-16-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]], %[[A_SCALE_EXPAND]], %[[B_SCALE_EXPAND]] : memref<16x256xf32>, memref<256x16xf32>, memref<16x16xf32>, memref<16x16xf32>) outs(%[[C_EXPAND]] : memref<16x16xf32>) {
// GEMM-SCALED-16: (%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-16-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-16-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-16-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-16-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-16-NEXT: linalg.yield

// GEMM-SCALED-BOTHTRANS: func.func @rock_gemm
// GEMM-SCALED-BOTHTRANS-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<128xf8E8M0FNU>, %[[ARG4:.*]]: memref<128xf8E8M0FNU>)
// GEMM-SCALED-BOTHTRANS: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<128xf8E8M0FNU> to memref<1x8x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<128xf8E8M0FNU> to memref<1x16x8xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x8x16xf8E8M0FNU> to memref<1x8x1x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEA_BROADCAST:.*]] = rock.transform %[[SCALEA_ADDDIM]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x8x1x16xf8E8M0FNU> to memref<1x8x32x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x8x32x16xf8E8M0FNU> to memref<1x256x16xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]] 
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x16x8xf8E8M0FNU> to memref<1x16x8x1xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEB_BROADCAST:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x16x8x1xf8E8M0FNU> to memref<1x16x8x32xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST]]
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x16x8x32xf8E8M0FNU> to memref<1x16x256xf8E8M0FNU>
// GEMM-SCALED-BOTHTRANS: rock.gemm %{{.*}} = %{{.*}} scaled by tr %[[SCALEA_MERGE]] * %{{.*}} scaled by tr %[[SCALEB_MERGE]]  
// GEMM-SCALED-BOTHTRANS-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x256x16xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x16x256xf8E8M0FNU>

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
// GEMM-SCALED-TRANSA: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]] 
// GEMM-SCALED-TRANSA-SAME: memref<1x8x16xf8E8M0FNU> to memref<1x8x1x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEA_BROADCAST:.*]] = rock.transform %[[SCALEA_ADDDIM]] 
// GEMM-SCALED-TRANSA-SAME: memref<1x8x1x16xf8E8M0FNU> to memref<1x8x32x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST]] 
// GEMM-SCALED-TRANSA-SAME: memref<1x8x32x16xf8E8M0FNU> to memref<1x256x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]] 
// GEMM-SCALED-TRANSA-SAME: memref<1x8x16xf8E8M0FNU> to memref<1x8x1x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEB_BROADCAST:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-TRANSA-SAME: memref<1x8x1x16xf8E8M0FNU> to memref<1x8x32x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST]]
// GEMM-SCALED-TRANSA-SAME: memref<1x8x32x16xf8E8M0FNU> to memref<1x256x16xf8E8M0FNU>
// GEMM-SCALED-TRANSA: rock.gemm %{{.*}} = %{{.*}} scaled by tr %[[SCALEA_MERGE]] * %{{.*}} scaled by %[[SCALEB_MERGE]]  
// GEMM-SCALED-TRANSA-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x256x16xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x256x16xf8E8M0FNU>

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
// GEMM-SCALED-TRANSB: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]] 
// GEMM-SCALED-TRANSB-SAME: memref<1x16x8xf8E8M0FNU> to memref<1x16x8x1xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEA_BROADCAST:.*]] = rock.transform %[[SCALEA_ADDDIM]] 
// GEMM-SCALED-TRANSB-SAME: memref<1x16x8x1xf8E8M0FNU> to memref<1x16x8x32xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST]] 
// GEMM-SCALED-TRANSB-SAME: memref<1x16x8x32xf8E8M0FNU> to memref<1x16x256xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]] 
// GEMM-SCALED-TRANSB-SAME: memref<1x16x8xf8E8M0FNU> to memref<1x16x8x1xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEB_BROADCAST:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-TRANSB-SAME: memref<1x16x8x1xf8E8M0FNU> to memref<1x16x8x32xf8E8M0FNU>
// GEMM-SCALED-TRANSB: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST]]
// GEMM-SCALED-TRANSB-SAME: memref<1x16x8x32xf8E8M0FNU> to memref<1x16x256xf8E8M0FNU>
// GEMM-SCALED-TRANSB: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_MERGE]] * %{{.*}} scaled by tr %[[SCALEB_MERGE]]  
// GEMM-SCALED-TRANSB-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x256xf8E8M0FNU> * memref<1x256x16xf4E2M1FN> scaled by memref<1x16x256xf8E8M0FNU>

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
// GEMM-SCALED-F32: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]] 
// GEMM-SCALED-F32-SAME: memref<1x16x8xf32> to memref<1x16x8x1xf32>
// GEMM-SCALED-F32: %[[SCALEA_BROADCAST:.*]] = rock.transform %[[SCALEA_ADDDIM]] 
// GEMM-SCALED-F32-SAME: memref<1x16x8x1xf32> to memref<1x16x8x32xf32>
// GEMM-SCALED-F32: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST]] 
// GEMM-SCALED-F32-SAME: memref<1x16x8x32xf32> to memref<1x16x256xf32>
// GEMM-SCALED-F32: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]] 
// GEMM-SCALED-F32-SAME: memref<1x8x16xf32> to memref<1x8x1x16xf32>
// GEMM-SCALED-F32: %[[SCALEB_BROADCAST:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-F32-SAME: memref<1x8x1x16xf32> to memref<1x8x32x16xf32>
// GEMM-SCALED-F32: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST]]
// GEMM-SCALED-F32-SAME: memref<1x8x32x16xf32> to memref<1x256x16xf32>
// GEMM-SCALED-F32: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_MERGE]] * %{{.*}} scaled by %[[SCALEB_MERGE]]  
// GEMM-SCALED-F32-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x256xf32> * memref<1x256x16xf4E2M1FN> scaled by memref<1x256x16xf32>

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

// GEMM-SCALED-F32-16: func.func @rock_gemm
// GEMM-SCALED-F32-16-SAME: (%[[ARG0:.*]]: memref<4096xf4E2M1FN>, %[[ARG1:.*]]: memref<4096xf4E2M1FN>, %[[ARG2:.*]]: memref<256xf32>, %[[ARG3:.*]]: memref<256xf32>, %[[ARG4:.*]]: memref<256xf32>)
// GEMM-SCALED-F32-16: %[[SCALEA_EXPAND:.*]] = rock.transform %[[ARG3]]
// GEMM-SCALED-F32-16-SAME: memref<256xf32> to memref<1x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEB_EXPAND:.*]] = rock.transform %[[ARG4]]
// GEMM-SCALED-F32-16-SAME: memref<256xf32> to memref<1x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEA_ADDDIM:.*]] = rock.transform %[[SCALEA_EXPAND]]
// GEMM-SCALED-F32-16-SAME: memref<1x16x16xf32> to memref<1x16x16x1xf32>
// GEMM-SCALED-F32-16: %[[SCALEA_BROADCAST_BLOCK:.*]] = rock.transform %[[SCALEA_ADDDIM]]
// GEMM-SCALED-F32-16-SAME: memref<1x16x16x1xf32> to memref<1x16x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEA_MERGE:.*]] = rock.transform %[[SCALEA_BROADCAST_BLOCK]]
// GEMM-SCALED-F32-16-SAME: memref<1x16x16x16xf32> to memref<1x16x256xf32>
// GEMM-SCALED-F32-16: %[[SCALEB_ADDDIM:.*]] = rock.transform %[[SCALEB_EXPAND]]
// GEMM-SCALED-F32-16-SAME: memref<1x16x16xf32> to memref<1x16x1x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEB_BROADCAST_BLOCK:.*]] = rock.transform %[[SCALEB_ADDDIM]]
// GEMM-SCALED-F32-16-SAME: memref<1x16x1x16xf32> to memref<1x16x16x16xf32>
// GEMM-SCALED-F32-16: %[[SCALEB_MERGE:.*]] = rock.transform %[[SCALEB_BROADCAST_BLOCK]]
// GEMM-SCALED-F32-16-SAME: memref<1x16x16x16xf32> to memref<1x256x16xf32>
// GEMM-SCALED-F32-16: rock.gemm %{{.*}} = %{{.*}} scaled by %[[SCALEA_MERGE]] * %{{.*}} scaled by %[[SCALEB_MERGE]]
// GEMM-SCALED-F32-16-SAME: memref<1x16x16xf32> = memref<1x16x256xf4E2M1FN> scaled by memref<1x16x256xf32> * memref<1x256x16xf4E2M1FN> scaled by memref<1x256x16xf32>

// GEMM-SCALED-F32-16: func.func @host_naive_gemm
// GEMM-SCALED-F32-16-SAME: (%[[A:.*]]: memref<4096xf4E2M1FN>, %[[B:.*]]: memref<4096xf4E2M1FN>, %[[C:.*]]: memref<256xf32>, %[[SCALEA:.*]]: memref<256xf32>, %[[SCALEB:.*]]: memref<256xf32>)
// GEMM-SCALED-F32-16: %[[A_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-F32-16: call @_memcpy_f4E2M1FN_f32_4096(%[[A]], %[[A_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-F32-16: %[[B_ALLOC:.*]] = memref.alloc() : memref<4096xf32>
// GEMM-SCALED-F32-16: call @_memcpy_f4E2M1FN_f32_4096(%[[B]], %[[B_ALLOC]]) : (memref<4096xf4E2M1FN>, memref<4096xf32>) -> ()
// GEMM-SCALED-F32-16: %[[A_EXPAND:.*]] = memref.expand_shape %[[A_ALLOC]]
// GEMM-SCALED-F32-16-SAME: memref<4096xf32> into memref<16x256xf32>
// GEMM-SCALED-F32-16: %[[B_EXPAND:.*]] = memref.expand_shape %[[B_ALLOC]]
// GEMM-SCALED-F32-16-SAME: memref<4096xf32> into memref<256x16xf32>
// GEMM-SCALED-F32-16: %[[A_SCALE_EXPAND:.*]] = memref.expand_shape %[[SCALEA]]
// GEMM-SCALED-F32-16-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-F32-16: %[[B_SCALE_EXPAND:.*]] = memref.expand_shape %[[SCALEB]]
// GEMM-SCALED-F32-16-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-F32-16: %[[C_EXPAND:.*]] = memref.expand_shape %[[C]]
// GEMM-SCALED-F32-16-SAME: memref<256xf32> into memref<16x16xf32>
// GEMM-SCALED-F32-16: linalg.generic
// GEMM-SCALED-F32-16-SAME: ins(%[[A_EXPAND]], %[[B_EXPAND]], %[[A_SCALE_EXPAND]], %[[B_SCALE_EXPAND]] : memref<16x256xf32>, memref<256x16xf32>, memref<16x16xf32>, memref<16x16xf32>) outs(%[[C_EXPAND]] : memref<16x16xf32>) {
// GEMM-SCALED-F32-16: (%[[A_IN:.*]]: f32, %[[B_IN:.*]]: f32, %[[A_SCALE_IN:.*]]: f32, %[[B_SCALE_IN:.*]]: f32, %[[C_OUT:.*]]: f32):
// GEMM-SCALED-F32-16-NEXT: %[[A_MUL:.*]] = arith.mulf %[[A_IN]], %[[A_SCALE_IN]] : f32
// GEMM-SCALED-F32-16-NEXT: %[[B_MUL:.*]] = arith.mulf %[[B_IN]], %[[B_SCALE_IN]] : f32
// GEMM-SCALED-F32-16-NEXT: %[[MUL_OUT:.*]] = arith.mulf %[[A_MUL]], %[[B_MUL]] : f32
// GEMM-SCALED-F32-16-NEXT: arith.addf %[[MUL_OUT]], %[[C_OUT]] : f32
// GEMM-SCALED-F32-16-NEXT: linalg.yield
