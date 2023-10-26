// RUN: rocmlir-opt --rock-blockwise-gemm-to-threadwise --rock-threadwise-gemm-lowering %s | FileCheck %s
#map5 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map6 = affine_map<(d0, d1) -> (d1, d0)>
#map7 = affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 8 + d2)>
#map8 = affine_map<(d0, d1) -> (0, d1, d0)>

#transform_map5 = #rock.transform_map<#map5 by [<Unmerge{2, 8} ["m_iter", "k_iter"] at [0, 1] -> ["iter"] at [0]>] bounds = [2, 8] -> [16]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{8} ["k"] at [0] -> ["k_iter"] at [1]>, <Merge{2} ["m"] at [1] -> ["m_iter"] at [0]>] bounds = [8, 2] -> [2, 8]>
#transform_map7 = #rock.transform_map<#map7 by [<Unmerge{1, 2, 8} ["kouterPerThread", "m_iter", "kpackPerThread"] at [0, 1, 2] -> ["iter"] at [0]>] bounds = [1, 2, 8] -> [16]>
#transform_map8 = #rock.transform_map<#map8 by [<Merge{1, 8} ["k"] at [0] -> ["kouterPerThread", "kpackPerThread"] at [0, 2]>, <Merge{2} ["m"] at [1] -> ["m_iter"] at [1]>] bounds = [8, 2] -> [1, 2, 8]>


// CHECK-LABEL: func.func @rock_threadwise_memcopy
func.func @rock_threadwise_memcopy(%input : memref<16xf16, #gpu.address_space<private>>,  %output : memref<16xf16, #gpu.address_space<private>>)  {
    // source
    %21 = rock.transform %input by #transform_map5 : memref<16xf16, #gpu.address_space<private>> to memref<2x8xf16, #gpu.address_space<private>>
    %22 = rock.transform %21 by #transform_map6 : memref<2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>

    // dest
    %23 = rock.transform %output by #transform_map7 : memref<16xf16, #gpu.address_space<private>> to memref<1x2x8xf16, #gpu.address_space<private>>
    %24 = rock.transform %23 by #transform_map8 : memref<1x2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>

    // copy from source to dest
    // CHECK: rock.transforming_for
    // CHECK: strides [8]
    rock.threadwise_copy %22 -> %24 : memref<8x2xf16, #gpu.address_space<private>> -> memref<8x2xf16, #gpu.address_space<private>>
    return
}

// If we use an embed instead of a Unmerge, we cannot invert the map. This should still work, but we won't be able to run the vectorizer
#transform_map7_noinv = #rock.transform_map<#map7 by [<Embed{16, 8, 1} ["kouterPerThread", "m_iter", "kpackPerThread"] at [0, 1, 2] -> ["iter"] at [0]>] bounds = [1, 2, 8] -> [16]>
// CHECK-LABEL: func.func @rock_threadwise_memcopy_no_inversion
func.func @rock_threadwise_memcopy_no_inversion(%input : memref<16xf16, #gpu.address_space<private>>,  %output : memref<16xf16, #gpu.address_space<private>>)  {
    // source
    %21 = rock.transform %input by #transform_map5 : memref<16xf16, #gpu.address_space<private>> to memref<2x8xf16, #gpu.address_space<private>>
    %22 = rock.transform %21 by #transform_map6 : memref<2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>

    // dest
    %23 = rock.transform %output by #transform_map7_noinv : memref<16xf16, #gpu.address_space<private>> to memref<1x2x8xf16, #gpu.address_space<private>>
    %24 = rock.transform %23 by #transform_map8 : memref<1x2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>

    // copy from source to dest
    // CHECK: rock.transforming_for
    // CHECK: strides [1, 1]
    rock.threadwise_copy %22 -> %24 : memref<8x2xf16, #gpu.address_space<private>> -> memref<8x2xf16, #gpu.address_space<private>>
    return
}
