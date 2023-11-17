// RUN: rocmlir-opt --rock-blockwise-gemm-to-threadwise --rock-threadwise-gemm-lowering %s | FileCheck %s
#map5 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map6 = affine_map<(d0, d1) -> (d1, d0)>
#map7 = affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 8 + d2)>
#map8 = affine_map<(d0, d1) -> (0, d1, d0)>

#map9 = affine_map<(d0, d1, d2) -> (d0, d1 * 8 + d2)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, (d1 * 2 + d2) * 8 + d3)>
#map12 = affine_map<(d0, d1, d2) -> (d0, 0, d2, d1)>

#transform_map5 = #rock.transform_map<#map5 by [<Unmerge{2, 8} ["m_iter", "k_iter"] at [0, 1] -> ["iter"] at [0]>] bounds = [2, 8] -> [16]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{8} ["k"] at [0] -> ["k_iter"] at [1]>, <Merge{2} ["m"] at [1] -> ["m_iter"] at [0]>] bounds = [8, 2] -> [2, 8]>
#transform_map7 = #rock.transform_map<#map7 by [<Unmerge{1, 2, 8} ["kouterPerThread", "m_iter", "kpackPerThread"] at [0, 1, 2] -> ["iter"] at [0]>] bounds = [1, 2, 8] -> [16]>
#transform_map8 = #rock.transform_map<#map8 by [<Merge{1, 8} ["k"] at [0] -> ["kouterPerThread", "kpackPerThread"] at [0, 2]>, <Merge{2} ["m"] at [1] -> ["m_iter"] at [1]>] bounds = [8, 2] -> [1, 2, 8]>

// Multi maps
#transform_map9 = #rock.transform_map<#map9 by [<PassThrough ["i"] at [0] -> ["i"] at [0]>, <Unmerge{2, 8} ["m_iter", "k_iter"] at [1, 2] -> ["iter"] at [1]>] bounds = [2, 2, 8] -> [2, 16]>
#transform_map10 = #rock.transform_map<#map10 by [<PassThrough ["i"] at [0] -> ["i"] at [0]>, <Merge{8} ["k"] at [1] -> ["k_iter"] at [2]>, <Merge{2} ["m"] at [2] -> ["m_iter"] at [1]>] bounds = [2, 8, 2] -> [2, 2, 8]>
#transform_map11 = #rock.transform_map<#map11 by [<PassThrough ["j"] at [0] -> ["j"] at [0]>, <Unmerge{1, 2, 8} ["kouterPerThread", "m_iter", "kpackPerThread"] at [1, 2, 3] -> ["iter"] at [1]>] bounds = [2, 1, 2, 8] -> [2, 16]>
#transform_map12 = #rock.transform_map<#map12 by [<PassThrough ["j"] at [0] -> ["j"] at [0]>, <Merge{1, 8} ["k"] at [1] -> ["kouterPerThread", "kpackPerThread"] at [1, 3]>, <Merge{2} ["m"] at [2] -> ["m_iter"] at [2]>] bounds = [2, 8, 2] -> [2, 1, 2, 8]>


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

// CHECK-LABEL: func.func @rock_threadwise_memcopy_extraindices
// CHECK: ({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[i:.*]]: index, %[[j:.*]]: index)
func.func @rock_threadwise_memcopy_extraindices(%input : memref<2x16xf16, #gpu.address_space<private>>,
                                                %input1 : memref<16xf16, #gpu.address_space<private>>,
                                                %output : memref<2x16xf16, #gpu.address_space<private>>,
                                                %output1 : memref<16xf16, #gpu.address_space<private>>, %i : index, %j : index)  {
    // source(multibuffer)
    %21 = rock.transform %input by #transform_map9 : memref<2x16xf16, #gpu.address_space<private>> to memref<2x2x8xf16, #gpu.address_space<private>>
    %22 = rock.transform %21 by #transform_map10 : memref<2x2x8xf16, #gpu.address_space<private>> to memref<2x8x2xf16, #gpu.address_space<private>>

    // source(single buffer)
    %25 = rock.transform %input1 by #transform_map5 : memref<16xf16, #gpu.address_space<private>> to memref<2x8xf16, #gpu.address_space<private>>
    %26 = rock.transform %25 by #transform_map6 : memref<2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>

    // dest
    %23 = rock.transform %output by #transform_map11 : memref<2x16xf16, #gpu.address_space<private>> to memref<2x1x2x8xf16, #gpu.address_space<private>>
    %24 = rock.transform %23 by #transform_map12 : memref<2x1x2x8xf16, #gpu.address_space<private>> to memref<2x8x2xf16, #gpu.address_space<private>>

    // dest(multibuffer/non-invertible)
    %27 = rock.transform %output1 by #transform_map7_noinv : memref<16xf16, #gpu.address_space<private>> to memref<1x2x8xf16, #gpu.address_space<private>>
    %28 = rock.transform %27 by #transform_map8 : memref<1x2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>

    // copy from source to dest
    // CHECK: %[[c00:.*]] = arith.constant 0 : index
    // CHECK: rock.transforming_for ({{.*}}) = [{{.*}}, {{.*}}, {{.*}}](%[[i]], %[[j]], %[[c00]])
    // CHECK: strides [1, 1, 8]
    rock.threadwise_copy %22[%i]-> %24[%j] : memref<2x8x2xf16, #gpu.address_space<private>> -> memref<2x8x2xf16, #gpu.address_space<private>>

    // CHECK: %[[c01:.*]] = arith.constant 0 : index
    // CHECK: rock.transforming_for ({{.*}}) = [{{.*}}, {{.*}}](%[[j]], %[[c01]])
    // CHECK: strides [1, 8]
    rock.threadwise_copy %26 -> %24[%j] : memref<8x2xf16, #gpu.address_space<private>> -> memref<2x8x2xf16, #gpu.address_space<private>>

    // CHECK: %[[c02:.*]] = arith.constant 0 : index
    // CHECK: rock.transforming_for ({{.*}}) = [{{.*}}, {{.*}}](%[[i]], %[[c02]], %[[c02]])
    // CHECK: strides [1, 1, 1]
    rock.threadwise_copy %22[%i]-> %28 : memref<2x8x2xf16, #gpu.address_space<private>> -> memref<8x2xf16, #gpu.address_space<private>>

    return
}
