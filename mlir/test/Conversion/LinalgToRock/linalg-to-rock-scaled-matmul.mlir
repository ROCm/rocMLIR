// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @quant_dot_with_scales
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %arg1
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %arg0
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape
// CHECK-DAG: %[[collapsed_4:.*]] = tensor.collapse_shape
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor() : tensor<1x64x64xf32>
// CHECK: %[[gemm:.*]] = rock.gemm %[[alloc]] = %[[expanded_0]] scaled by %[[collapsed]] * %[[expanded]] scaled by %[[collapsed_4]] storeMethod =  set
// CHECK-SAME: tensor<1x64x128xf4E2M1FN> scaled by tensor<1x64x128xf8E8M0FNU>
// CHECK-SAME: tensor<1x128x64xf4E2M1FN> scaled by tensor<1x128x64xf8E8M0FNU>
// CHECK: %[[result:.*]] = tensor.collapse_shape %[[gemm]]
// CHECK: return %[[result]]
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @quant_dot_with_scales(%arg0: tensor<8192xf4E2M1FN>, %arg1: tensor<8192xf4E2M1FN>, %arg2: tensor<256xf8E8M0FNU>, %arg3: tensor<256xf8E8M0FNU>) -> tensor<4096xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 128, 64] : tensor<8192xf4E2M1FN> into tensor<1x128x64xf4E2M1FN>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 64, 128] : tensor<8192xf4E2M1FN> into tensor<1x64x128xf4E2M1FN>
    %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 64, 4] : tensor<256xf8E8M0FNU> into tensor<1x64x4xf8E8M0FNU>
    %0 = tensor.empty() : tensor<1x64x4x32xf8E8M0FNU>
    %broadcasted = linalg.broadcast ins(%expanded_1 : tensor<1x64x4xf8E8M0FNU>) outs(%0 : tensor<1x64x4x32xf8E8M0FNU>) dimensions = [3] 
    %collapsed = tensor.collapse_shape %broadcasted [[0], [1], [2, 3]] : tensor<1x64x4x32xf8E8M0FNU> into tensor<1x64x128xf8E8M0FNU>
    %expanded_2 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [1, 4, 64] : tensor<256xf8E8M0FNU> into tensor<1x4x64xf8E8M0FNU>
    %1 = tensor.empty() : tensor<1x4x32x64xf8E8M0FNU>
    %broadcasted_3 = linalg.broadcast ins(%expanded_2 : tensor<1x4x64xf8E8M0FNU>) outs(%1 : tensor<1x4x32x64xf8E8M0FNU>) dimensions = [2] 
    %collapsed_4 = tensor.collapse_shape %broadcasted_3 [[0], [1, 2], [3]] : tensor<1x4x32x64xf8E8M0FNU> into tensor<1x128x64xf8E8M0FNU>
    %2 = tensor.empty() : tensor<1x64x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map1, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expanded_0, %collapsed, %expanded, %collapsed_4 : tensor<1x64x128xf4E2M1FN>, tensor<1x64x128xf8E8M0FNU>, tensor<1x128x64xf4E2M1FN>, tensor<1x128x64xf8E8M0FNU>) outs(%2 : tensor<1x64x64xf32>) attrs =  {quant_dot = true} {
    ^bb0(%in: f4E2M1FN, %in_6: f8E8M0FNU, %in_7: f4E2M1FN, %in_8: f8E8M0FNU, %out: f32):
        %4 = arith.extf %in : f4E2M1FN to f32
        %5 = arith.extf %in_6 : f8E8M0FNU to f32
        %6 = arith.extf %in_7 : f4E2M1FN to f32
        %7 = arith.extf %in_8 : f8E8M0FNU to f32
        %8 = arith.mulf %4, %5 : f32
        %9 = arith.mulf %8, %6 : f32
        %10 = arith.mulf %9, %7 : f32
        linalg.yield %10 : f32
    } -> tensor<1x64x64xf32>
    %collapsed_5 = tensor.collapse_shape %3 [[0, 1, 2]] : tensor<1x64x64xf32> into tensor<4096xf32>
    return %collapsed_5 : tensor<4096xf32>
}

// -----

// CHECK-LABEL: func.func @quant_dot_with_scales_perf_config
// CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %arg1
// CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %arg0
// CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape
// CHECK-DAG: %[[collapsed_4:.*]] = tensor.collapse_shape
// CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor() : tensor<1x64x64xf32>
// CHECK: %[[gemm:.*]] = rock.gemm %[[alloc]] = %[[expanded_0]] scaled by %[[collapsed]] * %[[expanded]] scaled by %[[collapsed_4]] storeMethod =  set {perf_config = "test_perf_config"}
// CHECK-SAME: tensor<1x64x128xf4E2M1FN> scaled by tensor<1x64x128xf8E8M0FNU>
// CHECK-SAME: tensor<1x128x64xf4E2M1FN> scaled by tensor<1x128x64xf8E8M0FNU>
// CHECK: %[[result:.*]] = tensor.collapse_shape %[[gemm]]
// CHECK: return %[[result]]
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @quant_dot_with_scales_perf_config(%arg0: tensor<8192xf4E2M1FN>, %arg1: tensor<8192xf4E2M1FN>, %arg2: tensor<256xf8E8M0FNU>, %arg3: tensor<256xf8E8M0FNU>) -> tensor<4096xf32> attributes {arch = "##TOKEN_ARCH##", kernel} {
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 128, 64] : tensor<8192xf4E2M1FN> into tensor<1x128x64xf4E2M1FN>
    %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 64, 128] : tensor<8192xf4E2M1FN> into tensor<1x64x128xf4E2M1FN>
    %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 64, 4] : tensor<256xf8E8M0FNU> into tensor<1x64x4xf8E8M0FNU>
    %0 = tensor.empty() : tensor<1x64x4x32xf8E8M0FNU>
    %broadcasted = linalg.broadcast ins(%expanded_1 : tensor<1x64x4xf8E8M0FNU>) outs(%0 : tensor<1x64x4x32xf8E8M0FNU>) dimensions = [3] 
    %collapsed = tensor.collapse_shape %broadcasted [[0], [1], [2, 3]] : tensor<1x64x4x32xf8E8M0FNU> into tensor<1x64x128xf8E8M0FNU>
    %expanded_2 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [1, 4, 64] : tensor<256xf8E8M0FNU> into tensor<1x4x64xf8E8M0FNU>
    %1 = tensor.empty() : tensor<1x4x32x64xf8E8M0FNU>
    %broadcasted_3 = linalg.broadcast ins(%expanded_2 : tensor<1x4x64xf8E8M0FNU>) outs(%1 : tensor<1x4x32x64xf8E8M0FNU>) dimensions = [2] 
    %collapsed_4 = tensor.collapse_shape %broadcasted_3 [[0], [1, 2], [3]] : tensor<1x4x32x64xf8E8M0FNU> into tensor<1x128x64xf8E8M0FNU>
    %2 = tensor.empty() : tensor<1x64x64xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map1, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expanded_0, %collapsed, %expanded, %collapsed_4 : tensor<1x64x128xf4E2M1FN>, tensor<1x64x128xf8E8M0FNU>, tensor<1x128x64xf4E2M1FN>, tensor<1x128x64xf8E8M0FNU>) outs(%2 : tensor<1x64x64xf32>) attrs =  {perf_config = "test_perf_config", quant_dot = true} {
    ^bb0(%in: f4E2M1FN, %in_6: f8E8M0FNU, %in_7: f4E2M1FN, %in_8: f8E8M0FNU, %out: f32):
        %4 = arith.extf %in : f4E2M1FN to f32
        %5 = arith.extf %in_6 : f8E8M0FNU to f32
        %6 = arith.extf %in_7 : f4E2M1FN to f32
        %7 = arith.extf %in_8 : f8E8M0FNU to f32
        %8 = arith.mulf %4, %5 : f32
        %9 = arith.mulf %8, %6 : f32
        %10 = arith.mulf %9, %7 : f32
        linalg.yield %10 : f32
    } -> tensor<1x64x64xf32>
    %collapsed_5 = tensor.collapse_shape %3 [[0, 1, 2]] : tensor<1x64x64xf32> into tensor<4096xf32>
    return %collapsed_5 : tensor<4096xf32>
}
