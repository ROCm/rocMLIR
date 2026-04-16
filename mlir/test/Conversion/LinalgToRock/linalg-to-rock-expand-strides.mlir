// RUN:  sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @mlir_dot_log
// CHECK-SAME: (%[[arg0:.*]]: tensor<1536xf16>, %[[arg1:.*]]: tensor<1536xf16>)
func.func @mlir_dot_log(%arg0: tensor<1536xf16>, %arg1: tensor<1536xf16>) -> tensor<4608xf16> attributes {rock.kernel, rock.arch="##TOKEN_ARCH##"} {
  //   CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
  //   CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [4, 16, 24] : tensor<1536xf16> into tensor<4x16x24xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [4, 24, 16] : tensor<1536xf16> into tensor<4x24x16xf16>
  //   CHECK-DAG: %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<4x24x24xf16>
  %cst = arith.constant dense<0.000000e+00> : tensor<4x24x24xf16>
  //   CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor() : tensor<4x24x24xf16>
  //   CHECK-DAG: %[[gemm:.*]] = rock.gemm %[[alloc]] = %[[expanded_0]] * %[[expanded]]{{.*}}storeMethod
  %0 = linalg.batch_matmul ins(%expanded_0, %expanded : tensor<4x24x16xf16>, tensor<4x16x24xf16>) outs(%cst : tensor<4x24x24xf16>) -> tensor<4x24x24xf16>
  //   CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4x24x24xf16>
  //   CHECK-DAG: %[[log:.*]] = linalg.log ins(%[[gemm]]{{.*}}) outs(%[[empty]]{{.*}}) -> tensor<4x24x24xf16>
  %1 = tensor.empty() : tensor<4x24x24xf16>
  %2 = linalg.log ins(%0 : tensor<4x24x24xf16>) outs(%1 : tensor<4x24x24xf16>) -> tensor<4x24x24xf16>
  //   CHECK-DAG: %[[alloc2:.*]] = bufferization.alloc_tensor() : tensor<4x48x24xf16>
  //   CHECK-DAG: %[[expand:.*]] = rock.expand_strides %[[log]] into %[[alloc2]]
  %3 = tensor.empty() : tensor<4x48x24xf16>
  %inserted_slice = tensor.insert_slice %2 into %3[0, 0, 0] [4, 24, 24] [1, 1, 1] {rock.is_expand_strides}: tensor<4x24x24xf16> into tensor<4x48x24xf16>
  //   CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[expand]]
  //   CHECK-DAG: return %[[collapsed]]
  %collapsed = tensor.collapse_shape %inserted_slice [[0, 1, 2]] : tensor<4x48x24xf16> into tensor<4608xf16>
  return %collapsed : tensor<4608xf16>
}

// -----


// CHECK-LABEL: func.func @mlir_dot_log
// CHECK-SAME: (%[[arg0:.*]]: tensor<320xf16>, %[[arg1:.*]]: tensor<1536xf16>)
func.func @mlir_dot_log(%arg0: tensor<320xf16>, %arg1: tensor<1536xf16>) -> tensor<1152xf16> attributes {rock.kernel, rock.arch="##TOKEN_ARCH##"} {
  //   CHECK-DAG: %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
  //   CHECK-DAG: %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [4, 16, 24] : tensor<1536xf16> into tensor<4x16x24xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [4, 5, 16] : tensor<320xf16> into tensor<4x5x16xf16>
  //   CHECK-DAG: %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<4x5x24xf16>
  %cst = arith.constant dense<0.000000e+00> : tensor<4x5x24xf16>
  //   CHECK-DAG: %[[alloc:.*]] = bufferization.alloc_tensor() : tensor<4x5x24xf16>
  //   CHECK-DAG: %[[gemm:.*]] = rock.gemm %[[alloc]] = %[[expanded_0]] * %[[expanded]]{{.*}}storeMethod
  %0 = linalg.batch_matmul ins(%expanded_0, %expanded : tensor<4x5x16xf16>, tensor<4x16x24xf16>) outs(%cst : tensor<4x5x24xf16>) -> tensor<4x5x24xf16>
  //   CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4x5x24xf16>
  //   CHECK-DAG: %[[log:.*]] = linalg.log ins(%[[gemm]]{{.*}}) outs(%[[empty]]{{.*}}) -> tensor<4x5x24xf16>
  %1 = tensor.empty() : tensor<4x5x24xf16>
  %2 = linalg.log ins(%0 : tensor<4x5x24xf16>) outs(%1 : tensor<4x5x24xf16>) -> tensor<4x5x24xf16>
  //   CHECK-DAG: %[[alloc2:.*]] = bufferization.alloc_tensor() : tensor<4x12x24xf16>
  //   CHECK-DAG: %[[expand:.*]] = rock.expand_strides %[[log]] into %[[alloc2]]
  %3 = tensor.empty() : tensor<4x12x24xf16>
  %inserted_slice = tensor.insert_slice %2 into %3[0, 0, 0] [4, 5, 24] [1, 1, 1] {rock.is_expand_strides} : tensor<4x5x24xf16> into tensor<4x12x24xf16>
  //   CHECK-DAG: %[[collapsed:.*]] = tensor.collapse_shape %[[expand]]
  //   CHECK-DAG: return %[[collapsed]]
  %collapsed = tensor.collapse_shape %inserted_slice [[0, 1, 2]] : tensor<4x12x24xf16> into tensor<1152xf16>
  return %collapsed : tensor<1152xf16>
}
