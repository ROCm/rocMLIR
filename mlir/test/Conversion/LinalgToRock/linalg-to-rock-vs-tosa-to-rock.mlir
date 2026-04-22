// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver --kernel-pipeline=migraphx-linalg,highlevel| FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel | FileCheck %s

// test file used to compare linalg-to-rock vs tosa-to-rock

// CHECK-LABEL: func.func @dot_3D(
// CHECK-SAME: %[[arg0:.*]]: memref{{.*}}, %[[arg1:.*]]: memref{{.*}}, %[[arg2:.*]]: memref{{.*}})
// CHECK-DAG: %[[zero:.*]] = rock.transform %[[arg1]]
// CHECK-DAG: %[[one:.*]] = rock.transform %[[arg0]]
// CHECK-DAG: %[[alloc:.*]] = memref.alloc
// CHECK-DAG: rock.gemm %[[alloc]] = %[[one]] * %[[zero]] storeMethod = set
// CHECK-DAG: %[[two:.*]] = rock.transform %[[alloc]]
func.func @dot_3D(%arg0 : !migraphx.shaped<2x3x2xf32, 6x2x1>, %arg1: !migraphx.shaped<2x2x3xf32, 6x3x1>)  -> !migraphx.shaped<2x3x3xf32, 9x3x1> attributes {rock.kernel, rock.arch="gfx950"}{
  %0 = migraphx.dot %arg0, %arg1 : <2x3x2xf32, 6x2x1>, <2x2x3xf32, 6x3x1> -> <2x3x3xf32, 9x3x1>
  func.return %0 : !migraphx.shaped<2x3x3xf32, 9x3x1> 
}
