//  miopen.in_warp_transpose requires that %lane be index, but the llvm.and
//  it expands into requires that it be an integral type.
// RUN: miopen-opt -miopen-lowering-step4 %s | FileCheck %s
func @in_warp_transpose_lowering(%arg0: vector<8xf32>) -> vector<8xf32> {
    // CHECK-LABEL: func @in_warp_transpose_lowering
    %cst64 = arith.constant 64 : index
    %workitem = miopen.workitem_id : index
    %lane = arith.remui %workitem, %cst64 : index

    %0 = miopen.in_warp_transpose { size = 4 : i32 } %arg0, %lane : vector<8xf32>, index
    // CHECK: %{{.*}} gpu.warp_swizzle {selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32]}
    // CHECK: %{{.*}} gpu.warp_swizzle {selector = [1 : i32, 0 : i32, 3 : i32, 2 : i32]}
    // CHECK: %{{.*}} gpu.warp_swizzle {selector = [2 : i32, 1 : i32, 0 : i32, 3 : i32]}
    // CHECK: %{{.*}} gpu.warp_swizzle {selector = [3 : i32, 2 : i32, 1 : i32, 0 : i32]}
    return %0 : vector<8xf32>
}

func @in_warp_transpose_lowering_2x2(%arg0: vector<8xf32>) -> vector<8xf32> {
    // CHECK-LABEL: func @in_warp_transpose_lowering_2x2
    %cst64 = arith.constant 64 : index
    %workitem = miopen.workitem_id : index
    %lane = arith.remui %workitem, %cst64 : index

    %0 = miopen.in_warp_transpose { size = 2 : i32 } %arg0, %lane : vector<8xf32>, index
    // CHECK: %{{.*}} gpu.warp_swizzle {selector = [1 : i32, 0 : i32, 3 : i32, 2 : i32]}
    return %0 : vector<8xf32>
}

func @in_warp_transpose_lowering_2x2_perm(%arg0: vector<8xf32>) -> vector<8xf32> {
    // CHECK-LABEL: func @in_warp_transpose_lowering_2x2_perm
    %cst64 = arith.constant 64 : index
    %workitem = miopen.workitem_id : index
    %lane = arith.remui %workitem, %cst64 : index

    %0 = miopen.in_warp_transpose { size = 2 : i32,
      inGroupPerm = [ 0 : i32, 2 : i32, 1 : i32, 3 : i32 ] } %arg0, %lane : vector<8xf32>, index
    // CHECK: %{{.*}} gpu.warp_swizzle {selector = [0 : i32, 2 : i32, 1 : i32, 3 : i32]}
    // CHECK: %{{.*}} gpu.warp_swizzle {selector = [1 : i32, 3 : i32, 0 : i32, 2 : i32]}
    return %0 : vector<8xf32>
}
