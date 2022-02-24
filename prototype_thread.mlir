#map = affine_map<(d0) -> (d0 + 512)>
module {
  func @miopen_xdlops_gemm_v2_one_result(%arg0: memref<1024xi8, 3>, %arg1: memref<1024xi8, #map, 3>, %arg2: memref<2xvector<4xi8>, 5>, %arg3: memref<2xvector<4xi8>, 5>) -> vector<16xi32> {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant dense<0> : vector<4xi8>
    %cst_0 = arith.constant dense<0> : vector<16xi32>
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %0 = miopen.workitem_id : index
    %1 = arith.remui %0, %c64 : index
    %2 = arith.divui %1, %c32 : index
    %3 = arith.remui %1, %c32 : index
    affine.for %arg4 = 0 to 1 {
      %5 = arith.muli %arg4, %c2 : index
      %6 = arith.addi %5, %2 : index
      %7 = arith.muli %6, %c64 : index
      %8 = arith.addi %7, %3 : index
      %9 = memref.load %arg0[%8] : memref<1024xi8, 3>
      %10 = vector.insertelement %9, %cst[%c0 : index] : vector<4xi8>
      %11 = arith.addi %8, %c1 : index
      %12 = memref.load %arg0[%11] : memref<1024xi8, 3>
      %13 = vector.insertelement %12, %10[%c1 : index] : vector<4xi8>
      %14 = arith.addi %8, %c2 : index
      %15 = memref.load %arg0[%14] : memref<1024xi8, 3>
      %16 = vector.insertelement %15, %13[%c2 : index] : vector<4xi8>
      %17 = arith.addi %8, %c3 : index
      %18 = memref.load %arg0[%17] : memref<1024xi8, 3>
      %19 = vector.insertelement %18, %16[%c3 : index] : vector<4xi8>
      memref.store %19, %arg2[%arg4] : memref<2xvector<4xi8>, 5>
      %20 = arith.muli %arg4, %c2 : index
      %21 = arith.addi %20, %2 : index
      %22 = arith.muli %21, %c64 : index
      %23 = arith.addi %22, %3 : index
      %24 = arith.addi %23, %c512 : index
      %25 = memref.load %arg1[%24] : memref<1024xi8, #map, 3>
      %26 = vector.insertelement %25, %cst[%c0 : index] : vector<4xi8>
      %27 = arith.addi %24, %c1 : index
      %28 = memref.load %arg1[%27] : memref<1024xi8, #map, 3>
      %29 = vector.insertelement %28, %26[%c1 : index] : vector<4xi8>
      %30 = arith.addi %24, %c2 : index
      %31 = memref.load %arg1[%30] : memref<1024xi8, #map, 3>
      %32 = vector.insertelement %31, %29[%c2 : index] : vector<4xi8>
      %33 = arith.addi %24, %c3 : index
      %34 = memref.load %arg1[%33] : memref<1024xi8, #map, 3>
      %35 = vector.insertelement %34, %32[%c3 : index] : vector<4xi8>
      memref.store %35, %arg3[%arg4] : memref<2xvector<4xi8>, 5>
    }
    %4 = affine.for %arg4 = 0 to 1 step 4 iter_args(%arg5 = %cst_0) -> (vector<16xi32>) {
      %5 = arith.muli %arg4, %c8 : index
      %6 = affine.for %arg6 = 0 to 8 step 4 iter_args(%arg7 = %arg5) -> (vector<16xi32>) {
        %7 = arith.addi %5, %arg6 : index
        %8 = vector.transfer_read %arg2[%7], %cst : memref<2xvector<4xi8>, 5>, vector<4xi8>
        %9 = vector.transfer_read %arg3[%7], %cst : memref<2xvector<4xi8>, 5>, vector<4xi8>
        %10 = miopen.mfma_v2(%8, %9, %arg7) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_i32_32x32x8i8"} : vector<4xi8>, vector<16xi32>
        affine.yield %10 : vector<16xi32>
      }
      affine.yield %6 : vector<16xi32>
    }
    return %4 : vector<16xi32>
  }
}

