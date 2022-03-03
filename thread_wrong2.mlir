Invoke XDLOPS code selection logic:
dataType: i8
MPerWave: 32
NPerWave: 32
argVectorType: vector<4xi8>
k_base: 4
KRepeats: 4
K: 16
bufferA type: memref<8xvector<16xi8>, 5>
bufferB type: memref<8xvector<16xi8>, 5>
Emitting load A
vector load A
vector load A
vector<16xi8>#map = affine_map<(d0) -> (d0 + 512)>
module {
  func @miopen_xdlops_gemm_v2_one_result(%arg0: memref<1024xi8, 3>, %arg1: memref<1024xi8, #map, 3>, %arg2: memref<8xvector<16xi8>, 5>, %arg3: memref<8xvector<16xi8>, 5>) -> vector<16xi32> {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %cst = arith.constant dense<0> : vector<16xi8>
    %cst_0 = arith.constant dense<0> : vector<16xi32>
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %0 = miopen.workitem_id : index
    %1 = arith.remui %0, %c64 : index
    %2 = arith.divui %1, %c32 : index
    %3 = arith.remui %1, %c32 : index
    affine.for %arg4 = 0 to 8 {
      %5 = arith.muli %arg4, %c2 : index
      %6 = arith.addi %5, %2 : index
      %7 = arith.muli %6, %c64 : index
      %8 = arith.addi %7, %3 : index
      %9 = memref.load %arg0[%8] : memref<1024xi8, 3>
      %10 = vector.insertelement %9, %cst[%c0 : index] : vector<16xi8>
      %11 = arith.addi %8, %c1 : index
      %12 = memref.load %arg0[%11] : memref<1024xi8, 3>
      %13 = vector.insertelement %12, %10[%c1 : index] : vector<16xi8>
      %14 = arith.addi %8, %c2 : index
      %15 = memref.load %arg0[%14] : memref<1024xi8, 3>
      %16 = vector.insertelement %15, %13[%c2 : index] : vector<16xi8>
      %17 = arith.addi %8, %c3 : index
      %18 = memref.load %arg0[%17] : memref<1024xi8, 3>
      %19 = vector.insertelement %18, %16[%c3 : index] : vector<16xi8>
      %20 = arith.addi %8, %c4 : index
      %21 = memref.load %arg0[%20] : memref<1024xi8, 3>
      %22 = vector.insertelement %21, %19[%c4 : index] : vector<16xi8>
      %23 = arith.addi %8, %c5 : index
      %24 = memref.load %arg0[%23] : memref<1024xi8, 3>
      %25 = vector.insertelement %24, %22[%c5 : index] : vector<16xi8>
      %26 = arith.addi %8, %c6 : index
      %27 = memref.load %arg0[%26] : memref<1024xi8, 3>
      %28 = vector.insertelement %27, %25[%c6 : index] : vector<16xi8>
      %29 = arith.addi %8, %c7 : index
      %30 = memref.load %arg0[%29] : memref<1024xi8, 3>
      %31 = vector.insertelement %30, %28[%c7 : index] : vector<16xi8>
      %32 = arith.addi %8, %c8 : index
      %33 = memref.load %arg0[%32] : memref<1024xi8, 3>
      %34 = vector.insertelement %33, %31[%c8 : index] : vector<16xi8>
      %35 = arith.addi %8, %c9 : index
      %36 = memref.load %arg0[%35] : memref<1024xi8, 3>
      %37 = vector.insertelement %36, %34[%c9 : index] : vector<16xi8>
      %38 = arith.addi %8, %c10 : index
      %39 = memref.load %arg0[%38] : memref<1024xi8, 3>
      %40 = vector.insertelement %39, %37[%c10 : index] : vector<16xi8>
      %41 = arith.addi %8, %c11 : index
      %42 = memref.load %arg0[%41] : memref<1024xi8, 3>
      %43 = vector.insertelement %42, %40[%c11 : index] : vector<16xi8>
      %44 = arith.addi %8, %c12 : index
      %45 = memref.load %arg0[%44] : memref<1024xi8, 3>
      %46 = vector.insertelement %45, %43[%c12 : index] : vector<16xi8>
      %47 = arith.addi %8, %c13 : index
      %48 = memref.load %arg0[%47] : memref<1024xi8, 3>
      %49 = vector.insertelement %48, %46[%c13 : index] : vector<16xi8>
      %50 = arith.addi %8, %c14 : index
      %51 = memref.load %arg0[%50] : memref<1024xi8, 3>
      %52 = vector.insertelement %51, %49[%c14 : index] : vector<16xi8>
      %53 = arith.addi %8, %c15 : index
      %54 = memref.load %arg0[%53] : memref<1024xi8, 3>
      %55 = vector.insertelement %54, %52[%c15 : index] : vector<16xi8>
      memref.store %55, %arg2[%arg4] : memref<8xvector<16xi8>, 5>
      %56 = arith.muli %arg4, %c2 : index
      %57 = arith.addi %56, %2 : index
      %58 = arith.muli %57, %c64 : index
      %59 = arith.addi %58, %3 : index
      %60 = arith.addi %59, %c512 : index
      %61 = memref.load %arg1[%60] : memref<1024xi8, #map, 3>
      %62 = vector.insertelement %61, %cst[%c0 : index] : vector<16xi8>
      %63 = arith.addi %60, %c1 : index
      %64 = memref.load %arg1[%63] : memref<1024xi8, #map, 3>
      %65 = vector.insertelement %64, %62[%c1 : index] : vector<16xi8>
      %66 = arith.addi %60, %c2 : index
      %67 = memref.load %arg1[%66] : memref<1024xi8, #map, 3>
      %68 = vector.insertelement %67, %65[%c2 : index] : vector<16xi8>
      %69 = arith.addi %60, %c3 : index
      %70 = memref.load %arg1[%69] : memref<1024xi8, #map, 3>
      %71 = vector.insertelement %70, %68[%c3 : index] : vector<16xi8>
      %72 = arith.addi %60, %c4 : index
      %73 = memref.load %arg1[%72] : memref<1024xi8, #map, 3>
      %74 = vector.insertelement %73, %71[%c4 : index] : vector<16xi8>
      %75 = arith.addi %60, %c5 : index
      %76 = memref.load %arg1[%75] : memref<1024xi8, #map, 3>
      %77 = vector.insertelement %76, %74[%c5 : index] : vector<16xi8>
      %78 = arith.addi %60, %c6 : index
      %79 = memref.load %arg1[%78] : memref<1024xi8, #map, 3>
      %80 = vector.insertelement %79, %77[%c6 : index] : vector<16xi8>
      %81 = arith.addi %60, %c7 : index
      %82 = memref.load %arg1[%81] : memref<1024xi8, #map, 3>
      %83 = vector.insertelement %82, %80[%c7 : index] : vector<16xi8>
      %84 = arith.addi %60, %c8 : index
      %85 = memref.load %arg1[%84] : memref<1024xi8, #map, 3>
      %86 = vector.insertelement %85, %83[%c8 : index] : vector<16xi8>
      %87 = arith.addi %60, %c9 : index
      %88 = memref.load %arg1[%87] : memref<1024xi8, #map, 3>
      %89 = vector.insertelement %88, %86[%c9 : index] : vector<16xi8>
      %90 = arith.addi %60, %c10 : index
      %91 = memref.load %arg1[%90] : memref<1024xi8, #map, 3>
      %92 = vector.insertelement %91, %89[%c10 : index] : vector<16xi8>
      %93 = arith.addi %60, %c11 : index
      %94 = memref.load %arg1[%93] : memref<1024xi8, #map, 3>
      %95 = vector.insertelement %94, %92[%c11 : index] : vector<16xi8>
      %96 = arith.addi %60, %c12 : index
      %97 = memref.load %arg1[%96] : memref<1024xi8, #map, 3>
      %98 = vector.insertelement %97, %95[%c12 : index] : vector<16xi8>
      %99 = arith.addi %60, %c13 : index
      %100 = memref.load %arg1[%99] : memref<1024xi8, #map, 3>
      %101 = vector.insertelement %100, %98[%c13 : index] : vector<16xi8>
      %102 = arith.addi %60, %c14 : index
      %103 = memref.load %arg1[%102] : memref<1024xi8, #map, 3>
      %104 = vector.insertelement %103, %101[%c14 : index] : vector<16xi8>
      %105 = arith.addi %60, %c15 : index
      %106 = memref.load %arg1[%105] : memref<1024xi8, #map, 3>
      %107 = vector.insertelement %106, %104[%c15 : index] : vector<16xi8>
      memref.store %107, %arg3[%arg4] : memref<8xvector<16xi8>, 5>
    }
    %4 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (vector<16xi32>) {
      %5 = arith.muli %arg4, %c16 : index
      %6 = affine.for %arg6 = 0 to 16 step 4 iter_args(%arg7 = %arg5) -> (vector<16xi32>) {
        %7 = arith.addi %5, %arg6 : index

        %cst = arith.constant dense<0> : vector<4xi8>

        %8 = vector.transfer_read %arg2[%7], %cst : memref<8xvector<16xi8>, 5>, vector<16xi8>
        %idx0 = addi %7 0
        %elem0 = vector.extractelement %8[%idx0]
        %arg0 = vector.insertelement %elem0 %cst[0]
        %idx1 = addi %7 1
        %elem1 = vector.extractelement %8[%idx1]
        %arg1 = vector.insertelement %elem1 %arg0[1]
        %idx2 = addi %7 2
        %elem2 = vector.extractelement %8[%idx2]
        %arg2 = vector.insertelement %elem2 %arg1[2]
        %idx3 = addi %7 3
        %elem3 = vector.extractelement %8[%idx3]
        %arg3 = vector.insertelement %elem3 %arg2[3]

        %9 = vector.transfer_read %arg3[%7], %cst : memref<8xvector<16xi8>, 5>, vector<16xi8>
        %10 = miopen.mfma_v2(%8, %9, %arg7) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_i32_32x32x8i8"} : vector<16xi8>, vector<16xi32>
        affine.yield %10 : vector<16xi32>
      }
      affine.yield %6 : vector<16xi32>
    }
    return %4 : vector<16xi32>
  }
}

