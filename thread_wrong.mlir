module {
  func @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x1024x1024x1x1xi8>, %arg1: memref<128x1x1024x14x14xi8>, %arg2: memref<128x1x1024x14x14xi32>) attributes {block_size = 256 : i32, grid_size = 6272 : i32, kernel = 0 : i32} {
    %cst = arith.constant dense<0> : vector<4xi8>
    %cst_0 = arith.constant dense<0> : vector<16xi8>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c6272 = arith.constant 6272 : index
    %c4 = arith.constant 4 : index
    %cst_1 = arith.constant dense<0> : vector<16xi32>
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c48 = arith.constant 48 : index
    %c80 = arith.constant 80 : index
    %c96 = arith.constant 96 : index
    %c112 = arith.constant 112 : index
    %c128 = arith.constant 128 : index
    %c144 = arith.constant 144 : index
    %c160 = arith.constant 160 : index
    %c176 = arith.constant 176 : index
    %c192 = arith.constant 192 : index
    %c208 = arith.constant 208 : index
    %c224 = arith.constant 224 : index
    %c240 = arith.constant 240 : index
    %c1024 = arith.constant 1024 : index
    %c1040 = arith.constant 1040 : index
    %c1056 = arith.constant 1056 : index
    %c1072 = arith.constant 1072 : index
    %c2048 = arith.constant 2048 : index
    %c2064 = arith.constant 2064 : index
    %c2080 = arith.constant 2080 : index
    %c2096 = arith.constant 2096 : index
    %c3072 = arith.constant 3072 : index
    %c3088 = arith.constant 3088 : index
    %c3104 = arith.constant 3104 : index
    %c3120 = arith.constant 3120 : index
    %c4096 = arith.constant 4096 : index
    %c4112 = arith.constant 4112 : index
    %c4128 = arith.constant 4128 : index
    %c4144 = arith.constant 4144 : index
    %c5120 = arith.constant 5120 : index
    %c5136 = arith.constant 5136 : index
    %c5152 = arith.constant 5152 : index
    %c5168 = arith.constant 5168 : index
    %c6144 = arith.constant 6144 : index
    %c6160 = arith.constant 6160 : index
    %c6176 = arith.constant 6176 : index
    %c6192 = arith.constant 6192 : index
    %c7168 = arith.constant 7168 : index
    %c7184 = arith.constant 7184 : index
    %c7200 = arith.constant 7200 : index
    %c7216 = arith.constant 7216 : index
    %c8192 = arith.constant 8192 : index
    %c8208 = arith.constant 8208 : index
    %c8224 = arith.constant 8224 : index
    %c8240 = arith.constant 8240 : index
    %c9216 = arith.constant 9216 : index
    %c9232 = arith.constant 9232 : index
    %c9248 = arith.constant 9248 : index
    %c9264 = arith.constant 9264 : index
    %c10240 = arith.constant 10240 : index
    %c10256 = arith.constant 10256 : index
    %c10272 = arith.constant 10272 : index
    %c10288 = arith.constant 10288 : index
    %c11264 = arith.constant 11264 : index
    %c11280 = arith.constant 11280 : index
    %c11296 = arith.constant 11296 : index
    %c11312 = arith.constant 11312 : index
    %c12288 = arith.constant 12288 : index
    %c12304 = arith.constant 12304 : index
    %c12320 = arith.constant 12320 : index
    %c12336 = arith.constant 12336 : index
    %c13312 = arith.constant 13312 : index
    %c13328 = arith.constant 13328 : index
    %c13344 = arith.constant 13344 : index
    %c13360 = arith.constant 13360 : index
    %c14336 = arith.constant 14336 : index
    %c14352 = arith.constant 14352 : index
    %c14368 = arith.constant 14368 : index
    %c14384 = arith.constant 14384 : index
    %c15360 = arith.constant 15360 : index
    %c15376 = arith.constant 15376 : index
    %c15392 = arith.constant 15392 : index
    %c15408 = arith.constant 15408 : index
    %c196 = arith.constant 196 : index
    %c-1 = arith.constant -1 : index
    %c14 = arith.constant 14 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c15 = arith.constant 15 : index
    %c17 = arith.constant 17 : index
    %c18 = arith.constant 18 : index
    %c19 = arith.constant 19 : index
    %c20 = arith.constant 20 : index
    %c21 = arith.constant 21 : index
    %c22 = arith.constant 22 : index
    %c23 = arith.constant 23 : index
    %c24 = arith.constant 24 : index
    %c25 = arith.constant 25 : index
    %c26 = arith.constant 26 : index
    %c27 = arith.constant 27 : index
    %c28 = arith.constant 28 : index
    %c29 = arith.constant 29 : index
    %c30 = arith.constant 30 : index
    %c31 = arith.constant 31 : index
    %c33 = arith.constant 33 : index
    %c34 = arith.constant 34 : index
    %c35 = arith.constant 35 : index
    %c36 = arith.constant 36 : index
    %c37 = arith.constant 37 : index
    %c38 = arith.constant 38 : index
    %c39 = arith.constant 39 : index
    %c40 = arith.constant 40 : index
    %c41 = arith.constant 41 : index
    %c42 = arith.constant 42 : index
    %c43 = arith.constant 43 : index
    %c44 = arith.constant 44 : index
    %c45 = arith.constant 45 : index
    %c46 = arith.constant 46 : index
    %c47 = arith.constant 47 : index
    %c49 = arith.constant 49 : index
    %c50 = arith.constant 50 : index
    %c51 = arith.constant 51 : index
    %c52 = arith.constant 52 : index
    %c53 = arith.constant 53 : index
    %c54 = arith.constant 54 : index
    %c55 = arith.constant 55 : index
    %c56 = arith.constant 56 : index
    %c57 = arith.constant 57 : index
    %c58 = arith.constant 58 : index
    %c59 = arith.constant 59 : index
    %c60 = arith.constant 60 : index
    %c61 = arith.constant 61 : index
    %c62 = arith.constant 62 : index
    %c63 = arith.constant 63 : index
    %c16384 = arith.constant 16384 : index
    %c1025 = arith.constant 1025 : index
    %c1026 = arith.constant 1026 : index
    %c1027 = arith.constant 1027 : index
    %c1028 = arith.constant 1028 : index
    %c1029 = arith.constant 1029 : index
    %c1030 = arith.constant 1030 : index
    %c1031 = arith.constant 1031 : index
    %c1032 = arith.constant 1032 : index
    %c1033 = arith.constant 1033 : index
    %c1034 = arith.constant 1034 : index
    %c1035 = arith.constant 1035 : index
    %c1036 = arith.constant 1036 : index
    %c1037 = arith.constant 1037 : index
    %c1038 = arith.constant 1038 : index
    %c1039 = arith.constant 1039 : index
    %c2049 = arith.constant 2049 : index
    %c2050 = arith.constant 2050 : index
    %c2051 = arith.constant 2051 : index
    %c2052 = arith.constant 2052 : index
    %c2053 = arith.constant 2053 : index
    %c2054 = arith.constant 2054 : index
    %c2055 = arith.constant 2055 : index
    %c2056 = arith.constant 2056 : index
    %c2057 = arith.constant 2057 : index
    %c2058 = arith.constant 2058 : index
    %c2059 = arith.constant 2059 : index
    %c2060 = arith.constant 2060 : index
    %c2061 = arith.constant 2061 : index
    %c2062 = arith.constant 2062 : index
    %c2063 = arith.constant 2063 : index
    %c3073 = arith.constant 3073 : index
    %c3074 = arith.constant 3074 : index
    %c3075 = arith.constant 3075 : index
    %c3076 = arith.constant 3076 : index
    %c3077 = arith.constant 3077 : index
    %c3078 = arith.constant 3078 : index
    %c3079 = arith.constant 3079 : index
    %c3080 = arith.constant 3080 : index
    %c3081 = arith.constant 3081 : index
    %c3082 = arith.constant 3082 : index
    %c3083 = arith.constant 3083 : index
    %c3084 = arith.constant 3084 : index
    %c3085 = arith.constant 3085 : index
    %c3086 = arith.constant 3086 : index
    %c3087 = arith.constant 3087 : index
    %cst_2 = arith.constant dense<0> : vector<4xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = miopen.workgroup_id : index
    %1 = miopen.workitem_id : index
    %2 = arith.divui %0, %c6272 : index
    %3 = arith.remui %0, %c6272 : index
    %4 = arith.remui %3, %c16 : index
    %5 = arith.divui %3, %c16 : index
    %6 = arith.muli %4, %c64 : index
    %7 = arith.muli %5, %c64 : index
    %8 = arith.divui %1, %c16 : index
    %9 = arith.remui %1, %c16 : index
    %10 = arith.muli %8, %c4 : index
    %11 = arith.addi %6, %10 : index
    %12 = arith.divui %1, %c64 : index
    %13 = arith.remui %1, %c64 : index
    %14 = arith.muli %12, %c4 : index
    %15 = arith.addi %7, %13 : index
    %16 = miopen.alloc() : memref<32768xi8, 3>
    %17 = memref.load %arg0[%2, %11, %9, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %18 = arith.addi %11, %c1 : index
    %19 = memref.load %arg0[%2, %18, %9, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %20 = arith.addi %11, %c2 : index
    %21 = memref.load %arg0[%2, %20, %9, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %22 = arith.addi %11, %c3 : index
    %23 = memref.load %arg0[%2, %22, %9, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %24 = arith.addi %9, %c16 : index
    %25 = memref.load %arg0[%2, %11, %24, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %26 = arith.addi %9, %c16 : index
    %27 = arith.addi %11, %c1 : index
    %28 = memref.load %arg0[%2, %27, %26, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %29 = arith.addi %9, %c16 : index
    %30 = arith.addi %11, %c2 : index
    %31 = memref.load %arg0[%2, %30, %29, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %32 = arith.addi %9, %c16 : index
    %33 = arith.addi %11, %c3 : index
    %34 = memref.load %arg0[%2, %33, %32, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %35 = arith.addi %9, %c32 : index
    %36 = memref.load %arg0[%2, %11, %35, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %37 = arith.addi %9, %c32 : index
    %38 = arith.addi %11, %c1 : index
    %39 = memref.load %arg0[%2, %38, %37, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %40 = arith.addi %9, %c32 : index
    %41 = arith.addi %11, %c2 : index
    %42 = memref.load %arg0[%2, %41, %40, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %43 = arith.addi %9, %c32 : index
    %44 = arith.addi %11, %c3 : index
    %45 = memref.load %arg0[%2, %44, %43, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %46 = arith.addi %9, %c48 : index
    %47 = memref.load %arg0[%2, %11, %46, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %48 = arith.addi %9, %c48 : index
    %49 = arith.addi %11, %c1 : index
    %50 = memref.load %arg0[%2, %49, %48, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %51 = arith.addi %9, %c48 : index
    %52 = arith.addi %11, %c2 : index
    %53 = memref.load %arg0[%2, %52, %51, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %54 = arith.addi %9, %c48 : index
    %55 = arith.addi %11, %c3 : index
    %56 = memref.load %arg0[%2, %55, %54, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %57 = arith.addi %9, %c64 : index
    %58 = memref.load %arg0[%2, %11, %57, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %59 = arith.addi %9, %c64 : index
    %60 = arith.addi %11, %c1 : index
    %61 = memref.load %arg0[%2, %60, %59, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %62 = arith.addi %9, %c64 : index
    %63 = arith.addi %11, %c2 : index
    %64 = memref.load %arg0[%2, %63, %62, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %65 = arith.addi %9, %c64 : index
    %66 = arith.addi %11, %c3 : index
    %67 = memref.load %arg0[%2, %66, %65, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %68 = arith.addi %9, %c80 : index
    %69 = memref.load %arg0[%2, %11, %68, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %70 = arith.addi %9, %c80 : index
    %71 = arith.addi %11, %c1 : index
    %72 = memref.load %arg0[%2, %71, %70, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %73 = arith.addi %9, %c80 : index
    %74 = arith.addi %11, %c2 : index
    %75 = memref.load %arg0[%2, %74, %73, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %76 = arith.addi %9, %c80 : index
    %77 = arith.addi %11, %c3 : index
    %78 = memref.load %arg0[%2, %77, %76, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %79 = arith.addi %9, %c96 : index
    %80 = memref.load %arg0[%2, %11, %79, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %81 = arith.addi %9, %c96 : index
    %82 = arith.addi %11, %c1 : index
    %83 = memref.load %arg0[%2, %82, %81, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %84 = arith.addi %9, %c96 : index
    %85 = arith.addi %11, %c2 : index
    %86 = memref.load %arg0[%2, %85, %84, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %87 = arith.addi %9, %c96 : index
    %88 = arith.addi %11, %c3 : index
    %89 = memref.load %arg0[%2, %88, %87, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %90 = arith.addi %9, %c112 : index
    %91 = memref.load %arg0[%2, %11, %90, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %92 = arith.addi %9, %c112 : index
    %93 = arith.addi %11, %c1 : index
    %94 = memref.load %arg0[%2, %93, %92, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %95 = arith.addi %9, %c112 : index
    %96 = arith.addi %11, %c2 : index
    %97 = memref.load %arg0[%2, %96, %95, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %98 = arith.addi %9, %c112 : index
    %99 = arith.addi %11, %c3 : index
    %100 = memref.load %arg0[%2, %99, %98, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %101 = arith.addi %9, %c128 : index
    %102 = memref.load %arg0[%2, %11, %101, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %103 = arith.addi %9, %c128 : index
    %104 = arith.addi %11, %c1 : index
    %105 = memref.load %arg0[%2, %104, %103, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %106 = arith.addi %9, %c128 : index
    %107 = arith.addi %11, %c2 : index
    %108 = memref.load %arg0[%2, %107, %106, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %109 = arith.addi %9, %c128 : index
    %110 = arith.addi %11, %c3 : index
    %111 = memref.load %arg0[%2, %110, %109, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %112 = arith.addi %9, %c144 : index
    %113 = memref.load %arg0[%2, %11, %112, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %114 = arith.addi %9, %c144 : index
    %115 = arith.addi %11, %c1 : index
    %116 = memref.load %arg0[%2, %115, %114, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %117 = arith.addi %9, %c144 : index
    %118 = arith.addi %11, %c2 : index
    %119 = memref.load %arg0[%2, %118, %117, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %120 = arith.addi %9, %c144 : index
    %121 = arith.addi %11, %c3 : index
    %122 = memref.load %arg0[%2, %121, %120, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %123 = arith.addi %9, %c160 : index
    %124 = memref.load %arg0[%2, %11, %123, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %125 = arith.addi %9, %c160 : index
    %126 = arith.addi %11, %c1 : index
    %127 = memref.load %arg0[%2, %126, %125, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %128 = arith.addi %9, %c160 : index
    %129 = arith.addi %11, %c2 : index
    %130 = memref.load %arg0[%2, %129, %128, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %131 = arith.addi %9, %c160 : index
    %132 = arith.addi %11, %c3 : index
    %133 = memref.load %arg0[%2, %132, %131, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %134 = arith.addi %9, %c176 : index
    %135 = memref.load %arg0[%2, %11, %134, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %136 = arith.addi %9, %c176 : index
    %137 = arith.addi %11, %c1 : index
    %138 = memref.load %arg0[%2, %137, %136, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %139 = arith.addi %9, %c176 : index
    %140 = arith.addi %11, %c2 : index
    %141 = memref.load %arg0[%2, %140, %139, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %142 = arith.addi %9, %c176 : index
    %143 = arith.addi %11, %c3 : index
    %144 = memref.load %arg0[%2, %143, %142, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %145 = arith.addi %9, %c192 : index
    %146 = memref.load %arg0[%2, %11, %145, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %147 = arith.addi %9, %c192 : index
    %148 = arith.addi %11, %c1 : index
    %149 = memref.load %arg0[%2, %148, %147, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %150 = arith.addi %9, %c192 : index
    %151 = arith.addi %11, %c2 : index
    %152 = memref.load %arg0[%2, %151, %150, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %153 = arith.addi %9, %c192 : index
    %154 = arith.addi %11, %c3 : index
    %155 = memref.load %arg0[%2, %154, %153, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %156 = arith.addi %9, %c208 : index
    %157 = memref.load %arg0[%2, %11, %156, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %158 = arith.addi %9, %c208 : index
    %159 = arith.addi %11, %c1 : index
    %160 = memref.load %arg0[%2, %159, %158, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %161 = arith.addi %9, %c208 : index
    %162 = arith.addi %11, %c2 : index
    %163 = memref.load %arg0[%2, %162, %161, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %164 = arith.addi %9, %c208 : index
    %165 = arith.addi %11, %c3 : index
    %166 = memref.load %arg0[%2, %165, %164, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %167 = arith.addi %9, %c224 : index
    %168 = memref.load %arg0[%2, %11, %167, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %169 = arith.addi %9, %c224 : index
    %170 = arith.addi %11, %c1 : index
    %171 = memref.load %arg0[%2, %170, %169, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %172 = arith.addi %9, %c224 : index
    %173 = arith.addi %11, %c2 : index
    %174 = memref.load %arg0[%2, %173, %172, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %175 = arith.addi %9, %c224 : index
    %176 = arith.addi %11, %c3 : index
    %177 = memref.load %arg0[%2, %176, %175, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %178 = arith.addi %9, %c240 : index
    %179 = memref.load %arg0[%2, %11, %178, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %180 = arith.addi %9, %c240 : index
    %181 = arith.addi %11, %c1 : index
    %182 = memref.load %arg0[%2, %181, %180, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %183 = arith.addi %9, %c240 : index
    %184 = arith.addi %11, %c2 : index
    %185 = memref.load %arg0[%2, %184, %183, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %186 = arith.addi %9, %c240 : index
    %187 = arith.addi %11, %c3 : index
    %188 = memref.load %arg0[%2, %187, %186, %c0, %c0] : memref<1x1024x1024x1x1xi8>
    %189 = arith.muli %10, %c16 : index
    %190 = arith.addi %189, %9 : index
    memref.store %17, %16[%190] : memref<32768xi8, 3>
    %191 = arith.addi %190, %c16 : index
    memref.store %19, %16[%191] : memref<32768xi8, 3>
    %192 = arith.addi %190, %c32 : index
    memref.store %21, %16[%192] : memref<32768xi8, 3>
    %193 = arith.addi %190, %c48 : index
    memref.store %23, %16[%193] : memref<32768xi8, 3>
    %194 = arith.addi %190, %c1024 : index
    memref.store %25, %16[%194] : memref<32768xi8, 3>
    %195 = arith.addi %190, %c1040 : index
    memref.store %28, %16[%195] : memref<32768xi8, 3>
    %196 = arith.addi %190, %c1056 : index
    memref.store %31, %16[%196] : memref<32768xi8, 3>
    %197 = arith.addi %190, %c1072 : index
    memref.store %34, %16[%197] : memref<32768xi8, 3>
    %198 = arith.addi %190, %c2048 : index
    memref.store %36, %16[%198] : memref<32768xi8, 3>
    %199 = arith.addi %190, %c2064 : index
    memref.store %39, %16[%199] : memref<32768xi8, 3>
    %200 = arith.addi %190, %c2080 : index
    memref.store %42, %16[%200] : memref<32768xi8, 3>
    %201 = arith.addi %190, %c2096 : index
    memref.store %45, %16[%201] : memref<32768xi8, 3>
    %202 = arith.addi %190, %c3072 : index
    memref.store %47, %16[%202] : memref<32768xi8, 3>
    %203 = arith.addi %190, %c3088 : index
    memref.store %50, %16[%203] : memref<32768xi8, 3>
    %204 = arith.addi %190, %c3104 : index
    memref.store %53, %16[%204] : memref<32768xi8, 3>
    %205 = arith.addi %190, %c3120 : index
    memref.store %56, %16[%205] : memref<32768xi8, 3>
    %206 = arith.addi %190, %c4096 : index
    memref.store %58, %16[%206] : memref<32768xi8, 3>
    %207 = arith.addi %190, %c4112 : index
    memref.store %61, %16[%207] : memref<32768xi8, 3>
    %208 = arith.addi %190, %c4128 : index
    memref.store %64, %16[%208] : memref<32768xi8, 3>
    %209 = arith.addi %190, %c4144 : index
    memref.store %67, %16[%209] : memref<32768xi8, 3>
    %210 = arith.addi %190, %c5120 : index
    memref.store %69, %16[%210] : memref<32768xi8, 3>
    %211 = arith.addi %190, %c5136 : index
    memref.store %72, %16[%211] : memref<32768xi8, 3>
    %212 = arith.addi %190, %c5152 : index
    memref.store %75, %16[%212] : memref<32768xi8, 3>
    %213 = arith.addi %190, %c5168 : index
    memref.store %78, %16[%213] : memref<32768xi8, 3>
    %214 = arith.addi %190, %c6144 : index
    memref.store %80, %16[%214] : memref<32768xi8, 3>
    %215 = arith.addi %190, %c6160 : index
    memref.store %83, %16[%215] : memref<32768xi8, 3>
    %216 = arith.addi %190, %c6176 : index
    memref.store %86, %16[%216] : memref<32768xi8, 3>
    %217 = arith.addi %190, %c6192 : index
    memref.store %89, %16[%217] : memref<32768xi8, 3>
    %218 = arith.addi %190, %c7168 : index
    memref.store %91, %16[%218] : memref<32768xi8, 3>
    %219 = arith.addi %190, %c7184 : index
    memref.store %94, %16[%219] : memref<32768xi8, 3>
    %220 = arith.addi %190, %c7200 : index
    memref.store %97, %16[%220] : memref<32768xi8, 3>
    %221 = arith.addi %190, %c7216 : index
    memref.store %100, %16[%221] : memref<32768xi8, 3>
    %222 = arith.addi %190, %c8192 : index
    memref.store %102, %16[%222] : memref<32768xi8, 3>
    %223 = arith.addi %190, %c8208 : index
    memref.store %105, %16[%223] : memref<32768xi8, 3>
    %224 = arith.addi %190, %c8224 : index
    memref.store %108, %16[%224] : memref<32768xi8, 3>
    %225 = arith.addi %190, %c8240 : index
    memref.store %111, %16[%225] : memref<32768xi8, 3>
    %226 = arith.addi %190, %c9216 : index
    memref.store %113, %16[%226] : memref<32768xi8, 3>
    %227 = arith.addi %190, %c9232 : index
    memref.store %116, %16[%227] : memref<32768xi8, 3>
    %228 = arith.addi %190, %c9248 : index
    memref.store %119, %16[%228] : memref<32768xi8, 3>
    %229 = arith.addi %190, %c9264 : index
    memref.store %122, %16[%229] : memref<32768xi8, 3>
    %230 = arith.addi %190, %c10240 : index
    memref.store %124, %16[%230] : memref<32768xi8, 3>
    %231 = arith.addi %190, %c10256 : index
    memref.store %127, %16[%231] : memref<32768xi8, 3>
    %232 = arith.addi %190, %c10272 : index
    memref.store %130, %16[%232] : memref<32768xi8, 3>
    %233 = arith.addi %190, %c10288 : index
    memref.store %133, %16[%233] : memref<32768xi8, 3>
    %234 = arith.addi %190, %c11264 : index
    memref.store %135, %16[%234] : memref<32768xi8, 3>
    %235 = arith.addi %190, %c11280 : index
    memref.store %138, %16[%235] : memref<32768xi8, 3>
    %236 = arith.addi %190, %c11296 : index
    memref.store %141, %16[%236] : memref<32768xi8, 3>
    %237 = arith.addi %190, %c11312 : index
    memref.store %144, %16[%237] : memref<32768xi8, 3>
    %238 = arith.addi %190, %c12288 : index
    memref.store %146, %16[%238] : memref<32768xi8, 3>
    %239 = arith.addi %190, %c12304 : index
    memref.store %149, %16[%239] : memref<32768xi8, 3>
    %240 = arith.addi %190, %c12320 : index
    memref.store %152, %16[%240] : memref<32768xi8, 3>
    %241 = arith.addi %190, %c12336 : index
    memref.store %155, %16[%241] : memref<32768xi8, 3>
    %242 = arith.addi %190, %c13312 : index
    memref.store %157, %16[%242] : memref<32768xi8, 3>
    %243 = arith.addi %190, %c13328 : index
    memref.store %160, %16[%243] : memref<32768xi8, 3>
    %244 = arith.addi %190, %c13344 : index
    memref.store %163, %16[%244] : memref<32768xi8, 3>
    %245 = arith.addi %190, %c13360 : index
    memref.store %166, %16[%245] : memref<32768xi8, 3>
    %246 = arith.addi %190, %c14336 : index
    memref.store %168, %16[%246] : memref<32768xi8, 3>
    %247 = arith.addi %190, %c14352 : index
    memref.store %171, %16[%247] : memref<32768xi8, 3>
    %248 = arith.addi %190, %c14368 : index
    memref.store %174, %16[%248] : memref<32768xi8, 3>
    %249 = arith.addi %190, %c14384 : index
    memref.store %177, %16[%249] : memref<32768xi8, 3>
    %250 = arith.addi %190, %c15360 : index
    memref.store %179, %16[%250] : memref<32768xi8, 3>
    %251 = arith.addi %190, %c15376 : index
    memref.store %182, %16[%251] : memref<32768xi8, 3>
    %252 = arith.addi %190, %c15392 : index
    memref.store %185, %16[%252] : memref<32768xi8, 3>
    %253 = arith.addi %190, %c15408 : index
    memref.store %188, %16[%253] : memref<32768xi8, 3>
    %254 = arith.muli %14, %c16 : index
    %255 = arith.cmpi slt, %15, %c0 : index
    %256 = arith.subi %c-1, %15 : index
    %257 = arith.select %255, %256, %15 : index
    %258 = arith.divsi %257, %c196 : index
    %259 = arith.subi %c-1, %258 : index
    %260 = arith.select %255, %259, %258 : index
    %261 = arith.remsi %15, %c196 : index
    %262 = arith.cmpi slt, %261, %c0 : index
    %263 = arith.addi %261, %c196 : index
    %264 = arith.select %262, %263, %261 : index
    %265 = arith.cmpi slt, %264, %c0 : index
    %266 = arith.subi %c-1, %264 : index
    %267 = arith.select %265, %266, %264 : index
    %268 = arith.divsi %267, %c14 : index
    %269 = arith.subi %c-1, %268 : index
    %270 = arith.select %265, %269, %268 : index
    %271 = arith.remsi %15, %c14 : index
    %272 = arith.cmpi slt, %271, %c0 : index
    %273 = arith.addi %271, %c14 : index
    %274 = arith.select %272, %273, %271 : index
    %275 = memref.load %arg1[%260, %2, %254, %270, %274] : memref<128x1x1024x14x14xi8>
    %276 = arith.addi %254, %c1 : index
    %277 = memref.load %arg1[%260, %2, %276, %270, %274] : memref<128x1x1024x14x14xi8>
    %278 = arith.addi %254, %c2 : index
    %279 = memref.load %arg1[%260, %2, %278, %270, %274] : memref<128x1x1024x14x14xi8>
    %280 = arith.addi %254, %c3 : index
    %281 = memref.load %arg1[%260, %2, %280, %270, %274] : memref<128x1x1024x14x14xi8>
    %282 = arith.addi %254, %c4 : index
    %283 = memref.load %arg1[%260, %2, %282, %270, %274] : memref<128x1x1024x14x14xi8>
    %284 = arith.addi %254, %c5 : index
    %285 = memref.load %arg1[%260, %2, %284, %270, %274] : memref<128x1x1024x14x14xi8>
    %286 = arith.addi %254, %c6 : index
    %287 = memref.load %arg1[%260, %2, %286, %270, %274] : memref<128x1x1024x14x14xi8>
    %288 = arith.addi %254, %c7 : index
    %289 = memref.load %arg1[%260, %2, %288, %270, %274] : memref<128x1x1024x14x14xi8>
    %290 = arith.addi %254, %c8 : index
    %291 = memref.load %arg1[%260, %2, %290, %270, %274] : memref<128x1x1024x14x14xi8>
    %292 = arith.addi %254, %c9 : index
    %293 = memref.load %arg1[%260, %2, %292, %270, %274] : memref<128x1x1024x14x14xi8>
    %294 = arith.addi %254, %c10 : index
    %295 = memref.load %arg1[%260, %2, %294, %270, %274] : memref<128x1x1024x14x14xi8>
    %296 = arith.addi %254, %c11 : index
    %297 = memref.load %arg1[%260, %2, %296, %270, %274] : memref<128x1x1024x14x14xi8>
    %298 = arith.addi %254, %c12 : index
    %299 = memref.load %arg1[%260, %2, %298, %270, %274] : memref<128x1x1024x14x14xi8>
    %300 = arith.addi %254, %c13 : index
    %301 = memref.load %arg1[%260, %2, %300, %270, %274] : memref<128x1x1024x14x14xi8>
    %302 = arith.addi %254, %c14 : index
    %303 = memref.load %arg1[%260, %2, %302, %270, %274] : memref<128x1x1024x14x14xi8>
    %304 = arith.addi %254, %c15 : index
    %305 = memref.load %arg1[%260, %2, %304, %270, %274] : memref<128x1x1024x14x14xi8>
    %306 = arith.addi %254, %c16 : index
    %307 = memref.load %arg1[%260, %2, %306, %270, %274] : memref<128x1x1024x14x14xi8>
    %308 = arith.addi %254, %c17 : index
    %309 = memref.load %arg1[%260, %2, %308, %270, %274] : memref<128x1x1024x14x14xi8>
    %310 = arith.addi %254, %c18 : index
    %311 = memref.load %arg1[%260, %2, %310, %270, %274] : memref<128x1x1024x14x14xi8>
    %312 = arith.addi %254, %c19 : index
    %313 = memref.load %arg1[%260, %2, %312, %270, %274] : memref<128x1x1024x14x14xi8>
    %314 = arith.addi %254, %c20 : index
    %315 = memref.load %arg1[%260, %2, %314, %270, %274] : memref<128x1x1024x14x14xi8>
    %316 = arith.addi %254, %c21 : index
    %317 = memref.load %arg1[%260, %2, %316, %270, %274] : memref<128x1x1024x14x14xi8>
    %318 = arith.addi %254, %c22 : index
    %319 = memref.load %arg1[%260, %2, %318, %270, %274] : memref<128x1x1024x14x14xi8>
    %320 = arith.addi %254, %c23 : index
    %321 = memref.load %arg1[%260, %2, %320, %270, %274] : memref<128x1x1024x14x14xi8>
    %322 = arith.addi %254, %c24 : index
    %323 = memref.load %arg1[%260, %2, %322, %270, %274] : memref<128x1x1024x14x14xi8>
    %324 = arith.addi %254, %c25 : index
    %325 = memref.load %arg1[%260, %2, %324, %270, %274] : memref<128x1x1024x14x14xi8>
    %326 = arith.addi %254, %c26 : index
    %327 = memref.load %arg1[%260, %2, %326, %270, %274] : memref<128x1x1024x14x14xi8>
    %328 = arith.addi %254, %c27 : index
    %329 = memref.load %arg1[%260, %2, %328, %270, %274] : memref<128x1x1024x14x14xi8>
    %330 = arith.addi %254, %c28 : index
    %331 = memref.load %arg1[%260, %2, %330, %270, %274] : memref<128x1x1024x14x14xi8>
    %332 = arith.addi %254, %c29 : index
    %333 = memref.load %arg1[%260, %2, %332, %270, %274] : memref<128x1x1024x14x14xi8>
    %334 = arith.addi %254, %c30 : index
    %335 = memref.load %arg1[%260, %2, %334, %270, %274] : memref<128x1x1024x14x14xi8>
    %336 = arith.addi %254, %c31 : index
    %337 = memref.load %arg1[%260, %2, %336, %270, %274] : memref<128x1x1024x14x14xi8>
    %338 = arith.addi %254, %c32 : index
    %339 = memref.load %arg1[%260, %2, %338, %270, %274] : memref<128x1x1024x14x14xi8>
    %340 = arith.addi %254, %c33 : index
    %341 = memref.load %arg1[%260, %2, %340, %270, %274] : memref<128x1x1024x14x14xi8>
    %342 = arith.addi %254, %c34 : index
    %343 = memref.load %arg1[%260, %2, %342, %270, %274] : memref<128x1x1024x14x14xi8>
    %344 = arith.addi %254, %c35 : index
    %345 = memref.load %arg1[%260, %2, %344, %270, %274] : memref<128x1x1024x14x14xi8>
    %346 = arith.addi %254, %c36 : index
    %347 = memref.load %arg1[%260, %2, %346, %270, %274] : memref<128x1x1024x14x14xi8>
    %348 = arith.addi %254, %c37 : index
    %349 = memref.load %arg1[%260, %2, %348, %270, %274] : memref<128x1x1024x14x14xi8>
    %350 = arith.addi %254, %c38 : index
    %351 = memref.load %arg1[%260, %2, %350, %270, %274] : memref<128x1x1024x14x14xi8>
    %352 = arith.addi %254, %c39 : index
    %353 = memref.load %arg1[%260, %2, %352, %270, %274] : memref<128x1x1024x14x14xi8>
    %354 = arith.addi %254, %c40 : index
    %355 = memref.load %arg1[%260, %2, %354, %270, %274] : memref<128x1x1024x14x14xi8>
    %356 = arith.addi %254, %c41 : index
    %357 = memref.load %arg1[%260, %2, %356, %270, %274] : memref<128x1x1024x14x14xi8>
    %358 = arith.addi %254, %c42 : index
    %359 = memref.load %arg1[%260, %2, %358, %270, %274] : memref<128x1x1024x14x14xi8>
    %360 = arith.addi %254, %c43 : index
    %361 = memref.load %arg1[%260, %2, %360, %270, %274] : memref<128x1x1024x14x14xi8>
    %362 = arith.addi %254, %c44 : index
    %363 = memref.load %arg1[%260, %2, %362, %270, %274] : memref<128x1x1024x14x14xi8>
    %364 = arith.addi %254, %c45 : index
    %365 = memref.load %arg1[%260, %2, %364, %270, %274] : memref<128x1x1024x14x14xi8>
    %366 = arith.addi %254, %c46 : index
    %367 = memref.load %arg1[%260, %2, %366, %270, %274] : memref<128x1x1024x14x14xi8>
    %368 = arith.addi %254, %c47 : index
    %369 = memref.load %arg1[%260, %2, %368, %270, %274] : memref<128x1x1024x14x14xi8>
    %370 = arith.addi %254, %c48 : index
    %371 = memref.load %arg1[%260, %2, %370, %270, %274] : memref<128x1x1024x14x14xi8>
    %372 = arith.addi %254, %c49 : index
    %373 = memref.load %arg1[%260, %2, %372, %270, %274] : memref<128x1x1024x14x14xi8>
    %374 = arith.addi %254, %c50 : index
    %375 = memref.load %arg1[%260, %2, %374, %270, %274] : memref<128x1x1024x14x14xi8>
    %376 = arith.addi %254, %c51 : index
    %377 = memref.load %arg1[%260, %2, %376, %270, %274] : memref<128x1x1024x14x14xi8>
    %378 = arith.addi %254, %c52 : index
    %379 = memref.load %arg1[%260, %2, %378, %270, %274] : memref<128x1x1024x14x14xi8>
    %380 = arith.addi %254, %c53 : index
    %381 = memref.load %arg1[%260, %2, %380, %270, %274] : memref<128x1x1024x14x14xi8>
    %382 = arith.addi %254, %c54 : index
    %383 = memref.load %arg1[%260, %2, %382, %270, %274] : memref<128x1x1024x14x14xi8>
    %384 = arith.addi %254, %c55 : index
    %385 = memref.load %arg1[%260, %2, %384, %270, %274] : memref<128x1x1024x14x14xi8>
    %386 = arith.addi %254, %c56 : index
    %387 = memref.load %arg1[%260, %2, %386, %270, %274] : memref<128x1x1024x14x14xi8>
    %388 = arith.addi %254, %c57 : index
    %389 = memref.load %arg1[%260, %2, %388, %270, %274] : memref<128x1x1024x14x14xi8>
    %390 = arith.addi %254, %c58 : index
    %391 = memref.load %arg1[%260, %2, %390, %270, %274] : memref<128x1x1024x14x14xi8>
    %392 = arith.addi %254, %c59 : index
    %393 = memref.load %arg1[%260, %2, %392, %270, %274] : memref<128x1x1024x14x14xi8>
    %394 = arith.addi %254, %c60 : index
    %395 = memref.load %arg1[%260, %2, %394, %270, %274] : memref<128x1x1024x14x14xi8>
    %396 = arith.addi %254, %c61 : index
    %397 = memref.load %arg1[%260, %2, %396, %270, %274] : memref<128x1x1024x14x14xi8>
    %398 = arith.addi %254, %c62 : index
    %399 = memref.load %arg1[%260, %2, %398, %270, %274] : memref<128x1x1024x14x14xi8>
    %400 = arith.addi %254, %c63 : index
    %401 = memref.load %arg1[%260, %2, %400, %270, %274] : memref<128x1x1024x14x14xi8>
    %402 = arith.muli %14, %c1024 : index
    %403 = arith.muli %13, %c16 : index
    %404 = arith.addi %402, %403 : index
    %405 = arith.addi %404, %c16384 : index
    memref.store %275, %16[%405] : memref<32768xi8, 3>
    %406 = arith.addi %405, %c1 : index
    memref.store %277, %16[%406] : memref<32768xi8, 3>
    %407 = arith.addi %405, %c2 : index
    memref.store %279, %16[%407] : memref<32768xi8, 3>
    %408 = arith.addi %405, %c3 : index
    memref.store %281, %16[%408] : memref<32768xi8, 3>
    %409 = arith.addi %405, %c4 : index
    memref.store %283, %16[%409] : memref<32768xi8, 3>
    %410 = arith.addi %405, %c5 : index
    memref.store %285, %16[%410] : memref<32768xi8, 3>
    %411 = arith.addi %405, %c6 : index
    memref.store %287, %16[%411] : memref<32768xi8, 3>
    %412 = arith.addi %405, %c7 : index
    memref.store %289, %16[%412] : memref<32768xi8, 3>
    %413 = arith.addi %405, %c8 : index
    memref.store %291, %16[%413] : memref<32768xi8, 3>
    %414 = arith.addi %405, %c9 : index
    memref.store %293, %16[%414] : memref<32768xi8, 3>
    %415 = arith.addi %405, %c10 : index
    memref.store %295, %16[%415] : memref<32768xi8, 3>
    %416 = arith.addi %405, %c11 : index
    memref.store %297, %16[%416] : memref<32768xi8, 3>
    %417 = arith.addi %405, %c12 : index
    memref.store %299, %16[%417] : memref<32768xi8, 3>
    %418 = arith.addi %405, %c13 : index
    memref.store %301, %16[%418] : memref<32768xi8, 3>
    %419 = arith.addi %405, %c14 : index
    memref.store %303, %16[%419] : memref<32768xi8, 3>
    %420 = arith.addi %405, %c15 : index
    memref.store %305, %16[%420] : memref<32768xi8, 3>
    %421 = arith.addi %405, %c1024 : index
    memref.store %307, %16[%421] : memref<32768xi8, 3>
    %422 = arith.addi %405, %c1025 : index
    memref.store %309, %16[%422] : memref<32768xi8, 3>
    %423 = arith.addi %405, %c1026 : index
    memref.store %311, %16[%423] : memref<32768xi8, 3>
    %424 = arith.addi %405, %c1027 : index
    memref.store %313, %16[%424] : memref<32768xi8, 3>
    %425 = arith.addi %405, %c1028 : index
    memref.store %315, %16[%425] : memref<32768xi8, 3>
    %426 = arith.addi %405, %c1029 : index
    memref.store %317, %16[%426] : memref<32768xi8, 3>
    %427 = arith.addi %405, %c1030 : index
    memref.store %319, %16[%427] : memref<32768xi8, 3>
    %428 = arith.addi %405, %c1031 : index
    memref.store %321, %16[%428] : memref<32768xi8, 3>
    %429 = arith.addi %405, %c1032 : index
    memref.store %323, %16[%429] : memref<32768xi8, 3>
    %430 = arith.addi %405, %c1033 : index
    memref.store %325, %16[%430] : memref<32768xi8, 3>
    %431 = arith.addi %405, %c1034 : index
    memref.store %327, %16[%431] : memref<32768xi8, 3>
    %432 = arith.addi %405, %c1035 : index
    memref.store %329, %16[%432] : memref<32768xi8, 3>
    %433 = arith.addi %405, %c1036 : index
    memref.store %331, %16[%433] : memref<32768xi8, 3>
    %434 = arith.addi %405, %c1037 : index
    memref.store %333, %16[%434] : memref<32768xi8, 3>
    %435 = arith.addi %405, %c1038 : index
    memref.store %335, %16[%435] : memref<32768xi8, 3>
    %436 = arith.addi %405, %c1039 : index
    memref.store %337, %16[%436] : memref<32768xi8, 3>
    %437 = arith.addi %405, %c2048 : index
    memref.store %339, %16[%437] : memref<32768xi8, 3>
    %438 = arith.addi %405, %c2049 : index
    memref.store %341, %16[%438] : memref<32768xi8, 3>
    %439 = arith.addi %405, %c2050 : index
    memref.store %343, %16[%439] : memref<32768xi8, 3>
    %440 = arith.addi %405, %c2051 : index
    memref.store %345, %16[%440] : memref<32768xi8, 3>
    %441 = arith.addi %405, %c2052 : index
    memref.store %347, %16[%441] : memref<32768xi8, 3>
    %442 = arith.addi %405, %c2053 : index
    memref.store %349, %16[%442] : memref<32768xi8, 3>
    %443 = arith.addi %405, %c2054 : index
    memref.store %351, %16[%443] : memref<32768xi8, 3>
    %444 = arith.addi %405, %c2055 : index
    memref.store %353, %16[%444] : memref<32768xi8, 3>
    %445 = arith.addi %405, %c2056 : index
    memref.store %355, %16[%445] : memref<32768xi8, 3>
    %446 = arith.addi %405, %c2057 : index
    memref.store %357, %16[%446] : memref<32768xi8, 3>
    %447 = arith.addi %405, %c2058 : index
    memref.store %359, %16[%447] : memref<32768xi8, 3>
    %448 = arith.addi %405, %c2059 : index
    memref.store %361, %16[%448] : memref<32768xi8, 3>
    %449 = arith.addi %405, %c2060 : index
    memref.store %363, %16[%449] : memref<32768xi8, 3>
    %450 = arith.addi %405, %c2061 : index
    memref.store %365, %16[%450] : memref<32768xi8, 3>
    %451 = arith.addi %405, %c2062 : index
    memref.store %367, %16[%451] : memref<32768xi8, 3>
    %452 = arith.addi %405, %c2063 : index
    memref.store %369, %16[%452] : memref<32768xi8, 3>
    %453 = arith.addi %405, %c3072 : index
    memref.store %371, %16[%453] : memref<32768xi8, 3>
    %454 = arith.addi %405, %c3073 : index
    memref.store %373, %16[%454] : memref<32768xi8, 3>
    %455 = arith.addi %405, %c3074 : index
    memref.store %375, %16[%455] : memref<32768xi8, 3>
    %456 = arith.addi %405, %c3075 : index
    memref.store %377, %16[%456] : memref<32768xi8, 3>
    %457 = arith.addi %405, %c3076 : index
    memref.store %379, %16[%457] : memref<32768xi8, 3>
    %458 = arith.addi %405, %c3077 : index
    memref.store %381, %16[%458] : memref<32768xi8, 3>
    %459 = arith.addi %405, %c3078 : index
    memref.store %383, %16[%459] : memref<32768xi8, 3>
    %460 = arith.addi %405, %c3079 : index
    memref.store %385, %16[%460] : memref<32768xi8, 3>
    %461 = arith.addi %405, %c3080 : index
    memref.store %387, %16[%461] : memref<32768xi8, 3>
    %462 = arith.addi %405, %c3081 : index
    memref.store %389, %16[%462] : memref<32768xi8, 3>
    %463 = arith.addi %405, %c3082 : index
    memref.store %391, %16[%463] : memref<32768xi8, 3>
    %464 = arith.addi %405, %c3083 : index
    memref.store %393, %16[%464] : memref<32768xi8, 3>
    %465 = arith.addi %405, %c3084 : index
    memref.store %395, %16[%465] : memref<32768xi8, 3>
    %466 = arith.addi %405, %c3085 : index
    memref.store %397, %16[%466] : memref<32768xi8, 3>
    %467 = arith.addi %405, %c3086 : index
    memref.store %399, %16[%467] : memref<32768xi8, 3>
    %468 = arith.addi %405, %c3087 : index
    memref.store %401, %16[%468] : memref<32768xi8, 3>
    %469 = arith.divui %1, %c64 : index
    %470 = arith.divui %469, %c2 : index
    %471 = arith.remui %469, %c2 : index
    %472 = arith.muli %470, %c32 : index
    %473 = arith.muli %471, %c32 : index
    %474 = miopen.alloc() : memref<8xvector<16xi8>, 5>
    %475 = miopen.alloc() : memref<8xvector<16xi8>, 5>
    %476:3 = affine.for %arg3 = 0 to 3 iter_args(%arg4 = %c0, %arg5 = %14, %arg6 = %cst_1) -> (index, index, vector<16xi32>) {
      %833 = arith.addi %arg4, %c16 : index
      %834 = arith.muli %833, %c16 : index
      %835 = arith.addi %834, %9 : index
      %836 = memref.load %arg0[%2, %11, %835, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %837 = arith.addi %11, %c1 : index
      %838 = memref.load %arg0[%2, %837, %835, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %839 = arith.addi %11, %c2 : index
      %840 = memref.load %arg0[%2, %839, %835, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %841 = arith.addi %11, %c3 : index
      %842 = memref.load %arg0[%2, %841, %835, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %843 = arith.addi %835, %c16 : index
      %844 = memref.load %arg0[%2, %11, %843, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %845 = arith.addi %835, %c16 : index
      %846 = arith.addi %11, %c1 : index
      %847 = memref.load %arg0[%2, %846, %845, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %848 = arith.addi %835, %c16 : index
      %849 = arith.addi %11, %c2 : index
      %850 = memref.load %arg0[%2, %849, %848, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %851 = arith.addi %835, %c16 : index
      %852 = arith.addi %11, %c3 : index
      %853 = memref.load %arg0[%2, %852, %851, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %854 = arith.addi %835, %c32 : index
      %855 = memref.load %arg0[%2, %11, %854, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %856 = arith.addi %835, %c32 : index
      %857 = arith.addi %11, %c1 : index
      %858 = memref.load %arg0[%2, %857, %856, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %859 = arith.addi %835, %c32 : index
      %860 = arith.addi %11, %c2 : index
      %861 = memref.load %arg0[%2, %860, %859, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %862 = arith.addi %835, %c32 : index
      %863 = arith.addi %11, %c3 : index
      %864 = memref.load %arg0[%2, %863, %862, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %865 = arith.addi %835, %c48 : index
      %866 = memref.load %arg0[%2, %11, %865, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %867 = arith.addi %835, %c48 : index
      %868 = arith.addi %11, %c1 : index
      %869 = memref.load %arg0[%2, %868, %867, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %870 = arith.addi %835, %c48 : index
      %871 = arith.addi %11, %c2 : index
      %872 = memref.load %arg0[%2, %871, %870, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %873 = arith.addi %835, %c48 : index
      %874 = arith.addi %11, %c3 : index
      %875 = memref.load %arg0[%2, %874, %873, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %876 = arith.addi %835, %c64 : index
      %877 = memref.load %arg0[%2, %11, %876, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %878 = arith.addi %835, %c64 : index
      %879 = arith.addi %11, %c1 : index
      %880 = memref.load %arg0[%2, %879, %878, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %881 = arith.addi %835, %c64 : index
      %882 = arith.addi %11, %c2 : index
      %883 = memref.load %arg0[%2, %882, %881, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %884 = arith.addi %835, %c64 : index
      %885 = arith.addi %11, %c3 : index
      %886 = memref.load %arg0[%2, %885, %884, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %887 = arith.addi %835, %c80 : index
      %888 = memref.load %arg0[%2, %11, %887, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %889 = arith.addi %835, %c80 : index
      %890 = arith.addi %11, %c1 : index
      %891 = memref.load %arg0[%2, %890, %889, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %892 = arith.addi %835, %c80 : index
      %893 = arith.addi %11, %c2 : index
      %894 = memref.load %arg0[%2, %893, %892, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %895 = arith.addi %835, %c80 : index
      %896 = arith.addi %11, %c3 : index
      %897 = memref.load %arg0[%2, %896, %895, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %898 = arith.addi %835, %c96 : index
      %899 = memref.load %arg0[%2, %11, %898, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %900 = arith.addi %835, %c96 : index
      %901 = arith.addi %11, %c1 : index
      %902 = memref.load %arg0[%2, %901, %900, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %903 = arith.addi %835, %c96 : index
      %904 = arith.addi %11, %c2 : index
      %905 = memref.load %arg0[%2, %904, %903, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %906 = arith.addi %835, %c96 : index
      %907 = arith.addi %11, %c3 : index
      %908 = memref.load %arg0[%2, %907, %906, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %909 = arith.addi %835, %c112 : index
      %910 = memref.load %arg0[%2, %11, %909, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %911 = arith.addi %835, %c112 : index
      %912 = arith.addi %11, %c1 : index
      %913 = memref.load %arg0[%2, %912, %911, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %914 = arith.addi %835, %c112 : index
      %915 = arith.addi %11, %c2 : index
      %916 = memref.load %arg0[%2, %915, %914, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %917 = arith.addi %835, %c112 : index
      %918 = arith.addi %11, %c3 : index
      %919 = memref.load %arg0[%2, %918, %917, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %920 = arith.addi %835, %c128 : index
      %921 = memref.load %arg0[%2, %11, %920, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %922 = arith.addi %835, %c128 : index
      %923 = arith.addi %11, %c1 : index
      %924 = memref.load %arg0[%2, %923, %922, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %925 = arith.addi %835, %c128 : index
      %926 = arith.addi %11, %c2 : index
      %927 = memref.load %arg0[%2, %926, %925, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %928 = arith.addi %835, %c128 : index
      %929 = arith.addi %11, %c3 : index
      %930 = memref.load %arg0[%2, %929, %928, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %931 = arith.addi %835, %c144 : index
      %932 = memref.load %arg0[%2, %11, %931, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %933 = arith.addi %835, %c144 : index
      %934 = arith.addi %11, %c1 : index
      %935 = memref.load %arg0[%2, %934, %933, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %936 = arith.addi %835, %c144 : index
      %937 = arith.addi %11, %c2 : index
      %938 = memref.load %arg0[%2, %937, %936, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %939 = arith.addi %835, %c144 : index
      %940 = arith.addi %11, %c3 : index
      %941 = memref.load %arg0[%2, %940, %939, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %942 = arith.addi %835, %c160 : index
      %943 = memref.load %arg0[%2, %11, %942, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %944 = arith.addi %835, %c160 : index
      %945 = arith.addi %11, %c1 : index
      %946 = memref.load %arg0[%2, %945, %944, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %947 = arith.addi %835, %c160 : index
      %948 = arith.addi %11, %c2 : index
      %949 = memref.load %arg0[%2, %948, %947, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %950 = arith.addi %835, %c160 : index
      %951 = arith.addi %11, %c3 : index
      %952 = memref.load %arg0[%2, %951, %950, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %953 = arith.addi %835, %c176 : index
      %954 = memref.load %arg0[%2, %11, %953, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %955 = arith.addi %835, %c176 : index
      %956 = arith.addi %11, %c1 : index
      %957 = memref.load %arg0[%2, %956, %955, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %958 = arith.addi %835, %c176 : index
      %959 = arith.addi %11, %c2 : index
      %960 = memref.load %arg0[%2, %959, %958, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %961 = arith.addi %835, %c176 : index
      %962 = arith.addi %11, %c3 : index
      %963 = memref.load %arg0[%2, %962, %961, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %964 = arith.addi %835, %c192 : index
      %965 = memref.load %arg0[%2, %11, %964, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %966 = arith.addi %835, %c192 : index
      %967 = arith.addi %11, %c1 : index
      %968 = memref.load %arg0[%2, %967, %966, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %969 = arith.addi %835, %c192 : index
      %970 = arith.addi %11, %c2 : index
      %971 = memref.load %arg0[%2, %970, %969, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %972 = arith.addi %835, %c192 : index
      %973 = arith.addi %11, %c3 : index
      %974 = memref.load %arg0[%2, %973, %972, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %975 = arith.addi %835, %c208 : index
      %976 = memref.load %arg0[%2, %11, %975, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %977 = arith.addi %835, %c208 : index
      %978 = arith.addi %11, %c1 : index
      %979 = memref.load %arg0[%2, %978, %977, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %980 = arith.addi %835, %c208 : index
      %981 = arith.addi %11, %c2 : index
      %982 = memref.load %arg0[%2, %981, %980, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %983 = arith.addi %835, %c208 : index
      %984 = arith.addi %11, %c3 : index
      %985 = memref.load %arg0[%2, %984, %983, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %986 = arith.addi %835, %c224 : index
      %987 = memref.load %arg0[%2, %11, %986, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %988 = arith.addi %835, %c224 : index
      %989 = arith.addi %11, %c1 : index
      %990 = memref.load %arg0[%2, %989, %988, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %991 = arith.addi %835, %c224 : index
      %992 = arith.addi %11, %c2 : index
      %993 = memref.load %arg0[%2, %992, %991, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %994 = arith.addi %835, %c224 : index
      %995 = arith.addi %11, %c3 : index
      %996 = memref.load %arg0[%2, %995, %994, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %997 = arith.addi %835, %c240 : index
      %998 = memref.load %arg0[%2, %11, %997, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %999 = arith.addi %835, %c240 : index
      %1000 = arith.addi %11, %c1 : index
      %1001 = memref.load %arg0[%2, %1000, %999, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %1002 = arith.addi %835, %c240 : index
      %1003 = arith.addi %11, %c2 : index
      %1004 = memref.load %arg0[%2, %1003, %1002, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %1005 = arith.addi %835, %c240 : index
      %1006 = arith.addi %11, %c3 : index
      %1007 = memref.load %arg0[%2, %1006, %1005, %c0, %c0] : memref<1x1024x1024x1x1xi8>
      %1008 = arith.addi %arg5, %c16 : index
      %1009 = arith.muli %1008, %c16 : index
      %1010 = arith.cmpi slt, %15, %c0 : index
      %1011 = arith.subi %c-1, %15 : index
      %1012 = arith.select %1010, %1011, %15 : index
      %1013 = arith.divsi %1012, %c196 : index
      %1014 = arith.subi %c-1, %1013 : index
      %1015 = arith.select %1010, %1014, %1013 : index
      %1016 = arith.remsi %15, %c196 : index
      %1017 = arith.cmpi slt, %1016, %c0 : index
      %1018 = arith.addi %1016, %c196 : index
      %1019 = arith.select %1017, %1018, %1016 : index
      %1020 = arith.cmpi slt, %1019, %c0 : index
      %1021 = arith.subi %c-1, %1019 : index
      %1022 = arith.select %1020, %1021, %1019 : index
      %1023 = arith.divsi %1022, %c14 : index
      %1024 = arith.subi %c-1, %1023 : index
      %1025 = arith.select %1020, %1024, %1023 : index
      %1026 = arith.remsi %15, %c14 : index
      %1027 = arith.cmpi slt, %1026, %c0 : index
      %1028 = arith.addi %1026, %c14 : index
      %1029 = arith.select %1027, %1028, %1026 : index
      %1030 = memref.load %arg1[%1015, %2, %1009, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1031 = arith.addi %1009, %c1 : index
      %1032 = memref.load %arg1[%1015, %2, %1031, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1033 = arith.addi %1009, %c2 : index
      %1034 = memref.load %arg1[%1015, %2, %1033, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1035 = arith.addi %1009, %c3 : index
      %1036 = memref.load %arg1[%1015, %2, %1035, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1037 = arith.addi %1009, %c4 : index
      %1038 = memref.load %arg1[%1015, %2, %1037, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1039 = arith.addi %1009, %c5 : index
      %1040 = memref.load %arg1[%1015, %2, %1039, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1041 = arith.addi %1009, %c6 : index
      %1042 = memref.load %arg1[%1015, %2, %1041, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1043 = arith.addi %1009, %c7 : index
      %1044 = memref.load %arg1[%1015, %2, %1043, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1045 = arith.addi %1009, %c8 : index
      %1046 = memref.load %arg1[%1015, %2, %1045, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1047 = arith.addi %1009, %c9 : index
      %1048 = memref.load %arg1[%1015, %2, %1047, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1049 = arith.addi %1009, %c10 : index
      %1050 = memref.load %arg1[%1015, %2, %1049, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1051 = arith.addi %1009, %c11 : index
      %1052 = memref.load %arg1[%1015, %2, %1051, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1053 = arith.addi %1009, %c12 : index
      %1054 = memref.load %arg1[%1015, %2, %1053, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1055 = arith.addi %1009, %c13 : index
      %1056 = memref.load %arg1[%1015, %2, %1055, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1057 = arith.addi %1009, %c14 : index
      %1058 = memref.load %arg1[%1015, %2, %1057, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1059 = arith.addi %1009, %c15 : index
      %1060 = memref.load %arg1[%1015, %2, %1059, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1061 = arith.addi %1009, %c16 : index
      %1062 = memref.load %arg1[%1015, %2, %1061, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1063 = arith.addi %1009, %c17 : index
      %1064 = memref.load %arg1[%1015, %2, %1063, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1065 = arith.addi %1009, %c18 : index
      %1066 = memref.load %arg1[%1015, %2, %1065, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1067 = arith.addi %1009, %c19 : index
      %1068 = memref.load %arg1[%1015, %2, %1067, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1069 = arith.addi %1009, %c20 : index
      %1070 = memref.load %arg1[%1015, %2, %1069, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1071 = arith.addi %1009, %c21 : index
      %1072 = memref.load %arg1[%1015, %2, %1071, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1073 = arith.addi %1009, %c22 : index
      %1074 = memref.load %arg1[%1015, %2, %1073, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1075 = arith.addi %1009, %c23 : index
      %1076 = memref.load %arg1[%1015, %2, %1075, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1077 = arith.addi %1009, %c24 : index
      %1078 = memref.load %arg1[%1015, %2, %1077, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1079 = arith.addi %1009, %c25 : index
      %1080 = memref.load %arg1[%1015, %2, %1079, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1081 = arith.addi %1009, %c26 : index
      %1082 = memref.load %arg1[%1015, %2, %1081, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1083 = arith.addi %1009, %c27 : index
      %1084 = memref.load %arg1[%1015, %2, %1083, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1085 = arith.addi %1009, %c28 : index
      %1086 = memref.load %arg1[%1015, %2, %1085, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1087 = arith.addi %1009, %c29 : index
      %1088 = memref.load %arg1[%1015, %2, %1087, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1089 = arith.addi %1009, %c30 : index
      %1090 = memref.load %arg1[%1015, %2, %1089, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1091 = arith.addi %1009, %c31 : index
      %1092 = memref.load %arg1[%1015, %2, %1091, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1093 = arith.addi %1009, %c32 : index
      %1094 = memref.load %arg1[%1015, %2, %1093, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1095 = arith.addi %1009, %c33 : index
      %1096 = memref.load %arg1[%1015, %2, %1095, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1097 = arith.addi %1009, %c34 : index
      %1098 = memref.load %arg1[%1015, %2, %1097, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1099 = arith.addi %1009, %c35 : index
      %1100 = memref.load %arg1[%1015, %2, %1099, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1101 = arith.addi %1009, %c36 : index
      %1102 = memref.load %arg1[%1015, %2, %1101, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1103 = arith.addi %1009, %c37 : index
      %1104 = memref.load %arg1[%1015, %2, %1103, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1105 = arith.addi %1009, %c38 : index
      %1106 = memref.load %arg1[%1015, %2, %1105, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1107 = arith.addi %1009, %c39 : index
      %1108 = memref.load %arg1[%1015, %2, %1107, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1109 = arith.addi %1009, %c40 : index
      %1110 = memref.load %arg1[%1015, %2, %1109, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1111 = arith.addi %1009, %c41 : index
      %1112 = memref.load %arg1[%1015, %2, %1111, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1113 = arith.addi %1009, %c42 : index
      %1114 = memref.load %arg1[%1015, %2, %1113, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1115 = arith.addi %1009, %c43 : index
      %1116 = memref.load %arg1[%1015, %2, %1115, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1117 = arith.addi %1009, %c44 : index
      %1118 = memref.load %arg1[%1015, %2, %1117, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1119 = arith.addi %1009, %c45 : index
      %1120 = memref.load %arg1[%1015, %2, %1119, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1121 = arith.addi %1009, %c46 : index
      %1122 = memref.load %arg1[%1015, %2, %1121, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1123 = arith.addi %1009, %c47 : index
      %1124 = memref.load %arg1[%1015, %2, %1123, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1125 = arith.addi %1009, %c48 : index
      %1126 = memref.load %arg1[%1015, %2, %1125, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1127 = arith.addi %1009, %c49 : index
      %1128 = memref.load %arg1[%1015, %2, %1127, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1129 = arith.addi %1009, %c50 : index
      %1130 = memref.load %arg1[%1015, %2, %1129, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1131 = arith.addi %1009, %c51 : index
      %1132 = memref.load %arg1[%1015, %2, %1131, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1133 = arith.addi %1009, %c52 : index
      %1134 = memref.load %arg1[%1015, %2, %1133, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1135 = arith.addi %1009, %c53 : index
      %1136 = memref.load %arg1[%1015, %2, %1135, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1137 = arith.addi %1009, %c54 : index
      %1138 = memref.load %arg1[%1015, %2, %1137, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1139 = arith.addi %1009, %c55 : index
      %1140 = memref.load %arg1[%1015, %2, %1139, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1141 = arith.addi %1009, %c56 : index
      %1142 = memref.load %arg1[%1015, %2, %1141, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1143 = arith.addi %1009, %c57 : index
      %1144 = memref.load %arg1[%1015, %2, %1143, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1145 = arith.addi %1009, %c58 : index
      %1146 = memref.load %arg1[%1015, %2, %1145, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1147 = arith.addi %1009, %c59 : index
      %1148 = memref.load %arg1[%1015, %2, %1147, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1149 = arith.addi %1009, %c60 : index
      %1150 = memref.load %arg1[%1015, %2, %1149, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1151 = arith.addi %1009, %c61 : index
      %1152 = memref.load %arg1[%1015, %2, %1151, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1153 = arith.addi %1009, %c62 : index
      %1154 = memref.load %arg1[%1015, %2, %1153, %1025, %1029] : memref<128x1x1024x14x14xi8>
      %1155 = arith.addi %1009, %c63 : index
      %1156 = memref.load %arg1[%1015, %2, %1155, %1025, %1029] : memref<128x1x1024x14x14xi8>
      miopen.lds_barrier
      %1157 = miopen.workitem_id : index
      %1158 = arith.remui %1157, %c64 : index
      %1159 = arith.divui %1158, %c32 : index
      %1160 = arith.remui %1158, %c32 : index
      affine.for %arg7 = 0 to 8 {
        %1294 = arith.muli %arg7, %c2 : index
        %1295 = arith.addi %1294, %1159 : index
        %1296 = arith.muli %1295, %c64 : index
        %1297 = arith.addi %1296, %1160 : index
        %1298 = arith.addi %472, %1297 : index
        %1299 = memref.load %16[%1298] : memref<32768xi8, 3>
        %1300 = vector.insertelement %1299, %cst[%c0 : index] : vector<4xi8>
        %1301 = arith.addi %1298, %c1 : index
        %1302 = memref.load %16[%1301] : memref<32768xi8, 3>
        %1303 = vector.insertelement %1302, %1300[%c1 : index] : vector<4xi8>
        %1304 = arith.addi %1298, %c2 : index
        %1305 = memref.load %16[%1304] : memref<32768xi8, 3>
        %1306 = vector.insertelement %1305, %1303[%c2 : index] : vector<4xi8>
        %1307 = arith.addi %1298, %c3 : index
        %1308 = memref.load %16[%1307] : memref<32768xi8, 3>
        %1309 = vector.insertelement %1308, %1306[%c3 : index] : vector<4xi8>
        memref.store %1309, %474[%arg7] : memref<8xvector<16xi8>, 5>
        %1310 = arith.muli %arg7, %c2 : index
        %1311 = arith.addi %1310, %1159 : index
        %1312 = arith.muli %1311, %c64 : index
        %1313 = arith.addi %1312, %1160 : index
        %1314 = arith.addi %473, %1313 : index
        %1315 = arith.addi %1314, %c16384 : index
        %1316 = memref.load %16[%1315] : memref<32768xi8, 3>
        %1317 = vector.insertelement %1316, %cst[%c0 : index] : vector<4xi8>
        %1318 = arith.addi %1315, %c1 : index
        %1319 = memref.load %16[%1318] : memref<32768xi8, 3>
        %1320 = vector.insertelement %1319, %1317[%c1 : index] : vector<4xi8>
        %1321 = arith.addi %1315, %c2 : index
        %1322 = memref.load %16[%1321] : memref<32768xi8, 3>
        %1323 = vector.insertelement %1322, %1320[%c2 : index] : vector<4xi8>
        %1324 = arith.addi %1315, %c3 : index
        %1325 = memref.load %16[%1324] : memref<32768xi8, 3>
        %1326 = vector.insertelement %1325, %1323[%c3 : index] : vector<4xi8>
        memref.store %1326, %475[%arg7] : memref<8xvector<16xi8>, 5>
      }
      %1161 = affine.for %arg7 = 0 to 8 step 4 iter_args(%arg8 = %arg6) -> (vector<16xi32>) {
        %1294 = arith.muli %arg7, %c16 : index
        %1295 = affine.for %arg9 = 0 to 16 step 4 iter_args(%arg10 = %arg8) -> (vector<16xi32>) {
          %1296 = arith.addi %1294, %arg9 : index
          %1297 = vector.transfer_read %474[%1296], %cst_0 : memref<8xvector<16xi8>, 5>, vector<4xi8>
          %1298 = vector.transfer_read %475[%1296], %cst_0 : memref<8xvector<16xi8>, 5>, vector<4xi8>
          %1299 = miopen.mfma_v2(%1297, %1298, %arg10) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_i32_32x32x8i8"} : vector<4xi8>, vector<16xi32>
          affine.yield %1299 : vector<16xi32>
        }
        affine.yield %1295 : vector<16xi32>
      }
      miopen.lds_barrier
      %1162 = arith.muli %10, %c16 : index
      %1163 = arith.addi %1162, %9 : index
      memref.store %836, %16[%1163] : memref<32768xi8, 3>
      %1164 = arith.addi %1163, %c16 : index
      memref.store %838, %16[%1164] : memref<32768xi8, 3>
      %1165 = arith.addi %1163, %c32 : index
      memref.store %840, %16[%1165] : memref<32768xi8, 3>
      %1166 = arith.addi %1163, %c48 : index
      memref.store %842, %16[%1166] : memref<32768xi8, 3>
      %1167 = arith.addi %1163, %c1024 : index
      memref.store %844, %16[%1167] : memref<32768xi8, 3>
      %1168 = arith.addi %1163, %c1040 : index
      memref.store %847, %16[%1168] : memref<32768xi8, 3>
      %1169 = arith.addi %1163, %c1056 : index
      memref.store %850, %16[%1169] : memref<32768xi8, 3>
      %1170 = arith.addi %1163, %c1072 : index
      memref.store %853, %16[%1170] : memref<32768xi8, 3>
      %1171 = arith.addi %1163, %c2048 : index
      memref.store %855, %16[%1171] : memref<32768xi8, 3>
      %1172 = arith.addi %1163, %c2064 : index
      memref.store %858, %16[%1172] : memref<32768xi8, 3>
      %1173 = arith.addi %1163, %c2080 : index
      memref.store %861, %16[%1173] : memref<32768xi8, 3>
      %1174 = arith.addi %1163, %c2096 : index
      memref.store %864, %16[%1174] : memref<32768xi8, 3>
      %1175 = arith.addi %1163, %c3072 : index
      memref.store %866, %16[%1175] : memref<32768xi8, 3>
      %1176 = arith.addi %1163, %c3088 : index
      memref.store %869, %16[%1176] : memref<32768xi8, 3>
      %1177 = arith.addi %1163, %c3104 : index
      memref.store %872, %16[%1177] : memref<32768xi8, 3>
      %1178 = arith.addi %1163, %c3120 : index
      memref.store %875, %16[%1178] : memref<32768xi8, 3>
      %1179 = arith.addi %1163, %c4096 : index
      memref.store %877, %16[%1179] : memref<32768xi8, 3>
      %1180 = arith.addi %1163, %c4112 : index
      memref.store %880, %16[%1180] : memref<32768xi8, 3>
      %1181 = arith.addi %1163, %c4128 : index
      memref.store %883, %16[%1181] : memref<32768xi8, 3>
      %1182 = arith.addi %1163, %c4144 : index
      memref.store %886, %16[%1182] : memref<32768xi8, 3>
      %1183 = arith.addi %1163, %c5120 : index
      memref.store %888, %16[%1183] : memref<32768xi8, 3>
      %1184 = arith.addi %1163, %c5136 : index
      memref.store %891, %16[%1184] : memref<32768xi8, 3>
      %1185 = arith.addi %1163, %c5152 : index
      memref.store %894, %16[%1185] : memref<32768xi8, 3>
      %1186 = arith.addi %1163, %c5168 : index
      memref.store %897, %16[%1186] : memref<32768xi8, 3>
      %1187 = arith.addi %1163, %c6144 : index
      memref.store %899, %16[%1187] : memref<32768xi8, 3>
      %1188 = arith.addi %1163, %c6160 : index
      memref.store %902, %16[%1188] : memref<32768xi8, 3>
      %1189 = arith.addi %1163, %c6176 : index
      memref.store %905, %16[%1189] : memref<32768xi8, 3>
      %1190 = arith.addi %1163, %c6192 : index
      memref.store %908, %16[%1190] : memref<32768xi8, 3>
      %1191 = arith.addi %1163, %c7168 : index
      memref.store %910, %16[%1191] : memref<32768xi8, 3>
      %1192 = arith.addi %1163, %c7184 : index
      memref.store %913, %16[%1192] : memref<32768xi8, 3>
      %1193 = arith.addi %1163, %c7200 : index
      memref.store %916, %16[%1193] : memref<32768xi8, 3>
      %1194 = arith.addi %1163, %c7216 : index
      memref.store %919, %16[%1194] : memref<32768xi8, 3>
      %1195 = arith.addi %1163, %c8192 : index
      memref.store %921, %16[%1195] : memref<32768xi8, 3>
      %1196 = arith.addi %1163, %c8208 : index
      memref.store %924, %16[%1196] : memref<32768xi8, 3>
      %1197 = arith.addi %1163, %c8224 : index
      memref.store %927, %16[%1197] : memref<32768xi8, 3>
      %1198 = arith.addi %1163, %c8240 : index
      memref.store %930, %16[%1198] : memref<32768xi8, 3>
      %1199 = arith.addi %1163, %c9216 : index
      memref.store %932, %16[%1199] : memref<32768xi8, 3>
      %1200 = arith.addi %1163, %c9232 : index
      memref.store %935, %16[%1200] : memref<32768xi8, 3>
      %1201 = arith.addi %1163, %c9248 : index
      memref.store %938, %16[%1201] : memref<32768xi8, 3>
      %1202 = arith.addi %1163, %c9264 : index
      memref.store %941, %16[%1202] : memref<32768xi8, 3>
      %1203 = arith.addi %1163, %c10240 : index
      memref.store %943, %16[%1203] : memref<32768xi8, 3>
      %1204 = arith.addi %1163, %c10256 : index
      memref.store %946, %16[%1204] : memref<32768xi8, 3>
      %1205 = arith.addi %1163, %c10272 : index
      memref.store %949, %16[%1205] : memref<32768xi8, 3>
      %1206 = arith.addi %1163, %c10288 : index
      memref.store %952, %16[%1206] : memref<32768xi8, 3>
      %1207 = arith.addi %1163, %c11264 : index
      memref.store %954, %16[%1207] : memref<32768xi8, 3>
      %1208 = arith.addi %1163, %c11280 : index
      memref.store %957, %16[%1208] : memref<32768xi8, 3>
      %1209 = arith.addi %1163, %c11296 : index
      memref.store %960, %16[%1209] : memref<32768xi8, 3>
      %1210 = arith.addi %1163, %c11312 : index
      memref.store %963, %16[%1210] : memref<32768xi8, 3>
      %1211 = arith.addi %1163, %c12288 : index
      memref.store %965, %16[%1211] : memref<32768xi8, 3>
      %1212 = arith.addi %1163, %c12304 : index
      memref.store %968, %16[%1212] : memref<32768xi8, 3>
      %1213 = arith.addi %1163, %c12320 : index
      memref.store %971, %16[%1213] : memref<32768xi8, 3>
      %1214 = arith.addi %1163, %c12336 : index
      memref.store %974, %16[%1214] : memref<32768xi8, 3>
      %1215 = arith.addi %1163, %c13312 : index
      memref.store %976, %16[%1215] : memref<32768xi8, 3>
      %1216 = arith.addi %1163, %c13328 : index
      memref.store %979, %16[%1216] : memref<32768xi8, 3>
      %1217 = arith.addi %1163, %c13344 : index
      memref.store %982, %16[%1217] : memref<32768xi8, 3>
      %1218 = arith.addi %1163, %c13360 : index
      memref.store %985, %16[%1218] : memref<32768xi8, 3>
      %1219 = arith.addi %1163, %c14336 : index
      memref.store %987, %16[%1219] : memref<32768xi8, 3>
      %1220 = arith.addi %1163, %c14352 : index
      memref.store %990, %16[%1220] : memref<32768xi8, 3>
      %1221 = arith.addi %1163, %c14368 : index
      memref.store %993, %16[%1221] : memref<32768xi8, 3>
      %1222 = arith.addi %1163, %c14384 : index
      memref.store %996, %16[%1222] : memref<32768xi8, 3>
      %1223 = arith.addi %1163, %c15360 : index
      memref.store %998, %16[%1223] : memref<32768xi8, 3>
      %1224 = arith.addi %1163, %c15376 : index
      memref.store %1001, %16[%1224] : memref<32768xi8, 3>
      %1225 = arith.addi %1163, %c15392 : index
      memref.store %1004, %16[%1225] : memref<32768xi8, 3>
      %1226 = arith.addi %1163, %c15408 : index
      memref.store %1007, %16[%1226] : memref<32768xi8, 3>
      %1227 = arith.muli %14, %c1024 : index
      %1228 = arith.muli %13, %c16 : index
      %1229 = arith.addi %1227, %1228 : index
      %1230 = arith.addi %1229, %c16384 : index
      memref.store %1030, %16[%1230] : memref<32768xi8, 3>
      %1231 = arith.addi %1230, %c1 : index
      memref.store %1032, %16[%1231] : memref<32768xi8, 3>
      %1232 = arith.addi %1230, %c2 : index
      memref.store %1034, %16[%1232] : memref<32768xi8, 3>
      %1233 = arith.addi %1230, %c3 : index
      memref.store %1036, %16[%1233] : memref<32768xi8, 3>
      %1234 = arith.addi %1230, %c4 : index
      memref.store %1038, %16[%1234] : memref<32768xi8, 3>
      %1235 = arith.addi %1230, %c5 : index
      memref.store %1040, %16[%1235] : memref<32768xi8, 3>
      %1236 = arith.addi %1230, %c6 : index
      memref.store %1042, %16[%1236] : memref<32768xi8, 3>
      %1237 = arith.addi %1230, %c7 : index
      memref.store %1044, %16[%1237] : memref<32768xi8, 3>
      %1238 = arith.addi %1230, %c8 : index
      memref.store %1046, %16[%1238] : memref<32768xi8, 3>
      %1239 = arith.addi %1230, %c9 : index
      memref.store %1048, %16[%1239] : memref<32768xi8, 3>
      %1240 = arith.addi %1230, %c10 : index
      memref.store %1050, %16[%1240] : memref<32768xi8, 3>
      %1241 = arith.addi %1230, %c11 : index
      memref.store %1052, %16[%1241] : memref<32768xi8, 3>
      %1242 = arith.addi %1230, %c12 : index
      memref.store %1054, %16[%1242] : memref<32768xi8, 3>
      %1243 = arith.addi %1230, %c13 : index
      memref.store %1056, %16[%1243] : memref<32768xi8, 3>
      %1244 = arith.addi %1230, %c14 : index
      memref.store %1058, %16[%1244] : memref<32768xi8, 3>
      %1245 = arith.addi %1230, %c15 : index
      memref.store %1060, %16[%1245] : memref<32768xi8, 3>
      %1246 = arith.addi %1230, %c1024 : index
      memref.store %1062, %16[%1246] : memref<32768xi8, 3>
      %1247 = arith.addi %1230, %c1025 : index
      memref.store %1064, %16[%1247] : memref<32768xi8, 3>
      %1248 = arith.addi %1230, %c1026 : index
      memref.store %1066, %16[%1248] : memref<32768xi8, 3>
      %1249 = arith.addi %1230, %c1027 : index
      memref.store %1068, %16[%1249] : memref<32768xi8, 3>
      %1250 = arith.addi %1230, %c1028 : index
      memref.store %1070, %16[%1250] : memref<32768xi8, 3>
      %1251 = arith.addi %1230, %c1029 : index
      memref.store %1072, %16[%1251] : memref<32768xi8, 3>
      %1252 = arith.addi %1230, %c1030 : index
      memref.store %1074, %16[%1252] : memref<32768xi8, 3>
      %1253 = arith.addi %1230, %c1031 : index
      memref.store %1076, %16[%1253] : memref<32768xi8, 3>
      %1254 = arith.addi %1230, %c1032 : index
      memref.store %1078, %16[%1254] : memref<32768xi8, 3>
      %1255 = arith.addi %1230, %c1033 : index
      memref.store %1080, %16[%1255] : memref<32768xi8, 3>
      %1256 = arith.addi %1230, %c1034 : index
      memref.store %1082, %16[%1256] : memref<32768xi8, 3>
      %1257 = arith.addi %1230, %c1035 : index
      memref.store %1084, %16[%1257] : memref<32768xi8, 3>
      %1258 = arith.addi %1230, %c1036 : index
      memref.store %1086, %16[%1258] : memref<32768xi8, 3>
      %1259 = arith.addi %1230, %c1037 : index
      memref.store %1088, %16[%1259] : memref<32768xi8, 3>
      %1260 = arith.addi %1230, %c1038 : index
      memref.store %1090, %16[%1260] : memref<32768xi8, 3>
      %1261 = arith.addi %1230, %c1039 : index
      memref.store %1092, %16[%1261] : memref<32768xi8, 3>
      %1262 = arith.addi %1230, %c2048 : index
      memref.store %1094, %16[%1262] : memref<32768xi8, 3>
      %1263 = arith.addi %1230, %c2049 : index
      memref.store %1096, %16[%1263] : memref<32768xi8, 3>
      %1264 = arith.addi %1230, %c2050 : index
      memref.store %1098, %16[%1264] : memref<32768xi8, 3>
      %1265 = arith.addi %1230, %c2051 : index
      memref.store %1100, %16[%1265] : memref<32768xi8, 3>
      %1266 = arith.addi %1230, %c2052 : index
      memref.store %1102, %16[%1266] : memref<32768xi8, 3>
      %1267 = arith.addi %1230, %c2053 : index
      memref.store %1104, %16[%1267] : memref<32768xi8, 3>
      %1268 = arith.addi %1230, %c2054 : index
      memref.store %1106, %16[%1268] : memref<32768xi8, 3>
      %1269 = arith.addi %1230, %c2055 : index
      memref.store %1108, %16[%1269] : memref<32768xi8, 3>
      %1270 = arith.addi %1230, %c2056 : index
      memref.store %1110, %16[%1270] : memref<32768xi8, 3>
      %1271 = arith.addi %1230, %c2057 : index
      memref.store %1112, %16[%1271] : memref<32768xi8, 3>
      %1272 = arith.addi %1230, %c2058 : index
      memref.store %1114, %16[%1272] : memref<32768xi8, 3>
      %1273 = arith.addi %1230, %c2059 : index
      memref.store %1116, %16[%1273] : memref<32768xi8, 3>
      %1274 = arith.addi %1230, %c2060 : index
      memref.store %1118, %16[%1274] : memref<32768xi8, 3>
      %1275 = arith.addi %1230, %c2061 : index
      memref.store %1120, %16[%1275] : memref<32768xi8, 3>
      %1276 = arith.addi %1230, %c2062 : index
      memref.store %1122, %16[%1276] : memref<32768xi8, 3>
      %1277 = arith.addi %1230, %c2063 : index
      memref.store %1124, %16[%1277] : memref<32768xi8, 3>
      %1278 = arith.addi %1230, %c3072 : index
      memref.store %1126, %16[%1278] : memref<32768xi8, 3>
      %1279 = arith.addi %1230, %c3073 : index
      memref.store %1128, %16[%1279] : memref<32768xi8, 3>
      %1280 = arith.addi %1230, %c3074 : index
      memref.store %1130, %16[%1280] : memref<32768xi8, 3>
      %1281 = arith.addi %1230, %c3075 : index
      memref.store %1132, %16[%1281] : memref<32768xi8, 3>
      %1282 = arith.addi %1230, %c3076 : index
      memref.store %1134, %16[%1282] : memref<32768xi8, 3>
      %1283 = arith.addi %1230, %c3077 : index
      memref.store %1136, %16[%1283] : memref<32768xi8, 3>
      %1284 = arith.addi %1230, %c3078 : index
      memref.store %1138, %16[%1284] : memref<32768xi8, 3>
      %1285 = arith.addi %1230, %c3079 : index
      memref.store %1140, %16[%1285] : memref<32768xi8, 3>
      %1286 = arith.addi %1230, %c3080 : index
      memref.store %1142, %16[%1286] : memref<32768xi8, 3>
      %1287 = arith.addi %1230, %c3081 : index
      memref.store %1144, %16[%1287] : memref<32768xi8, 3>
      %1288 = arith.addi %1230, %c3082 : index
      memref.store %1146, %16[%1288] : memref<32768xi8, 3>
      %1289 = arith.addi %1230, %c3083 : index
      memref.store %1148, %16[%1289] : memref<32768xi8, 3>
      %1290 = arith.addi %1230, %c3084 : index
      memref.store %1150, %16[%1290] : memref<32768xi8, 3>
      %1291 = arith.addi %1230, %c3085 : index
      memref.store %1152, %16[%1291] : memref<32768xi8, 3>
      %1292 = arith.addi %1230, %c3086 : index
      memref.store %1154, %16[%1292] : memref<32768xi8, 3>
      %1293 = arith.addi %1230, %c3087 : index
      memref.store %1156, %16[%1293] : memref<32768xi8, 3>
      affine.yield %833, %1008, %1161 : index, index, vector<16xi32>
    }
    miopen.lds_barrier
    %477 = miopen.workitem_id : index
    %478 = arith.remui %477, %c64 : index
    %479 = arith.divui %478, %c32 : index
    %480 = arith.remui %478, %c32 : index
    affine.for %arg3 = 0 to 8 {
      %833 = arith.muli %arg3, %c2 : index
      %834 = arith.addi %833, %479 : index
      %835 = arith.muli %834, %c64 : index
      %836 = arith.addi %835, %480 : index
      %837 = arith.addi %472, %836 : index
      %838 = memref.load %16[%837] : memref<32768xi8, 3>
      %839 = vector.insertelement %838, %cst[%c0 : index] : vector<4xi8>
      %840 = arith.addi %837, %c1 : index
      %841 = memref.load %16[%840] : memref<32768xi8, 3>
      %842 = vector.insertelement %841, %839[%c1 : index] : vector<4xi8>
      %843 = arith.addi %837, %c2 : index
      %844 = memref.load %16[%843] : memref<32768xi8, 3>
      %845 = vector.insertelement %844, %842[%c2 : index] : vector<4xi8>
      %846 = arith.addi %837, %c3 : index
      %847 = memref.load %16[%846] : memref<32768xi8, 3>
      %848 = vector.insertelement %847, %845[%c3 : index] : vector<4xi8>
      memref.store %848, %474[%arg3] : memref<8xvector<16xi8>, 5>
      %849 = arith.muli %arg3, %c2 : index
      %850 = arith.addi %849, %479 : index
      %851 = arith.muli %850, %c64 : index
      %852 = arith.addi %851, %480 : index
      %853 = arith.addi %473, %852 : index
      %854 = arith.addi %853, %c16384 : index
      %855 = memref.load %16[%854] : memref<32768xi8, 3>
      %856 = vector.insertelement %855, %cst[%c0 : index] : vector<4xi8>
      %857 = arith.addi %854, %c1 : index
      %858 = memref.load %16[%857] : memref<32768xi8, 3>
      %859 = vector.insertelement %858, %856[%c1 : index] : vector<4xi8>
      %860 = arith.addi %854, %c2 : index
      %861 = memref.load %16[%860] : memref<32768xi8, 3>
      %862 = vector.insertelement %861, %859[%c2 : index] : vector<4xi8>
      %863 = arith.addi %854, %c3 : index
      %864 = memref.load %16[%863] : memref<32768xi8, 3>
      %865 = vector.insertelement %864, %862[%c3 : index] : vector<4xi8>
      memref.store %865, %475[%arg3] : memref<8xvector<16xi8>, 5>
    }
    %481 = affine.for %arg3 = 0 to 8 step 4 iter_args(%arg4 = %476#2) -> (vector<16xi32>) {
      %833 = arith.muli %arg3, %c16 : index
      %834 = affine.for %arg5 = 0 to 16 step 4 iter_args(%arg6 = %arg4) -> (vector<16xi32>) {
        %835 = arith.addi %833, %arg5 : index
        %836 = vector.transfer_read %474[%835], %cst_0 : memref<8xvector<16xi8>, 5>, vector<4xi8>
        %837 = vector.transfer_read %475[%835], %cst_0 : memref<8xvector<16xi8>, 5>, vector<4xi8>
        %838 = miopen.mfma_v2(%836, %837, %arg6) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_i32_32x32x8i8"} : vector<4xi8>, vector<16xi32>
        affine.yield %838 : vector<16xi32>
      }
      affine.yield %834 : vector<16xi32>
    }
    %482 = arith.remui %1, %c64 : index
    %483 = arith.divui %482, %c32 : index
    %484 = arith.remui %482, %c32 : index
    %485 = arith.andi %482, %c2 : index
    %486 = arith.cmpi ne, %485, %c0 : index
    %487 = vector.extractelement %481[%c0 : index] : vector<16xi32>
    %488 = vector.extractelement %481[%c1 : index] : vector<16xi32>
    %489 = vector.extractelement %481[%c2 : index] : vector<16xi32>
    %490 = vector.extractelement %481[%c3 : index] : vector<16xi32>
    %491 = vector.extractelement %481[%c4 : index] : vector<16xi32>
    %492 = vector.extractelement %481[%c5 : index] : vector<16xi32>
    %493 = vector.extractelement %481[%c6 : index] : vector<16xi32>
    %494 = vector.extractelement %481[%c7 : index] : vector<16xi32>
    %495 = vector.extractelement %481[%c8 : index] : vector<16xi32>
    %496 = vector.extractelement %481[%c9 : index] : vector<16xi32>
    %497 = vector.extractelement %481[%c10 : index] : vector<16xi32>
    %498 = vector.extractelement %481[%c11 : index] : vector<16xi32>
    %499 = vector.extractelement %481[%c12 : index] : vector<16xi32>
    %500 = vector.extractelement %481[%c13 : index] : vector<16xi32>
    %501 = vector.extractelement %481[%c14 : index] : vector<16xi32>
    %502 = vector.extractelement %481[%c15 : index] : vector<16xi32>
    %503 = arith.select %486, %487, %489 : i32
    %504 = vector.insertelement %503, %481[%c2 : index] : vector<16xi32>
    %505 = arith.select %486, %488, %490 : i32
    %506 = vector.insertelement %505, %504[%c3 : index] : vector<16xi32>
    %507 = arith.select %486, %489, %487 : i32
    %508 = vector.insertelement %507, %506[%c0 : index] : vector<16xi32>
    %509 = arith.select %486, %490, %488 : i32
    %510 = vector.insertelement %509, %508[%c1 : index] : vector<16xi32>
    %511 = arith.select %486, %491, %493 : i32
    %512 = vector.insertelement %511, %510[%c6 : index] : vector<16xi32>
    %513 = arith.select %486, %492, %494 : i32
    %514 = vector.insertelement %513, %512[%c7 : index] : vector<16xi32>
    %515 = arith.select %486, %493, %491 : i32
    %516 = vector.insertelement %515, %514[%c4 : index] : vector<16xi32>
    %517 = arith.select %486, %494, %492 : i32
    %518 = vector.insertelement %517, %516[%c5 : index] : vector<16xi32>
    %519 = arith.select %486, %495, %497 : i32
    %520 = vector.insertelement %519, %518[%c10 : index] : vector<16xi32>
    %521 = arith.select %486, %496, %498 : i32
    %522 = vector.insertelement %521, %520[%c11 : index] : vector<16xi32>
    %523 = arith.select %486, %497, %495 : i32
    %524 = vector.insertelement %523, %522[%c8 : index] : vector<16xi32>
    %525 = arith.select %486, %498, %496 : i32
    %526 = vector.insertelement %525, %524[%c9 : index] : vector<16xi32>
    %527 = arith.select %486, %499, %501 : i32
    %528 = vector.insertelement %527, %526[%c14 : index] : vector<16xi32>
    %529 = arith.select %486, %500, %502 : i32
    %530 = vector.insertelement %529, %528[%c15 : index] : vector<16xi32>
    %531 = arith.select %486, %501, %499 : i32
    %532 = vector.insertelement %531, %530[%c12 : index] : vector<16xi32>
    %533 = arith.select %486, %502, %500 : i32
    %534 = vector.insertelement %533, %532[%c13 : index] : vector<16xi32>
    %535 = arith.andi %482, %c1 : index
    %536 = arith.cmpi ne, %535, %c0 : index
    %537 = vector.extractelement %534[%c0 : index] : vector<16xi32>
    %538 = vector.extractelement %534[%c1 : index] : vector<16xi32>
    %539 = vector.extractelement %534[%c2 : index] : vector<16xi32>
    %540 = vector.extractelement %534[%c3 : index] : vector<16xi32>
    %541 = vector.extractelement %534[%c4 : index] : vector<16xi32>
    %542 = vector.extractelement %534[%c5 : index] : vector<16xi32>
    %543 = vector.extractelement %534[%c6 : index] : vector<16xi32>
    %544 = vector.extractelement %534[%c7 : index] : vector<16xi32>
    %545 = vector.extractelement %534[%c8 : index] : vector<16xi32>
    %546 = vector.extractelement %534[%c9 : index] : vector<16xi32>
    %547 = vector.extractelement %534[%c10 : index] : vector<16xi32>
    %548 = vector.extractelement %534[%c11 : index] : vector<16xi32>
    %549 = vector.extractelement %534[%c12 : index] : vector<16xi32>
    %550 = vector.extractelement %534[%c13 : index] : vector<16xi32>
    %551 = vector.extractelement %534[%c14 : index] : vector<16xi32>
    %552 = vector.extractelement %534[%c15 : index] : vector<16xi32>
    %553 = arith.select %536, %537, %538 : i32
    %554 = vector.insertelement %553, %534[%c1 : index] : vector<16xi32>
    %555 = arith.select %536, %538, %539 : i32
    %556 = vector.insertelement %555, %554[%c2 : index] : vector<16xi32>
    %557 = arith.select %536, %539, %540 : i32
    %558 = vector.insertelement %557, %556[%c3 : index] : vector<16xi32>
    %559 = arith.select %536, %540, %537 : i32
    %560 = vector.insertelement %559, %558[%c0 : index] : vector<16xi32>
    %561 = arith.select %536, %541, %542 : i32
    %562 = vector.insertelement %561, %560[%c5 : index] : vector<16xi32>
    %563 = arith.select %536, %542, %543 : i32
    %564 = vector.insertelement %563, %562[%c6 : index] : vector<16xi32>
    %565 = arith.select %536, %543, %544 : i32
    %566 = vector.insertelement %565, %564[%c7 : index] : vector<16xi32>
    %567 = arith.select %536, %544, %541 : i32
    %568 = vector.insertelement %567, %566[%c4 : index] : vector<16xi32>
    %569 = arith.select %536, %545, %546 : i32
    %570 = vector.insertelement %569, %568[%c9 : index] : vector<16xi32>
    %571 = arith.select %536, %546, %547 : i32
    %572 = vector.insertelement %571, %570[%c10 : index] : vector<16xi32>
    %573 = arith.select %536, %547, %548 : i32
    %574 = vector.insertelement %573, %572[%c11 : index] : vector<16xi32>
    %575 = arith.select %536, %548, %545 : i32
    %576 = vector.insertelement %575, %574[%c8 : index] : vector<16xi32>
    %577 = arith.select %536, %549, %550 : i32
    %578 = vector.insertelement %577, %576[%c13 : index] : vector<16xi32>
    %579 = arith.select %536, %550, %551 : i32
    %580 = vector.insertelement %579, %578[%c14 : index] : vector<16xi32>
    %581 = arith.select %536, %551, %552 : i32
    %582 = vector.insertelement %581, %580[%c15 : index] : vector<16xi32>
    %583 = arith.select %536, %552, %549 : i32
    %584 = vector.insertelement %583, %582[%c12 : index] : vector<16xi32>
    %585 = vector.extractelement %584[%c0 : index] : vector<16xi32>
    %586 = vector.extractelement %584[%c1 : index] : vector<16xi32>
    %587 = vector.extractelement %584[%c2 : index] : vector<16xi32>
    %588 = vector.extractelement %584[%c3 : index] : vector<16xi32>
    %589 = vector.extractelement %584[%c4 : index] : vector<16xi32>
    %590 = vector.extractelement %584[%c5 : index] : vector<16xi32>
    %591 = vector.extractelement %584[%c6 : index] : vector<16xi32>
    %592 = vector.extractelement %584[%c7 : index] : vector<16xi32>
    %593 = vector.extractelement %584[%c8 : index] : vector<16xi32>
    %594 = vector.extractelement %584[%c9 : index] : vector<16xi32>
    %595 = vector.extractelement %584[%c10 : index] : vector<16xi32>
    %596 = vector.extractelement %584[%c11 : index] : vector<16xi32>
    %597 = vector.extractelement %584[%c12 : index] : vector<16xi32>
    %598 = vector.extractelement %584[%c13 : index] : vector<16xi32>
    %599 = vector.extractelement %584[%c14 : index] : vector<16xi32>
    %600 = vector.extractelement %584[%c15 : index] : vector<16xi32>
    %601 = gpu.warp_swizzle {selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32]} %585 : i32
    %602 = gpu.warp_swizzle {selector = [1 : i32, 0 : i32, 3 : i32, 2 : i32]} %586 : i32
    %603 = gpu.warp_swizzle {selector = [2 : i32, 1 : i32, 0 : i32, 3 : i32]} %587 : i32
    %604 = gpu.warp_swizzle {selector = [3 : i32, 2 : i32, 1 : i32, 0 : i32]} %588 : i32
    %605 = gpu.warp_swizzle {selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32]} %589 : i32
    %606 = gpu.warp_swizzle {selector = [1 : i32, 0 : i32, 3 : i32, 2 : i32]} %590 : i32
    %607 = gpu.warp_swizzle {selector = [2 : i32, 1 : i32, 0 : i32, 3 : i32]} %591 : i32
    %608 = gpu.warp_swizzle {selector = [3 : i32, 2 : i32, 1 : i32, 0 : i32]} %592 : i32
    %609 = gpu.warp_swizzle {selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32]} %593 : i32
    %610 = gpu.warp_swizzle {selector = [1 : i32, 0 : i32, 3 : i32, 2 : i32]} %594 : i32
    %611 = gpu.warp_swizzle {selector = [2 : i32, 1 : i32, 0 : i32, 3 : i32]} %595 : i32
    %612 = gpu.warp_swizzle {selector = [3 : i32, 2 : i32, 1 : i32, 0 : i32]} %596 : i32
    %613 = gpu.warp_swizzle {selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32]} %597 : i32
    %614 = gpu.warp_swizzle {selector = [1 : i32, 0 : i32, 3 : i32, 2 : i32]} %598 : i32
    %615 = gpu.warp_swizzle {selector = [2 : i32, 1 : i32, 0 : i32, 3 : i32]} %599 : i32
    %616 = gpu.warp_swizzle {selector = [3 : i32, 2 : i32, 1 : i32, 0 : i32]} %600 : i32
    %617 = vector.insertelement %601, %584[%c0 : index] : vector<16xi32>
    %618 = vector.insertelement %602, %617[%c1 : index] : vector<16xi32>
    %619 = vector.insertelement %603, %618[%c2 : index] : vector<16xi32>
    %620 = vector.insertelement %604, %619[%c3 : index] : vector<16xi32>
    %621 = vector.insertelement %605, %620[%c4 : index] : vector<16xi32>
    %622 = vector.insertelement %606, %621[%c5 : index] : vector<16xi32>
    %623 = vector.insertelement %607, %622[%c6 : index] : vector<16xi32>
    %624 = vector.insertelement %608, %623[%c7 : index] : vector<16xi32>
    %625 = vector.insertelement %609, %624[%c8 : index] : vector<16xi32>
    %626 = vector.insertelement %610, %625[%c9 : index] : vector<16xi32>
    %627 = vector.insertelement %611, %626[%c10 : index] : vector<16xi32>
    %628 = vector.insertelement %612, %627[%c11 : index] : vector<16xi32>
    %629 = vector.insertelement %613, %628[%c12 : index] : vector<16xi32>
    %630 = vector.insertelement %614, %629[%c13 : index] : vector<16xi32>
    %631 = vector.insertelement %615, %630[%c14 : index] : vector<16xi32>
    %632 = vector.insertelement %616, %631[%c15 : index] : vector<16xi32>
    %633 = arith.andi %482, %c1 : index
    %634 = arith.cmpi ne, %633, %c0 : index
    %635 = vector.extractelement %632[%c0 : index] : vector<16xi32>
    %636 = vector.extractelement %632[%c1 : index] : vector<16xi32>
    %637 = vector.extractelement %632[%c2 : index] : vector<16xi32>
    %638 = vector.extractelement %632[%c3 : index] : vector<16xi32>
    %639 = vector.extractelement %632[%c4 : index] : vector<16xi32>
    %640 = vector.extractelement %632[%c5 : index] : vector<16xi32>
    %641 = vector.extractelement %632[%c6 : index] : vector<16xi32>
    %642 = vector.extractelement %632[%c7 : index] : vector<16xi32>
    %643 = vector.extractelement %632[%c8 : index] : vector<16xi32>
    %644 = vector.extractelement %632[%c9 : index] : vector<16xi32>
    %645 = vector.extractelement %632[%c10 : index] : vector<16xi32>
    %646 = vector.extractelement %632[%c11 : index] : vector<16xi32>
    %647 = vector.extractelement %632[%c12 : index] : vector<16xi32>
    %648 = vector.extractelement %632[%c13 : index] : vector<16xi32>
    %649 = vector.extractelement %632[%c14 : index] : vector<16xi32>
    %650 = vector.extractelement %632[%c15 : index] : vector<16xi32>
    %651 = arith.select %634, %635, %638 : i32
    %652 = vector.insertelement %651, %632[%c3 : index] : vector<16xi32>
    %653 = arith.select %634, %636, %635 : i32
    %654 = vector.insertelement %653, %652[%c0 : index] : vector<16xi32>
    %655 = arith.select %634, %637, %636 : i32
    %656 = vector.insertelement %655, %654[%c1 : index] : vector<16xi32>
    %657 = arith.select %634, %638, %637 : i32
    %658 = vector.insertelement %657, %656[%c2 : index] : vector<16xi32>
    %659 = arith.select %634, %639, %642 : i32
    %660 = vector.insertelement %659, %658[%c7 : index] : vector<16xi32>
    %661 = arith.select %634, %640, %639 : i32
    %662 = vector.insertelement %661, %660[%c4 : index] : vector<16xi32>
    %663 = arith.select %634, %641, %640 : i32
    %664 = vector.insertelement %663, %662[%c5 : index] : vector<16xi32>
    %665 = arith.select %634, %642, %641 : i32
    %666 = vector.insertelement %665, %664[%c6 : index] : vector<16xi32>
    %667 = arith.select %634, %643, %646 : i32
    %668 = vector.insertelement %667, %666[%c11 : index] : vector<16xi32>
    %669 = arith.select %634, %644, %643 : i32
    %670 = vector.insertelement %669, %668[%c8 : index] : vector<16xi32>
    %671 = arith.select %634, %645, %644 : i32
    %672 = vector.insertelement %671, %670[%c9 : index] : vector<16xi32>
    %673 = arith.select %634, %646, %645 : i32
    %674 = vector.insertelement %673, %672[%c10 : index] : vector<16xi32>
    %675 = arith.select %634, %647, %650 : i32
    %676 = vector.insertelement %675, %674[%c15 : index] : vector<16xi32>
    %677 = arith.select %634, %648, %647 : i32
    %678 = vector.insertelement %677, %676[%c12 : index] : vector<16xi32>
    %679 = arith.select %634, %649, %648 : i32
    %680 = vector.insertelement %679, %678[%c13 : index] : vector<16xi32>
    %681 = arith.select %634, %650, %649 : i32
    %682 = vector.insertelement %681, %680[%c14 : index] : vector<16xi32>
    %683 = arith.andi %482, %c2 : index
    %684 = arith.cmpi ne, %683, %c0 : index
    %685 = vector.extractelement %682[%c0 : index] : vector<16xi32>
    %686 = vector.extractelement %682[%c1 : index] : vector<16xi32>
    %687 = vector.extractelement %682[%c2 : index] : vector<16xi32>
    %688 = vector.extractelement %682[%c3 : index] : vector<16xi32>
    %689 = vector.extractelement %682[%c4 : index] : vector<16xi32>
    %690 = vector.extractelement %682[%c5 : index] : vector<16xi32>
    %691 = vector.extractelement %682[%c6 : index] : vector<16xi32>
    %692 = vector.extractelement %682[%c7 : index] : vector<16xi32>
    %693 = vector.extractelement %682[%c8 : index] : vector<16xi32>
    %694 = vector.extractelement %682[%c9 : index] : vector<16xi32>
    %695 = vector.extractelement %682[%c10 : index] : vector<16xi32>
    %696 = vector.extractelement %682[%c11 : index] : vector<16xi32>
    %697 = vector.extractelement %682[%c12 : index] : vector<16xi32>
    %698 = vector.extractelement %682[%c13 : index] : vector<16xi32>
    %699 = vector.extractelement %682[%c14 : index] : vector<16xi32>
    %700 = vector.extractelement %682[%c15 : index] : vector<16xi32>
    %701 = arith.select %684, %685, %687 : i32
    %702 = vector.insertelement %701, %682[%c2 : index] : vector<16xi32>
    %703 = arith.select %684, %686, %688 : i32
    %704 = vector.insertelement %703, %702[%c3 : index] : vector<16xi32>
    %705 = arith.select %684, %687, %685 : i32
    %706 = vector.insertelement %705, %704[%c0 : index] : vector<16xi32>
    %707 = arith.select %684, %688, %686 : i32
    %708 = vector.insertelement %707, %706[%c1 : index] : vector<16xi32>
    %709 = arith.select %684, %689, %691 : i32
    %710 = vector.insertelement %709, %708[%c6 : index] : vector<16xi32>
    %711 = arith.select %684, %690, %692 : i32
    %712 = vector.insertelement %711, %710[%c7 : index] : vector<16xi32>
    %713 = arith.select %684, %691, %689 : i32
    %714 = vector.insertelement %713, %712[%c4 : index] : vector<16xi32>
    %715 = arith.select %684, %692, %690 : i32
    %716 = vector.insertelement %715, %714[%c5 : index] : vector<16xi32>
    %717 = arith.select %684, %693, %695 : i32
    %718 = vector.insertelement %717, %716[%c10 : index] : vector<16xi32>
    %719 = arith.select %684, %694, %696 : i32
    %720 = vector.insertelement %719, %718[%c11 : index] : vector<16xi32>
    %721 = arith.select %684, %695, %693 : i32
    %722 = vector.insertelement %721, %720[%c8 : index] : vector<16xi32>
    %723 = arith.select %684, %696, %694 : i32
    %724 = vector.insertelement %723, %722[%c9 : index] : vector<16xi32>
    %725 = arith.select %684, %697, %699 : i32
    %726 = vector.insertelement %725, %724[%c14 : index] : vector<16xi32>
    %727 = arith.select %684, %698, %700 : i32
    %728 = vector.insertelement %727, %726[%c15 : index] : vector<16xi32>
    %729 = arith.select %684, %699, %697 : i32
    %730 = vector.insertelement %729, %728[%c12 : index] : vector<16xi32>
    %731 = arith.select %684, %700, %698 : i32
    %732 = vector.insertelement %731, %730[%c13 : index] : vector<16xi32>
    %733 = arith.divui %484, %c4 : index
    %734 = arith.muli %733, %c4 : index
    %735 = arith.muli %483, %c4 : index
    %736 = arith.remui %484, %c4 : index
    %737 = arith.addi %735, %736 : index
    %738 = arith.remui %469, %c2 : index
    %739 = arith.muli %738, %c32 : index
    %740 = arith.addi %739, %734 : index
    %741 = arith.divui %469, %c2 : index
    %742 = arith.muli %741, %c32 : index
    %743 = arith.addi %742, %737 : index
    %744 = arith.addi %6, %743 : index
    %745 = arith.addi %7, %740 : index
    %746 = arith.divui %744, %c8 : index
    %747 = arith.remui %744, %c8 : index
    %748 = arith.divui %747, %c4 : index
    %749 = arith.remui %744, %c4 : index
    %750 = arith.divui %745, %c4 : index
    %751 = arith.remui %745, %c4 : index
    %752 = arith.muli %746, %c8 : index
    %753 = arith.muli %748, %c4 : index
    %754 = arith.addi %752, %753 : index
    %755 = arith.addi %754, %749 : index
    %756 = arith.muli %750, %c4 : index
    %757 = arith.addi %756, %751 : index
    %758 = arith.cmpi slt, %757, %c0 : index
    %759 = arith.subi %c-1, %757 : index
    %760 = arith.select %758, %759, %757 : index
    %761 = arith.divsi %760, %c196 : index
    %762 = arith.subi %c-1, %761 : index
    %763 = arith.select %758, %762, %761 : index
    %764 = arith.remsi %757, %c196 : index
    %765 = arith.cmpi slt, %764, %c0 : index
    %766 = arith.addi %764, %c196 : index
    %767 = arith.select %765, %766, %764 : index
    %768 = arith.cmpi slt, %767, %c0 : index
    %769 = arith.subi %c-1, %767 : index
    %770 = arith.select %768, %769, %767 : index
    %771 = arith.divsi %770, %c14 : index
    %772 = arith.subi %c-1, %771 : index
    %773 = arith.select %768, %772, %771 : index
    %774 = arith.remsi %757, %c14 : index
    %775 = arith.cmpi slt, %774, %c0 : index
    %776 = arith.addi %774, %c14 : index
    %777 = arith.select %775, %776, %774 : index
    %778 = vector.extractelement %732[%c0 : index] : vector<16xi32>
    %779 = vector.insertelement %778, %cst_2[%c0 : index] : vector<4xi32>
    %780 = vector.extractelement %732[%c1 : index] : vector<16xi32>
    %781 = vector.insertelement %780, %779[%c1 : index] : vector<4xi32>
    %782 = vector.extractelement %732[%c2 : index] : vector<16xi32>
    %783 = vector.insertelement %782, %781[%c2 : index] : vector<4xi32>
    %784 = vector.extractelement %732[%c3 : index] : vector<16xi32>
    %785 = vector.insertelement %784, %783[%c3 : index] : vector<4xi32>
    %786 = arith.index_cast %763 : index to i32
    %787 = arith.index_cast %2 : index to i32
    %788 = arith.index_cast %755 : index to i32
    %789 = arith.index_cast %773 : index to i32
    %790 = arith.index_cast %777 : index to i32
    gpu.raw_buffer_store(%785, %arg2, %c0_i32, %786, %787, %788, %789, %790) : vector<4xi32>, memref<128x1x1024x14x14xi32>, i32, i32, i32, i32, i32, i32
    %791 = vector.extractelement %732[%c4 : index] : vector<16xi32>
    %792 = vector.insertelement %791, %cst_2[%c0 : index] : vector<4xi32>
    %793 = vector.extractelement %732[%c5 : index] : vector<16xi32>
    %794 = vector.insertelement %793, %792[%c1 : index] : vector<4xi32>
    %795 = vector.extractelement %732[%c6 : index] : vector<16xi32>
    %796 = vector.insertelement %795, %794[%c2 : index] : vector<4xi32>
    %797 = vector.extractelement %732[%c7 : index] : vector<16xi32>
    %798 = vector.insertelement %797, %796[%c3 : index] : vector<4xi32>
    %799 = arith.addi %755, %c8 : index
    %800 = arith.index_cast %763 : index to i32
    %801 = arith.index_cast %2 : index to i32
    %802 = arith.index_cast %799 : index to i32
    %803 = arith.index_cast %773 : index to i32
    %804 = arith.index_cast %777 : index to i32
    gpu.raw_buffer_store(%798, %arg2, %c0_i32, %800, %801, %802, %803, %804) : vector<4xi32>, memref<128x1x1024x14x14xi32>, i32, i32, i32, i32, i32, i32
    %805 = vector.extractelement %732[%c8 : index] : vector<16xi32>
    %806 = vector.insertelement %805, %cst_2[%c0 : index] : vector<4xi32>
    %807 = vector.extractelement %732[%c9 : index] : vector<16xi32>
    %808 = vector.insertelement %807, %806[%c1 : index] : vector<4xi32>
    %809 = vector.extractelement %732[%c10 : index] : vector<16xi32>
    %810 = vector.insertelement %809, %808[%c2 : index] : vector<4xi32>
    %811 = vector.extractelement %732[%c11 : index] : vector<16xi32>
    %812 = vector.insertelement %811, %810[%c3 : index] : vector<4xi32>
    %813 = arith.addi %755, %c16 : index
    %814 = arith.index_cast %763 : index to i32
    %815 = arith.index_cast %2 : index to i32
    %816 = arith.index_cast %813 : index to i32
    %817 = arith.index_cast %773 : index to i32
    %818 = arith.index_cast %777 : index to i32
    gpu.raw_buffer_store(%812, %arg2, %c0_i32, %814, %815, %816, %817, %818) : vector<4xi32>, memref<128x1x1024x14x14xi32>, i32, i32, i32, i32, i32, i32
    %819 = vector.extractelement %732[%c12 : index] : vector<16xi32>
    %820 = vector.insertelement %819, %cst_2[%c0 : index] : vector<4xi32>
    %821 = vector.extractelement %732[%c13 : index] : vector<16xi32>
    %822 = vector.insertelement %821, %820[%c1 : index] : vector<4xi32>
    %823 = vector.extractelement %732[%c14 : index] : vector<16xi32>
    %824 = vector.insertelement %823, %822[%c2 : index] : vector<4xi32>
    %825 = vector.extractelement %732[%c15 : index] : vector<16xi32>
    %826 = vector.insertelement %825, %824[%c3 : index] : vector<4xi32>
    %827 = arith.addi %755, %c24 : index
    %828 = arith.index_cast %763 : index to i32
    %829 = arith.index_cast %2 : index to i32
    %830 = arith.index_cast %827 : index to i32
    %831 = arith.index_cast %773 : index to i32
    %832 = arith.index_cast %777 : index to i32
    gpu.raw_buffer_store(%826, %arg2, %c0_i32, %828, %829, %830, %831, %832) : vector<4xi32>, memref<128x1x1024x14x14xi32>, i32, i32, i32, i32, i32, i32
    return
  }
}

