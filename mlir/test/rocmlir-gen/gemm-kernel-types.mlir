// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,F16,FLOAT
// RUN: rocmlir-gen --arch %arch --operation gemm -t i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8,INT
// RUN: rocmlir-gen --arch %arch --operation gemm -t i8 -c_dtype i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8-I8,INT
// RUN: rocmlir-gen --arch %arch --operation gemm -ta fp8 -tb bf8 -tc f32 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8,FLOAT
// RUN: rocmlir-gen --arch %arch --operation gemm -t fp8_bf8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8,FLOAT

// CHECK-LABEL: func @rock_gemm
// F16-SAME: (%{{.*}}: memref<3x1024x769xf16>, %{{.*}}: memref<3x769x512xf16>, %{{.*}}: memref<3x1024x512xf16>)
// I8-SAME: (%{{.*}}: memref<3x1024x769xi8>, %{{.*}}: memref<3x769x512xi8>, %{{.*}}: memref<3x1024x512xi32>)
// I8-I8-SAME: (%{{.*}}: memref<3x1024x769xi8>, %{{.*}}: memref<3x769x512xi8>, %{{.*}}: memref<3x1024x512xi8>)
// FP8-BF8-SAME: (%{{.*}}: memref<3x1024x769xf8E4M3FNUZ>, %{{.*}}: memref<3x769x512xf8E5M2FNUZ>, %{{.*}}: memref<3x1024x512xf32>)

// CHECK-LABEL: func @host_naive_gemm
// F16-SAME: (%{{.*}}: memref<3x1024x769xf32>, %{{.*}}: memref<3x769x512xf32>, %{{.*}}: memref<3x1024x512xf32>)
// I8-SAME: (%{{.*}}: memref<3x1024x769xi8>, %{{.*}}: memref<3x769x512xi8>, %{{.*}}: memref<3x1024x512xi64>)
// I8-I8-SAME: (%{{.*}}: memref<3x1024x769xi8>, %{{.*}}: memref<3x769x512xi8>, %{{.*}}: memref<3x1024x512xi64>)
// FP8-BF8-SAME: (%{{.*}}: memref<3x1024x769xf32>, %{{.*}}: memref<3x769x512xf32>, %{{.*}}: memref<3x1024x512xf32>)

// FLOAT: arith.mulf
// FLOAT-NEXT: arith.addf
// INT: arith.extsi
// INT-NEXT: arith.extsi
// INT-NEXT: arith.muli
// INT-NEXT: arith.addi
// CHECK-NEXT: linalg.yield
