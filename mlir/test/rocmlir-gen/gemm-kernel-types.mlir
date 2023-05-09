// RUN: rocmlir-gen --arch %arch --operation gemm -t f16 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,F16
// RUN: rocmlir-gen --arch %arch --operation gemm -t i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8

// CHECK-LABEL: func @rock_gemm
// F16-SAME: (%{{.*}}: memref<3x1024x769xf16>, %{{.*}}: memref<3x769x512xf16>, %{{.*}}: memref<3x1024x512xf16>)
// I8-SAME: (%{{.*}}: memref<3x1024x769xi8>, %{{.*}}: memref<3x769x512xi8>, %{{.*}}: memref<3x1024x512xi32>)


// CHECK-LABEL: func @host_naive_gemm
// F16-SAME: (%{{.*}}: memref<3x1024x769xf16>, %{{.*}}: memref<3x769x512xf16>, %{{.*}}: memref<3x1024x512xf16>)
// I8-SAME: (%{{.*}}: memref<3x1024x769xi8>, %{{.*}}: memref<3x769x512xi8>, %{{.*}}: memref<3x1024x512xi64>)

// F16: arith.mulf
// F16-NEXT: arith.addf
// I8: arith.extsi
// I8-NEXT: arith.extsi
// I8-NEXT: arith.muli
// I8-NEXT: arith.addi
// CHECK-NEXT: linalg.yield
