// RUN: rocmlir-opt %s | FileCheck %s
// Test DXGML type system

// CHECK-LABEL: @test_scalar_types
dxgml.function @test_scalar_types(
  %i8: !dxgml.int8,
  %i16: !dxgml.int16,
  %i32: !dxgml.int32,
  %i64: !dxgml.int64,
  %u8: !dxgml.uint8,
  %u16: !dxgml.uint16,
  %u32: !dxgml.uint32,
  %u64: !dxgml.uint64,
  %f16: !dxgml.float16,
  %f32: !dxgml.float32,
  %f64: !dxgml.float64,
  %bf16: !dxgml.bfloat16,
  %bool: !dxgml.bool
) -> !dxgml.int32 {
  dxgml.return %i32 : !dxgml.int32
}

// CHECK-LABEL: @test_tensor_types
dxgml.function @test_tensor_types(
  %t1: !dxgml.tensor<1x4x224x224x!dxgml.float16>,
  %t2: !dxgml.tensor<32x32x3x3x!dxgml.float32>,
  %t3: !dxgml.tensor<1024x!dxgml.int64>
) -> !dxgml.tensor<1x4x224x224x!dxgml.float16> {
  dxgml.return %t1 : !dxgml.tensor<1x4x224x224x!dxgml.float16>
}

// CHECK-LABEL: @test_special_float_types
dxgml.function @test_special_float_types(
  %fp8e4m3: !dxgml.float8e4m3fn,
  %fp8e4m3fnuz: !dxgml.float8e4m3fnuz,
  %fp8e5m2fnuz: !dxgml.float8e5m2fnuz,
  %fp8e8m0: !dxgml.float8e8m0fnu,
  %fp4: !dxgml.float4e2m1fn
) -> !dxgml.float8e4m3fn {
  dxgml.return %fp8e4m3 : !dxgml.float8e4m3fn
}
