// Canary (CANARY-prefixed RUN below): without fix-tosa-cast-rounding,
// upstream tosa-to-linalg must still emit math.roundeven for migraphx.convert
// float-to-int. If this breaks, the pass needs revisiting.
// RUN: rocmlir-opt -pass-pipeline="builtin.module(func.func(migraphx-to-tosa),func.func(tosa-to-linalg),func.func(fix-tosa-cast-rounding))" --split-input-file --mlir-print-debuginfo %s | FileCheck %s
// RUN: rocmlir-opt -pass-pipeline="builtin.module(func.func(migraphx-to-tosa),func.func(tosa-to-linalg))" --split-input-file --mlir-print-debuginfo %s | FileCheck %s --check-prefix=CANARY

// Positive: math.roundeven with RTZ fused loc tag feeding into fptosi
// inside linalg.generic should be removed.
// CHECK-LABEL: @cast_rtz_tagged
// CHECK: linalg.generic
// CHECK-NOT: math.roundeven
// CHECK: arith.fptosi {{%.+}} : f32 to i32
func.func @cast_rtz_tagged(%arg0: tensor<4xf32>) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xi32>) {
  ^bb0(%in: f32, %out: i32):
    %1 = math.roundeven %in : f32 loc(fused<"rocmlir.rtz_cast">["test":0:0])
    %2 = arith.fptosi %1 : f32 to i32
    linalg.yield %2 : i32
  } -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Positive: RTZ-tagged roundeven with float clamp chain (narrowing-to-i8 path).
// CHECK-LABEL: @cast_rtz_tagged_with_clamp
// CHECK: linalg.generic
// CHECK-NOT: math.roundeven
// CHECK: arith.minimumf
// CHECK: arith.maximumf
// CHECK: arith.fptosi
func.func @cast_rtz_tagged_with_clamp(%arg0: tensor<4xf16>) -> tensor<4xi8> {
  %empty = tensor.empty() : tensor<4xi8>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf16>) outs(%empty : tensor<4xi8>) {
  ^bb0(%in: f16, %out: i8):
    %1 = math.roundeven %in : f16 loc(fused<"rocmlir.rtz_cast">["test":0:0])
    %cst_min = arith.constant -1.280000e+02 : f16
    %cst_max = arith.constant 1.270000e+02 : f16
    %2 = arith.minimumf %1, %cst_max : f16
    %3 = arith.maximumf %2, %cst_min : f16
    %4 = arith.fptosi %3 : f16 to i8
    linalg.yield %4 : i8
  } -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

// Positive: RTZ-tagged roundeven with the i32 saturation merge pattern that
// upstream tosa-to-linalg emits for f32->i32. Here the rounded value feeds
// BOTH a float clamp -> fptosi (low side) AND an arith.cmpf -> arith.select
// that picks between an i32 saturation constant and the fptosi result. This
// is the only test that covers the cmpf+select+yield branch of
// reachesFPToSI; if upstream changes its lowering, this test should still
// pin the matcher behaviour.
// CHECK-LABEL: @cast_rtz_tagged_i32_saturate
// CHECK: linalg.generic
// CHECK-NOT: math.roundeven
// CHECK: arith.maximumf %{{[^,]+}}, %{{[^ ]+}} : f32
// CHECK: arith.fptosi %{{.*}} : f32 to i32
// CHECK: arith.cmpf uge, %{{[^,]+}}, %{{[^ ]+}} : f32
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
func.func @cast_rtz_tagged_i32_saturate(%arg0: tensor<4xf32>) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %cst_min = arith.constant -2.14748365E+9 : f32
  %cst_max = arith.constant 2.14748365E+9 : f32
  %c_int_max = arith.constant 2147483647 : i32
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xi32>) {
  ^bb0(%in: f32, %out: i32):
    %1 = math.roundeven %in : f32 loc(fused<"rocmlir.rtz_cast">["test":0:0])
    %2 = arith.maximumf %1, %cst_min : f32
    %3 = arith.fptosi %2 : f32 to i32
    %4 = arith.cmpf uge, %1, %cst_max : f32
    %5 = arith.select %4, %c_int_max, %3 : i32
    linalg.yield %5 : i32
  } -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Negative: RTZ-tagged roundeven whose result has a sibling user outside the
// recognized cast chain (here arith.divf) MUST NOT be removed. The generic
// itself only yields i32 so the integer-output guard passes; this test pins
// the strict-matcher property of reachesFPToSI: even though one sibling user
// (arith.fptosi) does reach the cast chain, the other (arith.divf) does not,
// so the matcher bails to avoid silently changing the divf input from
// `round(%in)` to `%in`.
// CHECK-LABEL: @cast_rtz_tagged_extra_user
// CHECK: linalg.generic
// CHECK: math.roundeven
// CHECK: arith.divf
// CHECK: arith.fptosi
func.func @cast_rtz_tagged_extra_user(%arg0: tensor<4xf32>) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xi32>) {
  ^bb0(%in: f32, %out: i32):
    %1 = math.roundeven %in : f32 loc(fused<"rocmlir.rtz_cast">["test":0:0])
    %2 = arith.fptosi %1 : f32 to i32
    %3 = arith.divf %1, %in : f32
    %4 = arith.fptosi %3 : f32 to i32
    %5 = arith.addi %2, %4 : i32
    linalg.yield %5 : i32
  } -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Negative: RTZ-tagged generic that yields a float result alongside the
// int cast result. The pass requires exactly one (non-i1 integer) output
// from the generic, so this multi-output case is rejected -- otherwise
// removing roundeven would silently change the yielded float value from
// `max(round(%in), c)` to `max(%in, c)`.
// CHECK-LABEL: @cast_rtz_generic_with_float_output
// CHECK: linalg.generic
// CHECK: math.roundeven
// CHECK-DAG: arith.fptosi
// CHECK-DAG: arith.maximumf
func.func @cast_rtz_generic_with_float_output(%arg0: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
  %empty_i = tensor.empty() : tensor<4xi32>
  %empty_f = tensor.empty() : tensor<4xf32>
  %cst = arith.constant 1.0 : f32
  %0:2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty_i, %empty_f : tensor<4xi32>, tensor<4xf32>) {
  ^bb0(%in: f32, %out_i: i32, %out_f: f32):
    %1 = math.roundeven %in : f32 loc(fused<"rocmlir.rtz_cast">["test":0:0])
    %2 = arith.fptosi %1 : f32 to i32
    %3 = arith.maximumf %1, %cst : f32
    linalg.yield %2, %3 : i32, f32
  } -> (tensor<4xi32>, tensor<4xf32>)
  return %0#0, %0#1 : tensor<4xi32>, tensor<4xf32>
}

// -----

// Negative: RTZ-tagged generic with a sibling i1 output computed via
// arith.cmpf on the rounded value. This is the dangerous case: every user
// of math.roundeven is recognized by reachesFPToSI (fptosi for the i32
// branch, cmpf for the i1 branch), so without the single-output guard the
// pass would strip the rounding and silently flip the i1 result for any
// %in whose RNE-rounding crosses the comparison threshold.
// CHECK-LABEL: @cast_rtz_tagged_multi_result_i1
// CHECK: linalg.generic
// CHECK: math.roundeven
// CHECK-DAG: arith.fptosi
// CHECK-DAG: arith.cmpf
func.func @cast_rtz_tagged_multi_result_i1(%arg0: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xi1>) {
  %empty_i32 = tensor.empty() : tensor<4xi32>
  %empty_i1 = tensor.empty() : tensor<4xi1>
  %cst = arith.constant 0.0 : f32
  %0:2 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty_i32, %empty_i1 : tensor<4xi32>, tensor<4xi1>) {
  ^bb0(%in: f32, %out_i32: i32, %out_i1: i1):
    %1 = math.roundeven %in : f32 loc(fused<"rocmlir.rtz_cast">["test":0:0])
    %2 = arith.fptosi %1 : f32 to i32
    %3 = arith.cmpf une, %1, %cst : f32
    linalg.yield %2, %3 : i32, i1
  } -> (tensor<4xi32>, tensor<4xi1>)
  return %0#0, %0#1 : tensor<4xi32>, tensor<4xi1>
}

// -----

// Negative: only the *input* block argument carries the RTZ tag (which can
// happen when this generic consumes the result of a previously-tagged cast).
// The generic itself, the math.roundeven, and the output block arg are all
// untagged, so this generic is NOT an RTZ-tagged cast lowering and the
// roundeven must be preserved. Without restricting the block-arg scan to
// the output args, the matcher would wrongly strip the rounding.
// CHECK-LABEL: @cast_rtz_input_arg_tagged_only
// CHECK: linalg.generic
// CHECK: math.roundeven
// CHECK: arith.fptosi
func.func @cast_rtz_input_arg_tagged_only(%arg0: tensor<4xf32>) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xi32>) {
  ^bb0(%in: f32 loc(fused<"rocmlir.rtz_cast">["test":0:0]), %out: i32):
    %1 = math.roundeven %in : f32
    %2 = arith.fptosi %1 : f32 to i32
    linalg.yield %2 : i32
  } -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Negative: RTZ-tagged single-output generic with i1 output. ONNX/PyTorch
// float-to-bool semantics is "non-zero" rather than truncation, so even
// though MIGraphXToTosa never tags an f32->i1 cast today, the matcher
// rejects i1 outputs as defense-in-depth.
// CHECK-LABEL: @cast_rtz_tagged_i1_output
// CHECK: linalg.generic
// CHECK: math.roundeven
// CHECK: arith.fptosi
func.func @cast_rtz_tagged_i1_output(%arg0: tensor<4xf32>) -> tensor<4xi1> {
  %empty = tensor.empty() : tensor<4xi1>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xi1>) {
  ^bb0(%in: f32, %out: i1):
    %1 = math.roundeven %in : f32 loc(fused<"rocmlir.rtz_cast">["test":0:0])
    %2 = arith.fptosi %1 : f32 to i1
    linalg.yield %2 : i1
  } -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// Negative: math.roundeven WITHOUT RTZ tag (e.g. from quantization) must
// NOT be removed even though it feeds into fptosi.
// CHECK-LABEL: @cast_no_tag_quantization
// CHECK: linalg.generic
// CHECK: math.roundeven
// CHECK: arith.fptosi
func.func @cast_no_tag_quantization(%arg0: tensor<4xf32>) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xi32>) {
  ^bb0(%in: f32, %out: i32):
    %1 = math.roundeven %in : f32
    %2 = arith.fptosi %1 : f32 to i32
    linalg.yield %2 : i32
  } -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Negative: math.roundeven outside linalg.generic must NOT be removed.
// CHECK-LABEL: @roundeven_outside_generic
// CHECK: math.roundeven
func.func @roundeven_outside_generic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg0[%c0] : tensor<4xf32>
  %1 = math.roundeven %0 : f32 loc(fused<"rocmlir.rtz_cast">["test":0:0])
  %2 = tensor.from_elements %1, %1, %1, %1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// -----

// Negative: quantizelinear through the full MIGraphX pipeline must preserve
// math.roundeven (quantization needs RNE, not RTZ).
// CHECK-LABEL: @mlir_quantizelinear
// CHECK: math.roundeven
func.func @mlir_quantizelinear(%arg0: !migraphx.shaped<1x1x2x2xf32, 4x4x2x1>) -> !migraphx.shaped<1x1x2x2xi8, 4x4x2x1> attributes {kernel = "mixr"} {
  %scale = migraphx.literal (dense<0.5> : tensor<1x1x2x2xf32>) : <1x1x2x2xf32, 4x4x2x1>
  %zp = migraphx.literal (dense<0> : tensor<1x1x2x2xi8>) : <1x1x2x2xi8, 4x4x2x1>
  %0 = migraphx.quantizelinear %arg0, %scale, %zp : <1x1x2x2xf32, 4x4x2x1>, <1x1x2x2xf32, 4x4x2x1>, !migraphx.shaped<1x1x2x2xi8, 4x4x2x1> -> <1x1x2x2xi8, 4x4x2x1>
  return %0 : !migraphx.shaped<1x1x2x2xi8, 4x4x2x1>
}

// -----

// Positive integration: migraphx.convert from f32 to i32 through full
// pipeline. Verifies the RTZ loc tag survives migraphx-to-tosa + tosa-to-linalg
// and fix-tosa-cast-rounding removes the math.roundeven.
// CHECK-LABEL: @convert_f32_to_i32
// CHECK: linalg.generic
// CHECK-NOT: math.roundeven
// CHECK: arith.fptosi
// CANARY-LABEL: @convert_f32_to_i32
// CANARY: linalg.generic
// CANARY: math.roundeven
// CANARY: arith.fptosi
func.func @convert_f32_to_i32(%arg0: !migraphx.shaped<4xf32, 1>) -> !migraphx.shaped<4xi32, 1> attributes {kernel = "mixr"} {
  %0 = migraphx.convert %arg0 : <4xf32, 1> to <4xi32, 1>
  return %0 : !migraphx.shaped<4xi32, 1>
}

// -----

// Positive integration: migraphx.convert from f16 to i8 through full pipeline
// (exercises the clamped/saturated path).
// CHECK-LABEL: @convert_f16_to_i8
// CHECK: linalg.generic
// CHECK-NOT: math.roundeven
// CHECK: arith.fptosi
// CANARY-LABEL: @convert_f16_to_i8
// CANARY: linalg.generic
// CANARY: math.roundeven
// CANARY: arith.fptosi
func.func @convert_f16_to_i8(%arg0: !migraphx.shaped<4xf16, 1>) -> !migraphx.shaped<4xi8, 1> attributes {kernel = "mixr"} {
  %0 = migraphx.convert %arg0 : <4xf16, 1> to <4xi8, 1>
  return %0 : !migraphx.shaped<4xi8, 1>
}

// -----

// Negative integration: float-to-float convert is not RTZ-tagged (only
// float-to-int casts get the tag in MIGraphXToTosa::ConvertConverter), and
// upstream lowers it via arith.truncf without ever inserting math.roundeven.
// CHECK-LABEL: @convert_f32_to_f16
// CHECK-NOT: math.roundeven
// CHECK: arith.truncf
func.func @convert_f32_to_f16(%arg0: !migraphx.shaped<4xf32, 1>) -> !migraphx.shaped<4xf16, 1> attributes {kernel = "mixr"} {
  %0 = migraphx.convert %arg0 : <4xf32, 1> to <4xf16, 1>
  return %0 : !migraphx.shaped<4xf16, 1>
}

// -----

// Negative integration: float-to-bool convert is excluded from RTZ tagging
// in MIGraphXToTosa, and upstream tosa-to-linalg lowers it via arith.cmpf
// (non-zero comparison) -- not roundeven+fptosi -- so no math.roundeven is
// emitted at all. ONNX/PyTorch bool cast semantics is preserved.
// CHECK-LABEL: @convert_f32_to_i1
// CHECK-NOT: math.roundeven
// CHECK-NOT: arith.fptosi
// CHECK: arith.cmpf une
func.func @convert_f32_to_i1(%arg0: !migraphx.shaped<4xf32, 1>) -> !migraphx.shaped<4xi1, 1> attributes {kernel = "mixr"} {
  %0 = migraphx.convert %arg0 : <4xf32, 1> to <4xi1, 1>
  return %0 : !migraphx.shaped<4xi1, 1>
}

// -----

// Negative: untagged roundeven inside linalg.generic that does NOT feed
// into fptosi must NOT be removed (e.g. standalone rounding in user code).
// CHECK-LABEL: @roundeven_untagged_no_fptosi
// CHECK: linalg.generic
// CHECK: math.roundeven
func.func @roundeven_untagged_no_fptosi(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0 : tensor<4xf32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = math.roundeven %in : f32
    linalg.yield %1 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
