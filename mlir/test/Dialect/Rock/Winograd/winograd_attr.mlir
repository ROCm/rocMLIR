// Tests for WinogradParamsAttr parsing and printing.
// Verifies that the attribute roundtrips through rocmlir-opt.

// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @winograd_attr_roundtrip
func.func @winograd_attr_roundtrip() attributes {
  // CHECK: winograd_params = #rock.winograd_params
  winograd_params = #rock.winograd_params<
    familyId = 6, nGroups = 304, channelMode = 2,
    dataPath = "fp32_fp32acc_f2x3_stride1">
} {
  return
}

// CHECK-LABEL: func.func @winograd_attr_fury
func.func @winograd_attr_fury() attributes {
  // CHECK: winograd_params = #rock.winograd_params
  winograd_params = #rock.winograd_params<
    familyId = 3, nGroups = 120, channelMode = 0,
    dataPath = "fp16_fp16acc_f2x3_c16_stride1">
} {
  return
}

// CHECK-LABEL: func.func @winograd_attr_v21
func.func @winograd_attr_v21() attributes {
  // CHECK: winograd_params = #rock.winograd_params
  winograd_params = #rock.winograd_params<
    familyId = 0, nGroups = 60, channelMode = 2,
    dataPath = "fp32_f2x3_stride1">
} {
  return
}
