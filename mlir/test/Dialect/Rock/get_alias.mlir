// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getAlias for TransformMapAttr
//===----------------------------------------------------------------------===//
// CHECK: #[[MAP:.*]] = #rock.transform_map<PassThrough [] at [] -> [] at []>
// CHECK: #[[ALIAS:.*]] = #transform_map
module {
  // This attribute should be aliased as #transform_map
  %0 = "test.use_attr"() {attr = #rock.transform_map<PassThrough [] at [] -> [] at []>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Test: getAlias for GeneralGemmParamsAttr
//===----------------------------------------------------------------------===//
// CHECK: #[[GEN:.*]] = #rock.general_gemm_params<>
module {
  // This attribute should be aliased as #general_gemm_params
  %0 = "test.use_attr"() {attr = #rock.general_gemm_params<>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Test: getAlias for XdlopsGemmParamsAttr
//===----------------------------------------------------------------------===//
// CHECK: #[[XDL:.*]] = #rock.xdlops_gemm_params<>
module {
  // This attribute should be aliased as #xldops_gemm_params
  %0 = "test.use_attr"() {attr = #rock.xdlops_gemm_params<>} : () -> ()
}

//===----------------------------------------------------------------------===//
// Test: getAlias for unrelated attribute (should not alias)
//===----------------------------------------------------------------------===//
// CHECK: #[[STR:.*]] = "hello"
module {
  // This attribute should not be aliased
  %0 = "test.use_attr"() {attr = "hello"} : () -> ()
}

//===----------------------------------------------------------------------===//
// Test: getAlias for attribute of another dialect (should not alias)
//===----------------------------------------------------------------------===//
// CHECK: #[[STD:.*]] = #builtin.type<"i32">
module {
  // This attribute should not be aliased
  %0 = "test.use_attr"() {attr = #builtin.type<"i32">} : () -> ()
}