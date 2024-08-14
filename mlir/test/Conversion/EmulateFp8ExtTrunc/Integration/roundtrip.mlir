// COM: This is a host-side test and doesn't require a GPU
// RUN: sed -e 's/##TYPE##/f8E4M3FNUZ/g' -e 's/##OCP##/false/g' %s | \
// RUN: rocmlir-opt -emulate-fp8-ext-trunc - | \
// RUN: rocmlir-driver --host-pipeline=runner | \
// RUN: mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | \
// RUN: FileCheck %s --check-prefixes=CHECK,F8E4M3FNUZ

// RUN: sed -e 's/##TYPE##/f8E5M2FNUZ/g' -e 's/##OCP##/false/g' %s | \
// RUN: rocmlir-opt -emulate-fp8-ext-trunc - | \
// RUN: rocmlir-driver --host-pipeline=runner | \
// RUN: mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | \
// RUN: FileCheck %s --check-prefixes=CHECK,F8E5M2FNUZ

// RUN: sed -e 's/##TYPE##/f8E4M3FN/g'  -e 's/##OCP##/true/g' %s | \
// RUN: rocmlir-opt -emulate-fp8-ext-trunc - | \
// RUN: rocmlir-driver --host-pipeline=runner | \
// RUN: mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | \
// RUN: FileCheck %s --check-prefixes=CHECK,F8E4M3FN

// RUN: sed -e 's/##TYPE##/f8E5M2/g' -e 's/##OCP##/true/g' %s | \
// RUN: rocmlir-opt -emulate-fp8-ext-trunc - | \
// RUN: rocmlir-driver --host-pipeline=runner | \
// RUN: mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | \
// RUN: FileCheck %s --check-prefixes=CHECK,F8E5M2


func.func private @printI64(i64) -> ()
func.func private @printF32(f32) -> ()
func.func private @printNewline() -> ()
func.func private @printComma() -> ()

func.func @testTruncExt(%id : i64, %input: f32) {
  %truncated = arith.truncf %input : f32 to ##TYPE##
  %expanded = arith.extf %truncated : ##TYPE## to f32
  call @printI64(%id) : (i64) -> ()
  call @printComma() : () -> ()
  call @printF32(%expanded) : (f32) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @printFail(%wanted : i8, %got : i8, %intermediate : f32) {
  vector.print str "FAIL: \n"
  %wantedExt = arith.extui %wanted : i8 to i64
  %gotExt = arith.extui %got : i8 to i64
  call @printI64(%wantedExt) : (i64) -> ()
  call @printComma() : () -> ()
  call @printI64(%gotExt) : (i64) -> ()
  call @printComma() : () -> ()
  call @printF32(%intermediate) : (f32) -> ()
  %interbits = arith.bitcast %intermediate : f32 to i32
  %interbits64 = arith.extui %interbits : i32 to i64
  call @printComma() : () -> ()
  call @printI64(%interbits64) : (i64) -> ()
  call @printNewline() : () -> ()
  return
}

func.func @testAllExtTrunc() {
  %idx0 = arith.constant 0 : index
  %idx1 = arith.constant 1 : index
  %idx256 = arith.constant 256 : index

  scf.for %idxI = %idx0 to %idx256 step %idx1 {
    %i = arith.index_cast %idxI : index to i8
    %iF8 = arith.bitcast %i : i8 to ##TYPE##
    %iFloat = arith.extf %iF8 : ##TYPE## to f32
    %oF8 = arith.truncf %iFloat : f32 to ##TYPE##
    %o = arith.bitcast %oF8 : ##TYPE## to i8
    %mismatch = arith.cmpi ne, %i, %o : i8
    scf.if %mismatch {
      %false = arith.constant 0 : i1
      %true  = arith.constant 1 : i1
      scf.if %##OCP## {
        // 7d, 7e, and 7f are all NaN.  They become 7f800001 as f32
        // and all f32 NaN become 7f on truncation.
        %nan = arith.constant 127 : i8
        %omasked = arith.andi %o, %nan : i8
        %onanp = arith.cmpi eq, %omasked, %nan : i8
        %two = arith.constant 2 : i8
        %lower = arith.subi %nan, %two : i8
        %imasked = arith.andi %i, %nan : i8
        %inanp = arith.cmpi sge, %imasked, %lower : i8
        %notfail = arith.andi %onanp, %inanp : i1
        scf.if %notfail {
        } else {
          func.call @printFail(%i, %o, %iFloat) : (i8, i8, f32) -> ()
        }
      } else {
        func.call @printFail(%i, %o, %iFloat) : (i8, i8, f32) -> ()
      }
    }
  }
  return
}

func.func @main() {
  %id0 = arith.constant 0 : i64
  %in0 = arith.constant 0.0 : f32
  // CHECK: 0, 0
  call @testTruncExt(%id0, %in0) : (i64, f32) -> ()

  %id1 = arith.constant 1 : i64
  %in1 = arith.constant -0.0 : f32
  // F8E5M2FNUZ: 1, 0
  // F8E4M3FNUZ: 1, 0
  // F8E5M2:     1, -0
  // F8E4M3FN:   1, -0
  call @testTruncExt(%id1, %in1) : (i64, f32) -> ()

  %id2 = arith.constant 2 : i64
  %in2 = arith.constant 0x7f800000  : f32
  // F8E5M2FNUZ: 2, nan
  // F8E4M3FNUZ: 2, nan
  // F8E5M2:     2, inf
  // F8E4M3FN:   2, nan
  //                384?
  call @testTruncExt(%id2, %in2) : (i64, f32) -> ()

  %id3 = arith.constant 3 : i64
  // COM: Largest e5m2
  %in3 = arith.constant 57344.0 : f32
  // F8E5M2FNUZ: 3, 57344
  // F8E4M3FNUZ: 3, 240
  // F8E5M2:     3, 57344
  // F8E4M3FN:   3, nan
  call @testTruncExt(%id3, %in3) : (i64, f32) -> ()

  %id4 = arith.constant 4 : i64
  %in4 = arith.constant -57344.0 : f32
  // F8E5M2FNUZ: 4, -57344
  // F8E4M3FNUZ: 4, -240
  // F8E5M2:     4, -57344
  // F8E4M3FN:   4, -nan
  call @testTruncExt(%id4, %in4) : (i64, f32) -> ()

  %id5 = arith.constant 5 : i64
  %in5 = arith.constant 240.0 : f32
  // COM: Not enough mantissa
  // F8E5M2FNUZ: 5, 256
  // F8E4M3FNUZ: 5, 240
  // F8E5M2:     5, 256
  // F8E4M3FN:   5, 240
  call @testTruncExt(%id5, %in5) : (i64, f32) -> ()

  // CHECK-NOT: FAIL
  call @testAllExtTrunc() : () -> ()
  return
}
