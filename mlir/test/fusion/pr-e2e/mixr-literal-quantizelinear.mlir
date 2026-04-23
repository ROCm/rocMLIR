// RUN: rocmlir-driver --host-pipeline=migraphx-linalg,highlevel %s | rocmlir-gen -rand=none -ph -pr -fut literal_quantizelinear  - | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-driver --host-pipeline=migraphx,highlevel %s | rocmlir-gen -rand=none -ph -pr -fut literal_quantizelinear  - | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// Values come from the python script from below
// CHECK: [2, -128, 127, 127, 0, -2, 127, -128, 24, -25]
func.func @literal_quantizelinear(%dummy : !migraphx.shaped<1xi8, 1>) -> !migraphx.shaped<10xsi8, 1> {
    // IEEE 754: nan = 0x7fc00000, +inf = 0x7f800000, -inf = 0xff800000
    // input: [1.0, -inf, inf, nan, 0.0, -1.0, 127.0, -128.0, 12.1, -12.4]
    %input = migraphx.literal (dense<[1.0, 0xff800000, 0x7f800000, 0x7fc00000, 0.0, -1.0, 127.0, -128.0, 12.1, -12.4]> : tensor<10xf32>) : <10xf32, 1>
    %scale = migraphx.literal (dense<0.5> : tensor<1xf32>) : <1xf32, 1>
    %result = migraphx.quantizelinear %input, %scale : <10xf32, 1>, <1xf32, 1> -> <10xsi8, 1>
    return %result : !migraphx.shaped<10xsi8, 1>
}


// import numpy as np
// import onnx
// from onnx import helper, TensorProto, numpy_helper
// import onnxruntime as ort
// 
// input_data = np.array(
//     [1.0, -np.inf, np.inf, np.nan, 0.0, -1.0, 127.0, -128.0, 12.1, 12.4],
//     dtype=np.float32,
// )
// scale_data = np.array(0.5, dtype=np.float32)
// zero_point_data = np.array(0, dtype=np.int8)
// 
// input_init = numpy_helper.from_array(input_data, name="x")
// scale_init = numpy_helper.from_array(scale_data, name="y_scale")
// zp_init = numpy_helper.from_array(zero_point_data, name="y_zero_point")
// 
// node = helper.make_node(
//     "QuantizeLinear",
//     inputs=["x", "y_scale", "y_zero_point"],
//     outputs=["y"],
// )
// 
// graph = helper.make_graph(
//     [node],
//     "literal_quantizelinear",
//     inputs=[],
//     outputs=[
//         helper.make_tensor_value_info("y", TensorProto.INT8, [10]),
//     ],
//     initializer=[input_init, scale_init, zp_init],
// )
// 
// model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
// model.ir_version = 10
// onnx.checker.check_model(model)
// onnx.save(model, "main.onnx")
// 
// sess = ort.InferenceSession("main.onnx", providers=["CPUExecutionProvider"])
// outputs = sess.run(None, {})
// print("input: ", input_data)
// print("scale: ", scale_data)
// print("output:", outputs[0])
// outputs: 
// input:  [   1.    -inf    inf    nan    0.    -1.   127.  -128.    12.1   12.4]
// scale:  0.5
// output: [   2 -128  127 -128    0   -2  127 -128   24   25]
