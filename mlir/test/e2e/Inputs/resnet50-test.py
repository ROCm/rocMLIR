#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import argparse
import sys
from contextlib import redirect_stdout
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Red_Smooth_Saluki.jpg/590px-Red_Smooth_Saluki.jpg

def run_model_on_image(image_file):
    # Read image and massage appropriately.
    # (From https://keras.io/api/applications/#usage-examples-for-image-classification-models)
    img = image.load_img(image_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Stifle the "Downloading ..." message, which goes to stdout, because
    # we print the MLIR to stdout later and will pipe it to mlir-opt.
    with open('/dev/null', 'w') as f:
        with redirect_stdout(f):
            model = ResNet50(weights='imagenet')

    print(f">> Expected result is {tf.argmax(model(x), 1).numpy()}", file=sys.stderr)

    # (Can't find my bookmark that taught me this freezing technique.)
    # Single result because both tf.nn.top_k and tf.argsort produce tf.TopKV2
    # op which isn't convertible by mlir-opt.  One result is better than none.
    f = tf.function(lambda : tf.argmax(model(x), 1))
    ff = f.get_concrete_function()
    fff = convert_variables_to_constants_v2(ff)

    # Make MLIR from the function, append the support code that calls it and
    # prints the result(s).  The pass_pipeline is approximately what "tf-opt
    # --tf-to-tosa-pipeline" would do.
    mlir = tf.mlir.experimental.convert_function(fff, pass_pipeline='tf-standard-pipeline,canonicalize,builtin.func(affine-loop-fusion,affine-scalrep,tosa-fuse-bias-tf,tosa-legalize-tf,tosa-make-broadcastable),inline,symbol-dce')
    mlir = mlir.replace(" func @", " func private @") # Helps DCE after inlining
    edited_mlir = f"{mlir[0:-2]}\n\n{main_function(fff.name.decode('utf-8'))}\n}}\n"
    print(edited_mlir)

def main_function(callee_name):
    # The callee's return type is known because I inserted a tf.argmax above.
    return f'''
  func private @printF64(f64)
  func private @printI64(i64)
  func private @rtclock() -> f64
  func private @print_memref_i64(memref<*xi64>) -> ()
  func private @printNewline() -> ()
  func private @printOpen() -> ()
  func private @printClose() -> ()

  func @main() -> () {{
    %t0 = call @rtclock() : () -> f64
    %0 = call @{callee_name}() : () -> tensor<1xi64>
    %t1 = call @rtclock() : () -> f64
    %t1024 = arith.subf %t1, %t0 : f64
    %1 = bufferization.to_memref %0 :  memref<1xi64>
    %2 = memref.cast %1 : memref<1xi64> to memref<*xi64>
    call @print_memref_i64(%2) : (memref<*xi64>) -> ()
    call @printNewline() : () -> ()
    call @printF64(%t1024) : (f64) -> ()
    call @printNewline() : () -> ()
    return
  }}
'''

if __name__ == '__main__':
    print(f"TF version is {tf.__version__}.  sys.path is {sys.path}", file=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()
    run_model_on_image(args.image)


# Then it's all MLIR and execution, outside python or as a subprocess.
#
# cat maybe.mlir | tf-opt -tensor-bufferize --tf-to-tosa-pipeline | mlir-opt --tosa-partition --tosa-to-linalg-on-tensors --tosa-to-standard --linalg-detensorize -tensor-constant-bufferize -std-bufferize -linalg-bufferize -tensor-bufferize -func-bufferize -finalizing-bufferize --convert-linalg-to-loops --tosa-to-standard -lower-affine -convert-linalg-to-llvm --convert-scf-to-std --convert-math-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/home/pf/llvm-project/build/lib/libmlir_runner_utils.so,/home/pf/llvm-project/build/lib/libmlir_c_runner_utils.so
#
# should print "[176]"
