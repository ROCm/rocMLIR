def build() {
    def gpu_arch = get_gpu_architecture()
    sh 'rm -rf MIGraphX'
    dir('MIGraphX') {
        getAndBuildMIGraphX("""
                        -DCMAKE_PREFIX_PATH='${WORKSPACE}/MIGraphXDeps;/MIGraphXDeps;/opt/rocm'
                        -DMIGRAPHX_ENABLE_MLIR=On
                        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
                        -DMIGRAPHX_USE_HIPRTC=Off
                        -DGPU_TARGETS="${gpu_arch}"
                        """)
    }
}

def verify() {
    dir('MIGraphX/build') {
        timeout(time: 60, activity: true, unit: 'MINUTES') {
            withEnv(['MIGRAPHX_ENABLE_MLIR=1']) {
                // Verify MLIR unit tests
                sh 'make -j$(nproc) driver test_gpu_mlir'
                sh 'ctest -R test_gpu_mlir'
                // Verify ResNet50, Bert, Gpt2
                sh './bin/migraphx-driver verify --gpu --onnx /MIGraphXDeps/resnet50-v1-7.onnx'
                sh './bin/migraphx-driver verify --gpu --onnx /MIGraphXDeps/resnet50-v1-7.onnx --int8'
                sh './bin/migraphx-driver verify --gpu --onnx /MIGraphXDeps/bert_base_cased_1.onnx --fill1 input_ids --input-dim @input_ids 1 384'
                sh './bin/migraphx-driver verify --gpu --onnx /MIGraphXDeps/bert_base_cased_1.onnx --fill1 input_ids --input-dim @input_ids 1 384 --int8'
                sh './bin/migraphx-driver verify --gpu --onnx /MIGraphXDeps/distilgpt2_1.onnx --fill1 input_ids --input-dim @input_ids 1 384'
                sh './bin/migraphx-driver verify --gpu --onnx /MIGraphXDeps/distilgpt2_1.onnx --fill1 input_ids --input-dim @input_ids 1 384 --int8'
            }
        }
    }
    //Accuracy_checker will compare outputs from MIGraphX and onnx runtime
    dir('MIGraphX/tools/accuracy') {
        sh 'python3 accuracy_checker.py --onnx /MIGraphXDeps/resnet50-v1-7.onnx'
        sh 'python3 accuracy_checker.py --fill1 --onnx /MIGraphXDeps/bert_base_cased_1.onnx'
        sh 'python3 accuracy_checker.py --fill1 --onnx /MIGraphXDeps/distilgpt2_1.onnx'
    }
}
