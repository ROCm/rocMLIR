def vs_miopen_rocblas() {
    dir('build') {
        sh 'date --utc +%Y-%m-%d > perf-run-date'
        // Run MLIR vs MIOpend perf benchmarks.
        sh """python3 ./bin/perfRunner.py --op=conv --batch_all \
               --configs_file=${WORKSPACE}/mlir/utils/performance/conv-configs \
               --tuning_db=${WORKSPACE}/build/mlir_tuning_${CHIP}.tsv \
               --quick_tuning_db=${WORKSPACE}/build/mlir_quick_tuning_${CHIP}.tsv"""
        // Run MLIR vs rocBLAS perf benchmarks
        sh """python3 ./bin/perfRunner.py --op=gemm --batch_all \
               --configs_file=${WORKSPACE}/mlir/utils/performance/gemm-configs \
               --tuning_db=${WORKSPACE}/build/mlir_tuning_${CHIP}.tsv \
               --quick_tuning_db=${WORKSPACE}/build/mlir_quick_tuning_${CHIP}.tsv"""
    }
}

def fusion() {
    dir('build') {
        // Run fusion resnet50 perf benchmarks
        sh """python3 ./bin/perfRunner.py --op=fusion \
--test_dir=${WORKSPACE}/mlir/test/fusion/resnet50-e2e/ --tuning_db=${WORKSPACE}/build/tuning_fusion_${CHIP}.tsv"""
        // Run bert perf benchmarks
        sh """python3 ./bin/perfRunner.py --op fusion \
--test_dir=${WORKSPACE}/mlir/test/xmir/bert-torch-tosa-e2e/ --tuning_db=${WORKSPACE}/build/tuning_fusion_${CHIP}.tsv"""
    }
}

def attention() {
    dir('build') {
        // Run attention benchmarks
        sh """python3 ./bin/perfRunner.py --op=attention -b \
               --configs_file=${WORKSPACE}/mlir/utils/performance/attention-configs \
               --tuning_db=${WORKSPACE}/build/mlir_tuning_${CHIP}.tsv"""
    }
}

def mlir_vs_ck() {
    dir('composable_kernel') {
        sh 'rm -rf composable_kernel'
        getAndBuildCK('''
            -DGPU_TARGETS="gfx1030;gfx908;gfx90a"
            -DCMAKE_CXX_FLAGS="-O3"
            -DCMAKE_PREFIX_PATH="/opt/rocm"
            -DCMAKE_INSTALL_PREFIX=${WORKSPACE}/composable_kernel/build/CKInstallDir
            -DCMAKE_BUILD_TYPE=Release
            ''')
        sh 'cd build/library; make install; cp ../composable_kernelConfig*.cmake ${WORKSPACE}/composable_kernel/build/CKInstallDir/lib/cmake/composable_kernel'
        sh 'echo `git rev-parse HEAD`'
    }
    sh 'rm build/CMakeCache.txt'
    buildProject("ck-benchmark-driver",
                 '''-DCMAKE_PREFIX_PATH=${WORKSPACE}/composable_kernel/build/CKInstallDir
                    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
                    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
                    -DROCMLIR_ENABLE_BENCHMARKS=ck''')

    dir('build') {
        sh """python3 ./bin/perfRunner.py --op=gemm --batch_all \
    --configs_file=${WORKSPACE}/mlir/utils/performance/gemm-configs \
    --tuning_db=${WORKSPACE}/build/mlir_tuning_${CHIP}.tsv --data-type f32 f16 i8_i8 --external-gemm-library CK"""
        sh 'python3 ./bin/createPerformanceReports.py ${CHIP} CK'
    }
}

def reports() {
    dir('build') {
        sh 'ls -l'
        sh 'python3 ./bin/createPerformanceReports.py ${CHIP} MIOpen'
        sh 'python3 ./bin/createPerformanceReports.py ${CHIP} rocBLAS'
        sh 'python3 ./bin/createFusionPerformanceReports.py ${CHIP}'
        sh 'python3 ./bin/perfRegressionReport.py ${CHIP}'
        sh 'python3 ./bin/perfRegressionReport.py ${CHIP} ./oldData/${CHIP}_mlir_vs_rocblas_perf.csv ./${CHIP}_mlir_vs_rocblas_perf.csv'
        sh 'mkdir -p reports && cp ./*.html reports'
    }
    postProcessPerfRes("${CHIP}")
}
