def do_coverage() {
//    steps {
        // Build with profiling on, and just code-generation tests.
        sh 'rm -f build/CMakeCache.txt'
        sh 'rm -f build/*.profraw'
        buildProject('check-rocmlir-build-only',
                     '-DBUILD_FAT_LIBROCKCOMPILER=ON -DCMAKE_BUILD_TYPE=debug -DLLVM_BUILD_INSTRUMENTED_COVERAGE=ON')
        dir ('build') {
            // Run tests.
            sh 'ninja check-rocmlir'
            // Profile processing.
            sh '/opt/rocm/llvm/bin/llvm-profdata merge -sparse ./*.profraw -o ./coverage.profdata'
            sh '/opt/rocm/llvm/bin/llvm-cov report --object ./bin/rocmlir-opt --object ./bin/rocmlir-driver --object ./bin/rocmlir-gen --instr-profile ./coverage.profdata --ignore-filename-regex=external/llvm-project > ./coverage_${CHIP}.report'
            sh 'cat ./coverage_${CHIP}.report'
            sh '/opt/rocm/llvm/bin/llvm-cov show --object ./bin/rocmlir-opt --object ./bin/rocmlir-driver --object ./bin/rocmlir-gen --instr-profile ./coverage.profdata --ignore-filename-regex=external/llvm-project -Xdemangler=/opt/rocm/llvm/bin/llvm-cxxfilt > ./coverage_${CHIP}.info'
            sh '/opt/rocm/llvm/bin/llvm-cov show --object ./bin/rocmlir-opt --object ./bin/rocmlir-driver --object ./bin/rocmlir-gen --instr-profile ./coverage.profdata --ignore-filename-regex=external/llvm-project -Xdemangler=/opt/rocm/llvm/bin/llvm-cxxfilt --format=html > ./coverage_${CHIP}.html'
            // Upload to codecov.
            withCredentials([string(credentialsId: 'codecov-token-rocmlir', variable: 'CODECOV_TOKEN')]) {
                sh '''
                   curl -Os https://uploader.codecov.io/latest/linux/codecov && chmod +x ./codecov
                   proxy_opt=""
                   if [ -n "${http_proxy}" ]; then
                       proxy_opt="-U ${http_proxy}"
                   fi
                   ./codecov -t ${CODECOV_TOKEN} --flags "${CHIP}" -f ./coverage_${CHIP}.info ${proxy_opt}
                   '''
            }
        }
        archiveArtifacts artifacts: 'build/coverage*.report, build/coverage*.info, build/coverage*.html', onlyIfSuccessful: true
//    }
}
