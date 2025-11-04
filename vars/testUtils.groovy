void preMergeCheck(String codepath) {
    // Only do static check on mfma codepath during PR CI
    if ( (params.nightly == false) && (codepath == "mfma") ) {
        echo "Performing Static Test (preMergeCheck)"
        sh '''
        if [ ! -f ./build/compile_commands.json ];  then
          echo "No compile commands, bailing."
          exit 1
        fi
        if [ ! -f ./compile_commands.json ]; then
          ln -s build/compile_commands.json compile_commands.json
        fi
        '''
        def targetBranch = env.CHANGE_TARGET
        if (!targetBranch) {
            targetBranch = "develop"
        }
        if (params.ignoreExternalLinting == true) {
            sh "python3 ./mlir/utils/jenkins/static-checks/premerge-checks.py --base-commit=origin/${targetBranch} --ignore-external"
        }
        else {
            sh "python3 ./mlir/utils/jenkins/static-checks/premerge-checks.py --base-commit=origin/${targetBranch}"
        }
    } else {
        echo "Static Test step skipped"
    }
}

void preMergeCheckPackage(String codepath) {
    // Only do static check on mfma codepath during PR CI
    if ( (params.nightly == false) && (codepath == "mfma") ) {
        echo "Checking if the fat library target list is accurate"
        dir('build') {
            sh '../mlir/utils/jenkins/static-checks/get_fat_library_deps_list.pl > ./librockcompiler_deps.cmake.new'
        }
        sh 'diff -up mlir/tools/rocmlir-lib/librockcompiler_deps.cmake ./build/librockcompiler_deps.cmake.new'
    } else {
        echo "Skipping fat library target list check"
    }
}

int setLitWorkerCount() {
    int limit_lit_workers = 8
    def gpu_arch = get_gpu_architecture()
    if (gpu_arch.contains('gfx908') || gpu_arch.contains('gfx90a')) {
        limit_lit_workers = 20
    } else if (gpu_arch.contains('gfx942')) {
        limit_lit_workers = 64
    }
    return limit_lit_workers
}

void build_fixedE2ETests(String codepath) {
    // Limit the number of lit workers for gfx908, gfx90a to (8, 30) on CI as a workaround for issue #1845 and #1841
    int limit_lit_workers = setLitWorkerCount()
    buildProject('check-mlir-build-only check-rocmlir-build-only', """
              -DROCMLIR_DRIVER_PR_E2E_TEST_ENABLED=${params.nightly ? '0' : '1'}
              -DROCMLIR_DRIVER_E2E_TEST_ENABLED=${params.nightly ? '1' : '0'}
              -DROCK_E2E_TEST_ENABLED=${params.nightly ? '1' : '0'}
              -DROCMLIR_DRIVER_TEST_GPU_VALIDATION=1
              -DLLVM_LIT_ARGS='-v --time-tests --timeout=3600 --max-failures=1 -j ${limit_lit_workers}'
              -DCMAKE_EXPORT_COMPILE_COMMANDS=1
             """)
}

void check_randomE2ETests(String codepath) {
    // Limit the number of lit workers for gfx908, gfx90a to (8, 30) on CI as a workaround for issue #1845 and #1841
    int limit_lit_workers = setLitWorkerCount()
    buildProject('check-rocmlir', """
              -DROCMLIR_DRIVER_PR_E2E_TEST_ENABLED=0
              -DROCMLIR_DRIVER_E2E_TEST_ENABLED=1
              -DROCK_E2E_TEST_ENABLED=1
              -DROCMLIR_DRIVER_RANDOM_DATA_SEED=1
              -DROCMLIR_DRIVER_TEST_GPU_VALIDATION=0
              -DLLVM_LIT_ARGS='-v --time-tests --timeout=3600 --max-failures=1 -j ${limit_lit_workers}'
              -DCMAKE_EXPORT_COMPILE_COMMANDS=1
             """)
}

void parameterSweep(String CONFIG, String codepath) {
    int limit_lit_workers = setLitWorkerCount()
    timeout(time: 300, activity: true, unit: 'MINUTES') {
        dir('build') {
            sh """python3 ./bin/parameterSweeps.py -j ${limit_lit_workers} ${CONFIG} --log-failures"""
        }
    }
}

void collectCoverageData(String profdata, String cov, String cpath) {
    sh """
       rm -f *.profraw
       # Arbitrarily 150 GB;  we typically see 125 GB of *.profraw.
       if [ `df --output=avail -k . | tail -1` -lt 153600000 ]; then
          echo Not enough free disk space for profiling.
          exit 1
       fi
       ninja check-rocmlir
       # Profile processing.
       ${profdata} merge -sparse ./*.profraw -o ./coverage.profdata
       rm -f build/*.profraw
       ${cov} report --object ./bin/rocmlir-opt --object ./bin/rocmlir-driver      \
          --object ./bin/rocmlir-gen --instr-profile ./coverage.profdata           \
          --ignore-filename-regex=external/llvm-project > ./coverage_${cpath}.report
       cat ./coverage_${cpath}.report
       ${cov} export --object ./bin/rocmlir-opt --object ./bin/rocmlir-driver      \
          --object ./bin/rocmlir-gen --instr-profile ./coverage.profdata           \
          --ignore-filename-regex=external/llvm-project --format=lcov              \
          --compilation-dir ${WORKSPACE} > ./coverage_${cpath}.lcov
       ${cov} show --object ./bin/rocmlir-opt --object ./bin/rocmlir-driver        \
          --object ./bin/rocmlir-gen --instr-profile ./coverage.profdata           \
          --ignore-filename-regex=external/llvm-project -Xdemangler=llvm-cxxfilt   \
          --format=html > ./coverage_${cpath}.html
       """
}
