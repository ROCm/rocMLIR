void buildProject(String target, String cmakeOpts) {
    timeout(time: 60, activity: true, unit: 'MINUTES') {
        cmakeBuild generator: 'Ninja',\
            buildDir: 'build',\
            buildType: 'RelWithDebInfo',\
            installation: 'InSearchPath',\
            steps: [[args: target]],\
            cmakeArgs: """-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
              -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
              ${cmakeOpts}"""
    }
}

void buildCK(String cmakeOpts) {
    sh '[ ! -d build ] || rm -rf build'
    cmakeBuild generator: 'Unix Makefiles',\
        buildDir: 'build',\
        buildType: 'Release',\
        installation: 'InSearchPath',\
        cmakeArgs: """-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
                      -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
                     ${cmakeOpts}
                     """
    sh 'cd build; make -j $(nproc)'
}

void buildMIGraphX(String cmakeOpts) {
    sh '[ ! -d build ] || rm -rf build'
    cmakeBuild generator: 'Unix Makefiles',\
        buildDir: 'build',\
        buildType: 'Release',\
        installation: 'InSearchPath',\
        cmakeArgs: """-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
                      -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
                     ${cmakeOpts}
                     """
    sh 'cd build; make -j $(nproc)'
}

void getAndBuildMIGraphX(String cmakeOpts) {
    git branch: params.MIGraphXBranch, poll: false,\
        url: 'https://github.com/ROCm/AMDMIGraphX.git'
    buildMIGraphX(cmakeOpts)
}

void getAndBuildCK(String cmakeOpts) {
    git branch: params.CKBranch, poll: false,\
        url: 'https://github.com/ROCm/composable_kernel.git'
    buildCK(cmakeOpts)
}
