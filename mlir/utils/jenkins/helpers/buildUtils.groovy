// Build helpers: rocMLIR build, plus building/checking out CK and MIGraphX.
// Loaded by Jenkinsfile's Bootstrap stage; consumed as buildUtils.<method>().
// ON CHANGING THESE, ALSO CHANGE Jenkinsfile.downstream

// Run `script` through bash with errexit + pipefail. Use this whenever a
// command pipes through tee/awk/grep/etc. so failures in the upstream command
// are not masked by the pipeline's last exit code. Plain `sh` runs under
// /bin/sh -xe (errexit but no pipefail); a #!/bin/bash shebang bypasses
// Jenkins's default flags, so we re-enable both explicitly here.
def shStrict(String script) {
    sh "#!/bin/bash\nset -eo pipefail\n${script}"
}

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
                      -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF
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

return this
