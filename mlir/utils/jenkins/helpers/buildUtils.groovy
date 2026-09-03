// Build helpers: rocMLIR build, plus building/checking out CK and MIGraphX.
// Loaded by Jenkinsfile's Bootstrap stage; consumed as buildUtils.<method>().
// ON CHANGING THESE, ALSO CHANGE Jenkinsfile.downstream

import groovy.transform.Field

// Cross-helper handle, populated by Jenkinsfile's Bootstrap stage:
//   buildUtils.scmUtils = scmUtils
@Field def scmUtils

// Run `script` through bash with errexit + pipefail. Use this whenever a
// command pipes through tee/awk/grep/etc. so failures in the upstream command
// are not masked by the pipeline's last exit code. Plain `sh` runs under
// /bin/sh -xe (errexit but no pipefail); a #!/bin/bash shebang bypasses
// Jenkins's default flags, so we re-enable both explicitly here.
def shStrict(String script) {
    // When running inside withHealthyNode (REKICK_ROW_LOG set), mirror this step's output to a
    // per-row log so the retry handler can classify transient failures (e.g. GPU hang) that only
    // appear in stdout. pipefail keeps the real command's exit code from being masked by tee.
    if (env.REKICK_ROW_LOG) {
        sh "#!/bin/bash\nset -eo pipefail\n{\n${script}\n} 2>&1 | tee -a \"${env.REKICK_ROW_LOG}\""
    } else {
        sh "#!/bin/bash\nset -eo pipefail\n${script}"
    }
}

void buildProject(String target, String cmakeOpts) {
    timeout(time: 60, activity: true, unit: 'MINUTES') {
        // Configure with the CMake plugin (unchanged: same source/build dir resolution as before).
        cmakeBuild generator: 'Ninja',\
            buildDir: 'build',\
            buildType: 'RelWithDebInfo',\
            installation: 'InSearchPath',\
            cmakeArgs: """-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
              -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
              ${cmakeOpts}"""
        // Build via shStrict (was the plugin's `steps: [[args: target]]`, i.e. `ninja <target>` in
        // build dir) so build output is mirrored to the per-row log and build-time OOM
        // (ninja exit 137) can be classified as a per-server transient in withHealthyNode.
        // `ninja -C <dir>` is CWD-independent, matching the plugin's workspace-relative build dir.
        shStrict "ninja -C ${env.WORKSPACE}/build ${target}"
    }
}

void buildCK(String cmakeOpts, String buildTarget = '') {
    sh '[ ! -d build ] || rm -rf build'
    // CK's Unix Makefiles do not expose device_gemm_operations as a directly
    // buildable target, while Ninja handles this target graph correctly.
    cmakeBuild generator: (buildTarget ? 'Ninja' : 'Unix Makefiles'),\
        buildDir: 'build',\
        buildType: 'Release',\
        installation: 'InSearchPath',\
        cmakeArgs: """-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
                      -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
                     ${cmakeOpts}
                     """
    if (buildTarget) {
        shStrict "cmake --build build --target ${buildTarget} --parallel \$(nproc)"
    } else {
        shStrict 'cd build; make -j $(nproc)'
    }
}

void installCKGemmOnly(String installDir) {
    sh """#!/usr/bin/env bash
        set -euo pipefail

        install_dir="${installDir}"
        cmake_dir="\${install_dir}/lib/cmake/composable_kernel"
        mkdir -p "\${install_dir}/include/ck" "\${install_dir}/lib" "\${cmake_dir}"

        cp -R include/ck/. "\${install_dir}/include/ck/"
        cp -R library/include/ck/. "\${install_dir}/include/ck/"
        cp build/include/ck/config.h build/include/ck/version.h "\${install_dir}/include/ck/"
        cp build/lib/libdevice_gemm_operations.a "\${install_dir}/lib/"
        cp build/composable_kernelConfig.cmake \
           build/composable_kernelConfigVersion.cmake \
           "\${cmake_dir}/"

        mapfile -t gemm_export_files < <(find build -type f -name 'composable_kerneldevice_gemm_operationsTargets*.cmake' -print)
        if [ "\${#gemm_export_files[@]}" -eq 0 ]; then
            echo "Could not find CK device_gemm_operations CMake export files"
            exit 1
        fi
        cp "\${gemm_export_files[@]}" "\${cmake_dir}/"
    """
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
    shStrict 'cd build; make -j $(nproc)'
}

void getAndBuildMIGraphX(String cmakeOpts) {
    scmUtils.robustExternalCheckout('https://github.com/ROCm/AMDMIGraphX.git', params.MIGraphXBranch)
    buildMIGraphX(cmakeOpts)
}

void getAndBuildCK(String cmakeOpts, String buildTarget = '') {
    scmUtils.robustExternalCheckout('https://github.com/ROCm/composable_kernel.git', params.CKBranch)
    buildCK(cmakeOpts, buildTarget)
}

String ckFp8CmakeOptions(String chip) {
    // CK auto-derives CK_USE_*_FP8 from GPU_TARGETS, so we only force the
    // numeric macro values via CMAKE_CXX_FLAGS to keep CK's `#if` checks correct.
    if ("${chip}".startsWith('gfx94')) {
        return '''
                                                                        -DCMAKE_CXX_FLAGS="-O3 -DCK_USE_FNUZ_FP8=1"
                                                                        '''
    }
    if ("${chip}" == 'gfx950' || "${chip}".startsWith('gfx12')) {
        return '''
                                                                        -DCMAKE_CXX_FLAGS="-O3 -DCK_USE_OCP_FP8=1 -DCK_TILE_USE_OCP_FP8=1"
                                                                        '''
    }
    return '''
                                                                        -DCMAKE_CXX_FLAGS="-O3"
                                                                        '''
}

String ckDtypesCmakeOptions(String chip) {
    // The MLIR vs CK perf configs exercise f16/f32 GEMM, with int8 available
    // for the CK GEMM driver. Restricting DTYPES avoids building unused CK
    // operation instances on older arches such as gfx908.
    if ("${chip}".startsWith('gfx94') || "${chip}" == 'gfx950' || "${chip}".startsWith('gfx12')) {
        return '''
                                                                        -DDTYPES="fp8;bf8;fp16;fp32;int8"
                                                                        '''
    }
    return '''
                                                                        -DDTYPES="fp16;fp32;int8"
                                                                        '''
}

return this
