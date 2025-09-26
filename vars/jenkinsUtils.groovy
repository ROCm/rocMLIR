import groovy.transform.Field
import java.util.concurrent.ConcurrentHashMap
import org.jenkinsci.plugins.workflow.support.steps.AgentOfflineException

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

// Lightweight Git probe: verifies auth + network + ref exists
void gitHealthCheck() {
    // Check if git installed
    sh "git --version"

    // Check if git commands are healthy
    String repo = scm?.userRemoteConfigs?.getAt(0)?.url
    String cred = scm?.userRemoteConfigs?.getAt(0)?.credentialsId
    String ref  = env.CHANGE_ID ? "refs/pull/${env.CHANGE_ID}/head"
               : env.BRANCH_NAME ? "refs/heads/${env.BRANCH_NAME}"
               : "HEAD"

    if (!repo || !cred) {
        error "[healthcheck] SCM not configured (repo='${repo}', cred='${cred}')"
    }
    echo "[healthcheck] Probing git: repo=${repo}, ref=${ref}"

    timeout(time: 2, unit: 'MINUTES') {
        withCredentials([usernamePassword(credentialsId: cred,
                                          usernameVariable: 'GIT_USER',
                                          passwordVariable: 'GIT_PASS')]) {
            withEnv(["REPO=${repo}", "REF=${ref}"]) {
                sh '''
                    set -eu
                    ASK="$(mktemp)"; trap 'rm -f "$ASK"' EXIT
                    printf '#!/bin/sh\nprintf %s "$GIT_PASS"\n' > "$ASK"
                    chmod +x "$ASK"
                    GIT_ASKPASS="$ASK" \
                    git -c credential.username="$GIT_USER" \
                        ls-remote --exit-code "$REPO" "$REF" >/dev/null
                '''
            }
        }
    }
    echo "[healthcheck] Git OK"
}

// Retry checkout without shallow clone if GitSCM chokes on a specific SHA
void robustScmCheckout() {
    int maxAttempts = 2
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            // This inner 'try' handles the "reference is not a tree" fallback
            try {
                echo "[SCM] Attempting checkout (${attempt}/${maxAttempts})..."
                checkout scm
                echo "[SCM] Checkout successful"
                // If checkout succeeds, exit the function immediately
                return
            } catch (err) {
                def msg = "${err}".toLowerCase()
                if (!msg.contains("reference is not a tree")) {
                    // If it's not the "reference is not a tree" error, re-throw it to be caught by the outer block
                    throw err
                }

                // This is the fallback logic for the "reference is not a tree" error
                echo "[SCM] Default checkout failed: ${err}. Retrying ONCE with robust deep clone"
                String repo = scm?.userRemoteConfigs?.getAt(0)?.url
                String cred = scm?.userRemoteConfigs?.getAt(0)?.credentialsId
                String ref  = env.CHANGE_ID ? "refs/pull/${env.CHANGE_ID}/head"
                                          : env.BRANCH_NAME ? "refs/heads/${env.BRANCH_NAME}"
                                          : "HEAD"
                
                def deepScm = [
                    $class: 'GitSCM',
                    userRemoteConfigs: [[url: repo, credentialsId: cred, refspec: "+${ref}:${ref}"]],
                    branches: [[name: ref]],
                    doGenerateSubmoduleConfigurations: false,
                    extensions: [
                        [$class: 'CloneOption', depth: 0, shallow: false, noTags: false, honorRefspec: true],
                        [$class: 'CheckoutOption', timeout: 20]
                    ]
                ]
                checkout(deepScm)
                echo "[SCM] Deep clone checkout successful."
                // If the deep clone succeeds, exit the function
                return
            }
        } catch (err) {
            // This outer 'catch' block is specifically for retrying network errors
            def msg = "${err}".toLowerCase()
            if (msg.contains("connection reset by peer") && attempt < maxAttempts) {
                echo "[SCM] Attempt ${attempt}/${maxAttempts} failed due to a network error."
                echo "[SCM] Waiting 2 minutes before retrying..."
                sleep(time: 2, unit: 'MINUTES')
                // The loop will now continue to the next attempt.
            } else {
                // This is either not a network error, or it was the final attempt. Fail the build
                echo "[SCM] Unrecoverable SCM error after ${attempt} attempt(s)."
                throw err
            }
        }
    }
}

def resetGPUs() {
    // Abort this if runs longer than 10 minutes
    timeout(time: 10, unit: 'MINUTES') {
        // Run the reset, but don't fail the build if anything is wrong
        def rc = sh(
            script: '''
            reset_all_gpus() {
                echo "Scanning GPUs..."
                GPU_IDS=$(rocm-smi | awk '/^[0-9]+[[:space:]]+[0-9]+[[:space:]]+0x/ { print $1 }')
                if [ -z "$GPU_IDS" ]; then
                    echo "WARNING: No GPUs found to reset."
                    return 0
                fi
                for id in $GPU_IDS; do
                    echo "Resetting GPU ID: $id"
                    if ! rocm-smi --gpureset -d $id; then
                        echo "WARNING: Unable to reset GPU $id"
                    fi
                    sleep 2
                done
                return 0
            }
            reset_all_gpus
            ''',
            returnStatus: true
        )
        if (rc != 0) {
            echo "WARNING: reset_all_gpus exited with code ${rc}, but continuing anyway"
        }
    }
}

def advancedNodeCheck(Map params) {
    script {
        echo "Jenkins-side PATH = '${env.PATH}'"
    }
    boolean doCleanWs = params.doCleanWs
    boolean doGPUcheck = params.doGPUcheck
    resetGPUs()

    if (doCleanWs) {
        timeout(time: 15, unit: 'MINUTES', activity: true) {
            cleanWs()
        }
    }

    timeout(time: 5, unit: 'MINUTES', activity: true) { sh 'docker version' }

    ['ls -la /dev/kfd', 'ls -la /dev/dri'].each { cmd ->
        timeout(time: 5, unit: 'MINUTES', activity: true) { sh cmd }
    }

    String nodeSpecMessage = "\nNode specification:\n"
    timeout(time: 5, unit: 'MINUTES', activity: true) {
        nodeSpecMessage += "\nOS info:\n" + sh(script: 'sudo dkms status', returnStdout: true).trim() + '\n'
    }
    echo nodeSpecMessage

    if (env.NODE_LABELS && !env.NODE_LABELS.contains('build-only')) {
        timeout(time: 5, unit: 'MINUTES', activity: true) { sh 'rocminfo' }
        timeout(time: 5, unit: 'MINUTES', activity: true) { sh 'rocm-smi' }
        timeout(time: 5, unit: 'MINUTES', activity: true) { sh 'cat /opt/rocm/.info/version' }
        if (doGPUcheck) {
            timeout(time: 5, unit: 'MINUTES', activity: true) {
                def n = sh(script: "lspci | grep -e 'controller' -e 'accelerators' | grep 'AMD/ATI' | wc -l",
                           returnStdout: true).trim().toInteger()
                if (n == 0) {
                    error "No GPUs detected on ${env.NODE_NAME}"
                }
                echo "Number of GPUs on ${env.NODE_NAME}: ${n}"
            }
        }
    } else {
        echo 'Skipping GPU checks…'
    }
}

def checkNodeHealth(Map opts = [:]) {
    advancedNodeCheck(
        doCleanWs:                 opts.get('doCleanWs', true),
        doGPUcheck:                opts.get('doGPUcheck', true)
    )
}

Map<String,String> dockerArgs() {
    echo "Getting Docker args from ${env.NODE_NAME}..."
    def run = { cmd -> sh(script: cmd, returnStdout: true).trim() }
    // discover devices
    String renderFlags = run("ls -1 /dev/dri/renderD* 2>/dev/null || true")
                       .split()
                       .collect { "--device=${it}" }
                       .join(' ')
    // /dev/kfd appears only on GPU-enabled nodes
    boolean haveKfd = sh(script: '[ -e /dev/kfd ]', returnStatus: true) == 0
    String kfdFlg = haveKfd ? '--device=/dev/kfd' : ''

    // Get the GIDs of the render and video groups
    String renderGid = run("getent group render | cut -d':' -f3")
    String videoGid = run("getent group video | cut -d':' -f3")

    String args = """
        ${kfdFlg} \
        ${renderFlags} \
        --group-add ${renderGid} --group-add ${videoGid}
    """.trim().replaceAll(/\s+/, ' ')

    DOCKER_ARGS_BY_NODE[env.NODE_NAME] = args
    echo "Received Docker args for ${env.NODE_NAME}: ${args}"
    return DOCKER_ARGS_BY_NODE // ConcurrentHashMap
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

void showEnv() {
    echo "$env.NODE_NAME"
    sh 'cat /etc/os-release'
    sh 'ulimit -a'
    // Ignore rocm-smi failures in ixt-sjc2-05
    sh '/opt/rocm/bin/rocm-smi || true'
    sh '/opt/rocm/bin/rocm_agent_enumerator'
    sh 'id'
    sh 'printenv'
}

String dockerImage() {
    // If this is being changed please change Dockerfile.migraphx-ci's base image as well
    return 'rocm/mlir:rocm6.4-latest'
}

String dockerImageCIMIGraphX() {
    return 'rocm/mlir-migraphx-ci:rocm6.4-latest'
}

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

void splitConfigFile(String inputFilePath, String outputFilePath, int run, int totalSplits = 5) {
    sh """
    lines=\$(grep -Ev '(^\\s*\$|^\\s*#)' ${inputFilePath} | wc -l)
    lines_per_chunk=\$(((lines + ${totalSplits} - 1) / ${totalSplits}))
    start_line=\$((lines_per_chunk * (${run} - 1) + 1))
    end_line=\$((lines_per_chunk * ${run}))
    
    grep -Ev '(^\\s*\$|^\\s*#)' ${inputFilePath} | sed -n "\${start_line},\${end_line}p" | tee ${outputFilePath}
    """
}

void postProcessPerfRes(String chip) {
    publishHTML (target: [
        allowMissing: false,
        alwaysLinkToLastBuild: false,
        keepAll: true,
        reportDir: 'build/reports',
        reportFiles: "${chip}_MLIR_Performance_Changes.html,${chip}_MLIR_vs_MIOpen.html,${chip}_MLIR_Performance_Changes_Gemm.html,${chip}_MLIR_vs_rocBLAS.html,${chip}_MLIR_vs_CK.html,${chip}_conv_fusion.html,${chip}_gemm_fusion.html",
        reportName: "Performance report for ${chip}"
    ])

  if (fileExists("build/${chip}_mlir_vs_miopen_perf_for_plot.csv")) {
    plot csvFileName: "${chip}_plot-nightly-perf-results-000001.csv",\
        csvSeries: [[file: "build/${chip}_mlir_vs_miopen_perf_for_plot.csv", displayTableFlag: false]],\
        title: "Test performance summary ${chip}, Conv",\
        yaxis: 'TFlops',\
        style: 'line',\
        group: 'Performance plots'
  }
  if (fileExists("build/${chip}_mlir_vs_rocblas_perf_for_plot.csv")) {
    plot csvFileName: "${chip}_plot-nightly-perf-results-gemm-000001.csv",\
        csvSeries: [[file: "build/${chip}_mlir_vs_rocblas_perf_for_plot.csv", displayTableFlag: false]],\
        title: "Test performance summary ${chip}, GEMM",\
        yaxis: 'TFlops',\
        style: 'line',\
        group: 'Performance plots'
  }
    // Save results for future comparison
    archiveArtifacts artifacts: 'build/*_mlir_*.csv,build/perf-run-date', allowEmptyArchive: true, onlyIfSuccessful: true
}

//Get the GPU name of architecture
def get_gpu_architecture() {
    try {
        def result = sh(script: 'rocminfo', returnStdout: true).trim()
        def arch_pattern = /Name:\s+amdgcn-amd-amdhsa--(gfx\d+\w*((:\w+[\+\-]))*)/
        def matches = (result =~ arch_pattern)
        if (matches) {
            return matches[0][1]
        }
        return 'N/A'
    } catch (Exception e) {
        echo "Error getting GPU architecture name: ${e}"
        return 'N/A'
    }
}

//makes sure multiple builds are not triggered for branch indexing
def resetBuild() {
    if (currentBuild.getPreviousBuild() == null
        || currentBuild.getPreviousBuild().getBuildCauses().toString().contains('BranchIndexingCause')) {
        def buildNumber = BUILD_NUMBER as int;
        if (buildNumber > 1)
            milestone(buildNumber - 1);
        milestone(buildNumber)
    }
}

void setHeartbeat() {
    script {
        System.setProperty("org.jenkinsci.plugins.durabletask.BourneShellScript.HEARTBEAT_CHECK_INTERVAL", "86400");
    }
}

String getLabelFromCodepath(String codepath) {
    echo "codepath is ${codepath}"
    String label = ''
    if (codepath == "mfma") {
        label = 'mlir && (gfx950 || gfx942 || gfx908 || gfx90a)'
    } else if (codepath == "navi21") {
        // For non-performance related testing, use both workstations (gfx1030w)
        // and server nodes (gfx1030)
        label = 'mlir && ( gfx1030w || gfx1030 )'
    } else if (codepath == "vanilla"){
        label = 'mlir'
    } else if (codepath == "navi3x") {
        label = 'mlir && ( gfx1100 || gfx1101 )'
    } else if (codepath == "navi4x") {
        label = 'mlir && ( gfx1200 || gfx1201 )'
    } else {
        echo "${codepath} is not supported"
        label = 'wrongLabel'
    }
    echo "label is ${label}"
    return label
}

String getLabelFromChip(String chip) {
    switch (chip) {
        case "gfx906":
            return getLabelFromCodepath("vanilla")
        case "gfx908":
            return "mlir && gfx908"
        case "gfx90a":
            return "mlir && gfx90a"
        case "gfx942":
            return "mlir && gfx942"
        case "gfx950":
            return "mlir && gfx950"
        case "gfx1030":
            // For [Tune MLIR Kernels] and [Performance report] stages,
            // fix the vm-5 workstation for testing
            return "mlir && vm-5"
        case "gfx1100":
            return "mlir && gfx1100"
        case "gfx1101":
            return "mlir && gfx1101"
        case "gfx1200":
            return "mlir && gfx1200"
        case "gfx1201":
            return "mlir && gfx1201"
    }
}

int setLitWorkerCount() {
    int limit_lit_workers = 8
    def gpu_arch = get_gpu_architecture()
    if (gpu_arch.contains('gfx908') || gpu_arch.contains('gfx90a')) {
        limit_lit_workers = 20
    } else if (gpu_arch.contains('gfx942') || gpu_arch.contains('gfx950')) {
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
    timeout(time: 300, activity: true, unit: 'MINUTES') {
        dir('build') {
            sh """python3 ./bin/parameterSweeps.py -j 5 ${CONFIG} --log-failures"""
        }
    }
}

boolean shouldRunFromCodepath(String codepath) {
    // Run vanilla on public CI
    if ((codepath == "vanilla") && (params.canXdlops == false)) {
        return true
    }
    // Run mfma on private CI
    if ((codepath == "mfma") && params.canXdlops) {
        return true
    }
    // Run navi21 on private nightly or weekly CI if it is not disabled
    if (params.canXdlops && (params.disableNavi21 == false) && (codepath == "navi21") &&
        (params.nightly || params.weekly)) {
        return true
    }
    // Run navi3x on private CI if it is not disabled
    if (params.canXdlops && (params.disableNavi3x == false) && (codepath == "navi3x")) {
        return true
    }
    // Run navi4x on private CI if it is not disabled
    if (params.canXdlops && (params.disableNavi4x == false) && (codepath == "navi4x")) {  
        return true;  
    }  
    return false
}

boolean shouldRunFromChip(String chip) {
    switch (chip) {
        default:
            return shouldRunFromCodepath("vanilla")
        case "gfx90a":
            // Special case because all our "vanilla" hosts are gfx90a.
            return params.disable90a == false &&
                   (shouldRunFromCodepath("mfma") || shouldRunFromCodepath("vanilla"))
        case "gfx908":
            return params.disable908 == false && shouldRunFromCodepath("mfma")
        case "gfx942":
            return params.disable942 == false && shouldRunFromCodepath("mfma")
        case "gfx950":
            return params.disable950 == false && shouldRunFromCodepath("mfma")
        case "gfx1030":
            return shouldRunFromCodepath("navi21")
        case "gfx1100":
        case "gfx1101":
            return shouldRunFromCodepath("navi3x")
        case "gfx1200":
        case "gfx1201":
            return shouldRunFromCodepath("navi4x")
    }
}

void archivePerfDB() {
    // Note: add additional architectures here
    dir ('build/perfDB') {
        def architectures = params.canXdlops ? ['gfx908', 'gfx90a', 'gfx942', 'gfx1100', 'gfx1201'] : ['vanilla']
        for (arch in architectures) {
            try {
                unstash name: "MLIR-PerfDB-${arch}"
            } catch (Exception e) {
                echo "No stash found for MLIR-PerfDB-${arch}, skipping."
            }
        }
        sh 'date --utc +%Y-%m-%d >tuning-date'
    }
    archiveArtifacts artifacts: 'build/perfDB/**',\
    onlyIfSuccessful: true
}

boolean shouldRunBuildAndTest(String codepath) {
    // When default codepath is selected, we test mfma, navi21, navi3x and navi4x on
    // private CI and vanilla on public CI
    if (params.codepath == "default" && shouldRunFromCodepath(codepath))
        return true

    // When a particular codepath is selected, we only test the codepath
    // on private CI
    if (params.codepath == codepath && params.canXdlops) {
        if (params.codepath == "mfma") return true
        if (params.codepath == "vanilla") return true
        if (params.codepath == "navi21" && params.disableNavi21 == false) return true
        if (params.codepath == "navi3x" && params.disableNavi3x == false) return true
        if (params.codepath == "navi4x" && params.disableNavi4x == false) return true
        return false
    }
}

boolean isNotNavi3x(String chip) {
    return "${chip}" != 'gfx1100' && "${chip}" != 'gfx1101'
}

void collectCoverageData(String profdata, String cov, String cpath) {
    sh """
       rm -f *.profraw
       # Arbitrarily 150 GB;  we typically see 125 GB of *.profraw.
       if [ `df --output=avail -k . | tail -1l` -lt 153600000 ]; then
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

// Run the body on a node that passes the supplied healthChecks() block
// The health check is retried on fresh executors; the body is not retried.
// This function also retries the main 'body' if it fails due to a recoverable node-related issue (e.g., agent disconnect).
def withHealthyNode(String baseLabel, Closure<?> healthChecks, Closure<?> body, int maxAttempts = 3) {
    def blacklist = [] // nodes and pods that already failed the check
    int attempt = 0
    boolean done = false

    while (!done && attempt < maxAttempts) {
        attempt += 1

        // Build a dynamic label that excludes everything that failed before
        def expr = new StringBuilder(baseLabel)
        blacklist.each { expr.append(' && !').append(it) }

        echo "[withHealthyNode] attempt #${attempt}: looking for '${expr}'"
        node(expr.toString()) {
            // Retry ONLY the health-check. We don't want to retry the actual stages
            try {
                stage("Health checks on ${env.NODE_NAME}") {
                    healthChecks()
                    gitHealthCheck()
                }
            } catch (Exception err) {
                echo "[withHealthyNode] ❌  ${env.NODE_NAME} rejected: ${err}"
                blacklist << env.NODE_NAME
                // return exits the node {} block here, not the whole function. Some groovy magic
                return
            }
            stage("Node selected") {
                // Health-check passed. Do real work
                echo "[withHealthyNode] ✅  using ${env.NODE_NAME}"
            }
            try {
                body()
                // If body succeeds, we're done with the loop
                done = true
                
            } catch (Exception err) {
                def msg = "${err}".toLowerCase()
                def isNodeFailure = msg.contains("removed or offline") || msg.contains("issue with creating launcher for agent") ||
                                    err instanceof org.jenkinsci.plugins.workflow.support.steps.AgentOfflineException
                
                if (isNodeFailure) {
                    echo "[withHealthyNode] Execution on ${env.NODE_NAME} failed due to a node-specific issue. Blacklisting the node and retrying.."
                    echo "[withHealthyNode] Error was: ${err}"
                    blacklist << env.NODE_NAME
                    // return will exit the node block, and the 'while' loop will continue to the next attempt
                    // 'done' variable is still false, so the loop continues if maxAttempts is not reached.
                    return 
                } else {
                    // This is a regular build/test/whatever failure, not a node issue.
                    echo "[withHealthyNode] Execution failed with a non-recoverable error on ${env.NODE_NAME}"
                    echo "[withHealthyNode] Error was: ${err}"
                    // Re-throw the exception to fail the build immediately
                    throw err
                }
            }
        }
    }

    if (!done) {
        error "No healthy node found for '${baseLabel}' after ${maxAttempts} attempts"
    }
}