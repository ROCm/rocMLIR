// Node lifecycle helpers: GPU reset, health checks, Docker discovery,
// docker image names, and the withHealthyNode() retry harness.
// Loaded by Jenkinsfile's Bootstrap stage; consumed as nodeUtils.<method>().
// ON CHANGING THESE, ALSO CHANGE Jenkinsfile.downstream

import groovy.transform.Field
import java.util.concurrent.ConcurrentHashMap

// ConcurrentHashMap helps when we need to write variables in parallel
// one instance for the whole run
@Field
ConcurrentHashMap<String,String> DOCKER_ARGS_BY_NODE = new ConcurrentHashMap<>()

// Characters from the end of the per-row log scanned to classify a transient failure;
// the decisive cause sits at the end.
@Field
final int FAILURE_LOG_TAIL_CHARS = 1000000

// Cross-helper handle, populated by Jenkinsfile's Bootstrap stage:
//   nodeUtils.scmUtils = scmUtils
// Used by withHealthyNode() to invoke scmUtils.gitHealthCheck().
@Field
def scmUtils

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

    if (doCleanWs) {
        timeout(time: 15, unit: 'MINUTES', activity: true) {
            cleanWs()
        }
    }

    resetGPUs()

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
    return 'rocm/mlir:rocm7.2-latest'
}

String dockerImageCIMIGraphX() {
    return 'rocm/mlir-migraphx-ci:rocm7.2-latest'
}

def retryDockerOperation(String description, Closure operation) {
    int attempt = 0
    def result = null
    retry(10) {
        attempt += 1
        try {
            result = operation()
        } catch (err) {
            echo "[Docker retry] ${description} failed on attempt ${attempt}/10 on ${env.NODE_NAME}: ${err}"
            if (attempt < 10) {
                echo "[Docker retry] Waiting 5 seconds before retrying ${description}"
                sleep(time: 5, unit: 'SECONDS')
            }
            throw err
        }
    }
    return result
}

// For when the docker image is in a private repo
void explicitDockerLogin() {
    withCredentials([usernamePassword(credentialsId: 'DOCKER_HUB_CREDS',
                                      usernameVariable: 'D_USER',
                                      passwordVariable: 'D_PASS')]) {
        retryDockerOperation('docker login to DockerHub') {
            sh '''
                set +x
                printf "%s\n" "$D_PASS" | docker login -u "$D_USER" --password-stdin
            '''
        }
    }
}

def pullDockerImage(String imageName) {
    def img = docker.image(imageName)
    retryDockerOperation("docker pull ${imageName}") {
        img?.pull()
    }
    return img
}

// Get the base GPU chip name as reported by the runtime (e.g. gfx1200, gfx942).
def get_gpu_architecture() {
    try {
        def result = sh(script: 'rocminfo', returnStdout: true).trim()
        def arch_pattern = /Name:\s+amdgcn-amd-amdhsa--(gfx[0-9a-z]+)/
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

// Run the body on a node that passes the supplied healthChecks() block
// The health check is retried on fresh executors; the body is not retried.
// This function also retries the main 'body' if it fails due to a recoverable node-related issue (e.g., agent disconnect).
// Genuine logic/test failures that must never be retried/re-kicked. Shared veto for the
// per-server classifier below and ciLogic's whole-job classifier (invoked via its nodeUtils
// handle as nodeUtils.realTestFailureSignals()).
List<String> realTestFailureSignals() {
    return [
        'failed tests (',                   // lit
        'error: no match found',            // FileCheck
        'filecheck error',
        '*** summary of failures ***',      // conv/perf sweeps
        'failing configurations',           // attention sweeps
        'tuning failed: detected errors',
        'invalid mlir created',             // MIGraphX
    ]
}

// Group-1 transients that can be retried on a fresh node in-pipeline (per matrix row), as opposed
// to whole-job transients like "no healthy node found" (handled by the post-block re-kick). `text`
// is the thrown exception plus the row's console tail. Case-insensitive. Deliberately excludes
// "no healthy node found"/"[withHealthyNode] transient" (nothing to retry on), "InterruptedException"
// (failFast collateral), and all genuine test-failure markers.
boolean isPerServerTransient(String text) {
    if (!text) return false
    String t = text.toLowerCase()

    // GPU lost/hung on this node; these surface in the test stdout, not in the thrown exception.
    // Pre-veto: a dead GPU also makes lit report spurious test failures, so it wins over realSignals.
    def gpuSignals = [
        'hiperror_t.hiperrornodevice',
        'unable to reset gpu',
        'unsupported hip gpu architecture: n/a',
        'no performance report found for n/a',
        'gpu hang',
        'hw exception by gpu',
    ]
    if (gpuSignals.any { t.contains(it) }) return true

    // Veto: genuine test failures are never per-server retried (mirror the whole-job classifier).
    if (realTestFailureSignals().any { t.contains(it) }) return false
    if (t =~ /no performance report found for gfx/) return false

    // Node/agent died mid-run, or docker/OOM on this node; these surface in the exception.
    def nodeSignals = [
        'seems to be removed or offline',
        'agentofflineexception',
        'issue with creating launcher for agent',
        'closedchannelexception',
        'requestabortedexception',
        'broken pipe',
        'script returned exit code -1',
        'script returned exit code -2',
        'failed to run image',
        'outofmemoryerror',
        'ninja exited with error code 137',
        'maximum checkout retry attempts reached',
        'error cloning remote repo',
        'error fetching remote repo',
    ]
    if (nodeSignals.any { t.contains(it) }) return true
    return scmUtils.isRetriableScmCheckoutError(t)
}

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
                    echo 'Cleaning up old Docker images...'
                    def pruneStatus = sh(script: 'docker image prune -af --filter "until=720h"', returnStatus: true)
                    if (pruneStatus != 0) {
                        echo "[withHealthyNode] WARNING: Docker image prune failed with exit code ${pruneStatus}. Continuing health check."
                    }
                    healthChecks()
                    scmUtils.gitHealthCheck()
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
            // Per-row console log: shStrict mirrors output here so we can classify transient
            // failures (e.g. GPU hang) that only appear in stdout, not in the thrown exception.
            String rowLog = "${env.WORKSPACE}/.rekick-row.log"
            try {
                withEnv(["REKICK_ROW_LOG=${rowLog}"]) {
                    body()
                }
                // If body succeeds, we're done with the loop
                done = true
            } catch (Exception err) {
                String rowText = ''
                try {
                    if (fileExists(rowLog)) {
                        rowText = readFile(rowLog)
                        if (rowText.length() > FAILURE_LOG_TAIL_CHARS) {
                            rowText = rowText.substring(rowText.length() - FAILURE_LOG_TAIL_CHARS)
                        }
                    }
                } catch (Exception ignored) { }

                if (isPerServerTransient("${err}\n${rowText}")) {
                    // Group-1 transient on this node: blacklist it and retry the same arch on a
                    // fresh node. The while loop continues (done still false); if attempts run out
                    // this becomes "no healthy node found", which the post-block re-kicks whole-job.
                    echo "[withHealthyNode] Per-server transient on ${env.NODE_NAME}. Blacklisting the node and retrying.."
                    echo "[withHealthyNode] Error was: ${err}"
                    blacklist << env.NODE_NAME
                    return
                }
                // Real failure (or a whole-job transient like no-healthy-node): fail immediately.
                echo "[withHealthyNode] Execution failed with a non-recoverable error on ${env.NODE_NAME}"
                echo "[withHealthyNode] Error was: ${err}"
                throw err
            } finally {
                // Clean here (moved out of the matrix bodies) so the per-row log above survives
                // until it has been classified. Never let cleanup mask the body's exception.
                try {
                    cleanWs()
                } catch (Exception cleanErr) {
                    echo "[withHealthyNode] cleanWs failed: ${cleanErr}"
                }
            }
        }
    }

    if (!done) {
        // In-stage breadcrumb: the post block reads the log before the final "error" line is printed.
        echo "[withHealthyNode] TRANSIENT: no healthy node for '${baseLabel}' after ${maxAttempts} attempts"
        error "No healthy node found for '${baseLabel}' after ${maxAttempts} attempts"
    }
}

return this
