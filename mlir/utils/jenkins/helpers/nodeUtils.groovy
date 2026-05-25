// Node lifecycle helpers: GPU reset, health checks, Docker discovery,
// docker image names, and the withHealthyNode() retry harness.
// Loaded by Jenkinsfile's Bootstrap stage; consumed as nodeUtils.<method>().
// ON CHANGING THESE, ALSO CHANGE Jenkinsfile.downstream

import groovy.transform.Field
import java.util.concurrent.ConcurrentHashMap
import org.jenkinsci.plugins.workflow.support.steps.AgentOfflineException

// ConcurrentHashMap helps when we need to write variables in parallel
// one instance for the whole run
@Field
ConcurrentHashMap<String,String> DOCKER_ARGS_BY_NODE = new ConcurrentHashMap<>()

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

return this
