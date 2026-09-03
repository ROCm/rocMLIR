// SCM helpers: git health check and a robust checkout with fallback to a deep clone.
// Loaded by Jenkinsfile's Bootstrap stage; consumed as scmUtils.<method>().
// ON CHANGING THESE, ALSO CHANGE Jenkinsfile.downstream

import com.cloudbees.groovy.cps.NonCPS
import groovy.transform.Field
import hudson.plugins.git.extensions.impl.CheckoutOption
import hudson.plugins.git.extensions.impl.CloneOption

// Jenkins Git plugin defaults to 10 minutes per command. Use 2h for fetches and
// checkouts that can exceed that on a slow network.
@Field
final int GIT_SCM_TIMEOUT_MINUTES = 120

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

@NonCPS
Map scmWithGitTimeout(Object baseScm) {
    List extensions = []
    boolean hasCloneOption = false
    boolean hasCheckoutOption = false

    (baseScm.extensions ?: []).each { ext ->
        if (ext instanceof CloneOption) {
            extensions << [
                $class: 'CloneOption',
                depth: ext.depth ?: 0,
                shallow: ext.shallow ?: false,
                noTags: ext.noTags ?: false,
                reference: ext.reference ?: '',
                honorRefspec: ext.honorRefspec ?: false,
                timeout: gitTimeoutAtLeast(ext.timeout)
            ]
            hasCloneOption = true
        } else if (ext instanceof CheckoutOption) {
            extensions << [$class: 'CheckoutOption', timeout: gitTimeoutAtLeast(ext.timeout)]
            hasCheckoutOption = true
        } else {
            extensions << ext
        }
    }

    if (!hasCloneOption) {
        extensions << [$class: 'CloneOption', timeout: GIT_SCM_TIMEOUT_MINUTES]
    }
    if (!hasCheckoutOption) {
        extensions << [$class: 'CheckoutOption', timeout: GIT_SCM_TIMEOUT_MINUTES]
    }

    Map checkoutScm = [
        $class: 'GitSCM',
        branches: baseScm.branches,
        doGenerateSubmoduleConfigurations: baseScm.doGenerateSubmoduleConfigurations ?: false,
        extensions: extensions,
        submoduleCfg: baseScm.submoduleCfg ?: [],
        userRemoteConfigs: baseScm.userRemoteConfigs
    ]
    if (baseScm.gitTool) {
        checkoutScm.gitTool = baseScm.gitTool
    }
    if (baseScm.browser) {
        checkoutScm.browser = baseScm.browser
    }
    return checkoutScm
}

@NonCPS
int gitTimeoutAtLeast(Integer timeout) {
    int currentTimeout = timeout ?: 0
    return Math.max(currentTimeout, GIT_SCM_TIMEOUT_MINUTES)
}

String scmCheckoutRetryContext(Object err) {
    String msg = "${err}".toLowerCase()
    try {
        def logLines = currentBuild?.rawBuild?.getLog(500) ?: []
        msg = msg + '\n' + logLines.join('\n').toLowerCase()
    } catch (ignored) {
        // Fall back to the exception text; retry classification should not mask the checkout failure.
    }
    return msg
}

boolean isRetriableScmCheckoutError(String msg) {
    return [
        "connection reset by peer",
        "curl 18",
        "transfer closed with outstanding read data remaining",
        "bytes of body are still expected",
        "unexpected disconnect while reading sideband packet",
        "bad pack header",
        "git-remote-https died of signal 15",
        "early eof",
        "invalid index-pack output"
    ].any { msg.contains(it) }
}

// Retry checkout without shallow clone if GitSCM chokes on a specific SHA
void robustScmCheckout() {
    int maxAttempts = 2
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            // This inner 'try' handles the "reference is not a tree" fallback
            try {
                echo "[SCM] Attempting checkout (${attempt}/${maxAttempts})..."
                checkout(scmWithGitTimeout(scm))
                echo "[SCM] Checkout successful"
                // If checkout succeeds, exit the function immediately
                return
            } catch (err) {
                def msg = "${err}".toLowerCase()
                if (!msg.contains("reference is not a tree") && !msg.contains("could not checkout")) {
                    // If it's not a known transient error, re-throw it to be caught by the outer block
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
                        [$class: 'CloneOption', depth: 0, shallow: false, noTags: false, honorRefspec: true, timeout: GIT_SCM_TIMEOUT_MINUTES],
                        [$class: 'CheckoutOption', timeout: GIT_SCM_TIMEOUT_MINUTES]
                    ]
                ]
                checkout(deepScm)
                echo "[SCM] Deep clone checkout successful."
                // If the deep clone succeeds, exit the function
                return
            }
        } catch (err) {
            // This outer 'catch' block is specifically for retrying network errors
            def msg = scmCheckoutRetryContext(err)
            if (isRetriableScmCheckoutError(msg) && attempt < maxAttempts) {
                echo "[SCM] Attempt ${attempt}/${maxAttempts} failed due to a transient git fetch error."
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

Map externalGitScm(String url, String branch) {
    return [
        $class: 'GitSCM',
        branches: [[name: "*/${branch}"]],
        doGenerateSubmoduleConfigurations: false,
        extensions: [
            [
                $class: 'CloneOption',
                depth: 0,
                shallow: false,
                noTags: false,
                reference: '',
                honorRefspec: false,
                timeout: GIT_SCM_TIMEOUT_MINUTES
            ],
            [$class: 'CheckoutOption', timeout: GIT_SCM_TIMEOUT_MINUTES]
        ],
        submoduleCfg: [],
        userRemoteConfigs: [[url: url]]
    ]
}

void robustExternalCheckout(String url, String branch) {
    int maxAttempts = 2
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            // Discard a partial pack before retrying the clone.
            deleteDir()
            checkout(
                changelog: false,
                poll: false,
                scm: externalGitScm(url, branch)
            )
            return
        } catch (err) {
            String context = scmCheckoutRetryContext(err)
            if (attempt == maxAttempts || !isRetriableScmCheckoutError(context)) {
                throw err
            }

            echo "[SCM] External checkout attempt ${attempt}/${maxAttempts} failed due to a transient git fetch error."
            echo "[SCM] Waiting 2 minutes before retrying..."
            sleep(time: 2, unit: 'MINUTES')
        }
    }
}

return this
