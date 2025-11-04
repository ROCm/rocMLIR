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
