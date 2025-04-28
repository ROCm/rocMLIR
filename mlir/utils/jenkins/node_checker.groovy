/**
 * Custom exception for node check failures.
 */
class NodeCheckException extends RuntimeException {
    NodeCheckException(String message) {
        super(message)
    }
    static NodeCheckException newInstance(String message) {
        return new NodeCheckException(message)
    }
}

/**
 * Runs basic health checks on the current node.
 * Throws NodeCheckException if any check fails.
 */
def checkNodeHealth() {
    echo "Running basic health checks on node ${env.NODE_NAME}..."
    def checks = [
        { sh(script: "docker version", returnStatus: true) == 0 } : "Docker check failed",
        { sh(script: "ls /dev/kfd", returnStatus: true) == 0 }    : "/dev/kfd not found",
        { sh(script: "ls /dev/dri", returnStatus: true) == 0 }    : "/dev/dri not found",
        // Optional: Only run rocminfo/rocm-smi if it's not just a build-only node
        { env.NODE_LABELS == null || !env.NODE_LABELS.contains('build-only') ? (sh(script: "rocminfo", returnStatus: true) == 0) : true } : "rocminfo command failed",
        { env.NODE_LABELS == null || !env.NODE_LABELS.contains('build-only') ? (sh(script: "rocm-smi", returnStatus: true) == 0) : true } : "rocm-smi command failed",
        // Optional: Basic GPU presence check (can be refined if needed)
        { env.NODE_LABELS == null || !env.NODE_LABELS.contains('build-only') ? (sh(script: "lspci | grep -e 'controller' -e 'accelerator' | grep 'AMD/ATI' | wc -l", returnStdout: true).trim().toInteger() > 0) : true } : "No AMD GPUs found via lspci"
    ]

    checks.each { checkClosure, errorMessage ->
        try {
            timeout(time: 1, unit: 'MINUTES') { // Short timeout for each check
                if (!checkClosure()) {
                    throw NodeCheckException.newInstance(errorMessage)
                }
            }
        } catch (Exception e) {
            // Catch timeout or other execution errors
             if (e instanceof NodeCheckException) {
                throw e // Re-throw our specific exception
             }
             // Throw failure for unexpected errors during the check itself
             throw NodeCheckException.newInstance("${errorMessage} (Execution Error: ${e.getMessage()})")
        }
    }
    echo "Node ${env.NODE_NAME} passed basic health checks."
}

/**
 * Tries to acquire a node matching the label, checks its health,
 * and executes the provided code block on it. Retries on failure.
 *
 * @param label The Jenkins label expression for desired nodes.
 * @param retries The number of times to retry on different nodes if checks fail.
 * @param code The closure to execute on a healthy node.
 */
def acquireHealthyNode(String label, int retries = 3, Closure code) {
    String currentLabel = label
    List triedNodes = []

    for (int attempt = 0; attempt <= retries; attempt++) {
        def healthyNodeName = null
        def nodeToTry = null
        def checkException = null

        try {
            // Find a node matching the current label that hasn't been tried yet
            List potentialNodes = nodesByLabel(label: currentLabel, offline: false)
            nodeToTry = potentialNodes.find { !triedNodes.contains(it) }

            if (!nodeToTry) {
                echo "No suitable *new* nodes found matching label: ${currentLabel}. Tried: ${triedNodes}"
                // If it's the last attempt, throw error, otherwise loop will end/retry logic below handles it.
                if (attempt == retries) {
                     throw new RuntimeException("Failed to find any available and untested node after ${attempt} attempts.")
                }
                 // Wait a bit before checking nodesByLabel again on the next attempt
                sleep(time: 1, unit: 'MINUTES')
                continue // Go to next attempt iteration
            }

            echo "Attempt ${attempt + 1}/${retries + 1}: Trying node ${nodeToTry} with label ${currentLabel}"
            triedNodes.add(nodeToTry)

            // Allocate the node
            node(nodeToTry) {
                // Run the health check *on the allocated node*
                checkNodeHealth() // Throws NodeCheckException on failure

                // If check passes, store name and break inner try to execute code
                healthyNodeName = env.NODE_NAME
                 echo "Node ${healthyNodeName} is healthy. Proceeding with execution."
            } // Node block ends here

            // If we get here without exception, the node is healthy
            // Now, re-allocate the *same* healthy node to run the user's code
            node(healthyNodeName) {
                 ws { // Ensure we are in a workspace
                    code() // Execute the main logic passed to this function
                 }
            }
            // If code() completes successfully, we're done.
            return

        } catch (NodeCheckException e) {
            checkException = e
            echo "Node check failed on ${nodeToTry}: ${e.getMessage()}"
        } catch (Exception e) { // Catch Jenkins agent allocation errors etc.
            checkException = e // Treat allocation errors like check failures
            echo "Failed to allocate or run checks on node ${nodeToTry}: ${e.getMessage()}"
        }

        // If we are here, either check failed or node allocation failed
        if (nodeToTry && checkException) {
             // Add failure message to build description
             def message = "Skipped node ${nodeToTry} due to: ${checkException.getMessage()}"
             currentBuild.description = (currentBuild.description ?: "") + "<b style='color: #7d6608'>Node Check Failed:</b> <span style='color: #b7950b'>${message}</span><br/>"

             // Exclude this node from future attempts in this run
             currentLabel = "${label} && !${nodeToTry}" // Build exclusion label
             // Add all tried nodes to exclusion label for safety
             triedNodes.each { tried ->
                 if (currentLabel.indexOf("!${tried}") == -1) { // Avoid duplicates
                     currentLabel += " && !${tried}"
                 }
             }
             echo "Updated label for next attempt: ${currentLabel}"
        }

        // If this was the last retry attempt and it failed
        if (attempt == retries) {
             error("Failed to acquire a healthy node matching label [${label}] after ${retries + 1} attempts. Last error on ${nodeToTry}: ${checkException?.getMessage()}")
        }
        // If not the last attempt, loop continues with the updated exclusion label
         echo "Retrying..."

    }
}

// Export the functions so they can be called from the Jenkinsfile
return this