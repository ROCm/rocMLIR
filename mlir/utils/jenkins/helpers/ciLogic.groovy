// CI flow helpers: heartbeat, build resets, label resolution, codepath/chip
// gating, config-file splitting, build-failure classification, and the
// Teams notification card.
// Loaded by Jenkinsfile's Bootstrap stage; consumed as ciLogic.<method>().
// ON CHANGING THESE, ALSO CHANGE Jenkinsfile.downstream

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
        label = 'mlir && (gfx942 || gfx908 || gfx90a)'
    } else if (codepath == "gfx950") {
        if (params.weekly) {
            label = 'mlir && linux-mi350-8'
        } else {
            label = 'mlir && linux-mi350-1'
        }
    } else if (codepath == "navi21") {
        // For non-performance related testing, use both workstations (gfx1030w)
        // and server nodes (gfx1030)
        label = 'mlir && ( gfx1030w || gfx1030 )'
    } else if (codepath == "vanilla"){
        label = 'mlir'
    } else if (codepath == "navi3x") {
        if (params.nightly || params.weekly) {
            label = 'mlir && gfx1100'
        } else {
            label = 'mlir && ( gfx1100 || gfx1101 )'
        }
    } else if (codepath == "navi4x") {
        if (params.nightly || params.weekly) {
            label = 'mlir && gfx1201'
        } else {
            label = 'mlir && ( gfx1200 || gfx1201 )'
        }
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
            if (params.weekly) {
                return "mlir && linux-mi350-8"
            } else {
                return "mlir && linux-mi350-1"
            }
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

boolean shouldRunFromCodepath(String codepath) {
    // Run vanilla on public CI
    if ((codepath == "vanilla") && (params.canXdlops == false)) {
        return true
    }
    // Run mfma on private CI
    if ((codepath == "mfma") && params.canXdlops) {
        return true
    }
    if (codepath == "gfx950" && params.canXdlops && params.disable950 == false) {
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
            return params.disable950 == false && shouldRunFromCodepath("gfx950")
        case "gfx1030":
            return shouldRunFromCodepath("navi21")
        case "gfx1100":
            return shouldRunFromCodepath("navi3x")
        case "gfx1200":
        case "gfx1201":
            return shouldRunFromCodepath("navi4x")
    }
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
        if (params.codepath == "gfx950" && params.disable950 == false) return true
        if (params.codepath == "navi21" && params.disableNavi21 == false) return true
        if (params.codepath == "navi3x" && params.disableNavi3x == false) return true
        if (params.codepath == "navi4x" && params.disableNavi4x == false) return true
        return false
    }
}

boolean isNotNavi3x(String chip) {
    return "${chip}" != 'gfx1100' && "${chip}" != 'gfx1101'
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

// Classifies build failure from console log. Returns [reason:, codepath:, stage:] (empty string = not found).
// Add new scenarios here by matching log patterns (order = first match wins).
Map<String,String> classifyBuildFailure(String logText) {
    def reason = ''
    def codepath = ''
    def stage = ''
    def failureList = ''
    def failureListLabel = ''
    def failedTestsSnippet = ''
    if (!logText) return [reason: reason, codepath: codepath, stage: stage, failureList: failureList, failureListLabel: failureListLabel, failedTestsSnippet: failedTestsSnippet]

    // Scenario 1: Tuning failed - errors detected in tuning log (Tune rocMLIR)
    if (!reason && logText.contains('Tuning failed: Detected errors in tuning log')) {
        reason = 'Tune rocMLIR: errors in tuning log (check logs for details)'
    }

    // Scenario 2: SCM checkout failed (max retries, clone error, or channel error)
    if (!reason && (logText.contains('ERROR: Checkout failed') || logText.contains('Maximum checkout retry attempts reached') || logText.contains("ERROR: Error cloning remote repo"))) {
        reason = 'SCM checkout failed (max retries or agent/channel error)'
    }

    // Scenario 3: Parameter sweeps - failing configurations discovered.
    // 3a: Conv/perf sweeps (parameterSweeps.py) use "*** Summary of failures ***".
    if (!reason && logText.contains('*** Summary of failures ***')) {
        reason = 'Parameter sweeps: failing configurations discovered'
        def summaryStart = logText.indexOf('*** Summary of failures ***')
        def summaryEnd = logText.indexOf('Passed:', summaryStart)
        if (summaryEnd < 0) summaryEnd = logText.indexOf('script returned exit code', summaryStart)
        if (summaryEnd < 0) summaryEnd = logText.length()
        failureList = logText.substring(summaryStart, summaryEnd).trim()
        if (failureList.length() > 2000) failureList = failureList.substring(0, 2000) + '\n... (truncated)'
    }
    // 3b: Attention sweeps (attentionSweeps.py) use "Failing Configurations".
    if (!reason && logText.contains('Failing Configurations')) {
        reason = 'Attention parameter sweeps: failing configurations discovered'
        def headerPos = logText.lastIndexOf('Failing Configurations')
        def configStart = logText.indexOf('\n', headerPos)
        configStart = (configStart >= 0) ? configStart + 1 : headerPos
        def summaryEnd = logText.indexOf('Passed:', configStart)
        if (summaryEnd < 0) summaryEnd = logText.indexOf('script returned exit code', configStart)
        if (summaryEnd < 0) summaryEnd = Math.min(configStart + 3000, logText.length())
        def snippet = logText.substring(configStart, summaryEnd).trim()
        snippet = snippet.replaceAll(/\[\d{4}-\d{2}-\d{2}T[\d:.]+Z\]\s*/, '')
        // Append the Passed/Invalid/Failed summary line if present.
        def statsLineEnd = logText.indexOf('\n', summaryEnd)
        if (statsLineEnd < 0) statsLineEnd = logText.length()
        def statsLine = logText.substring(summaryEnd, statsLineEnd).trim()
            .replaceAll(/\[\d{4}-\d{2}-\d{2}T[\d:.]+Z\]\s*/, '')
        if (statsLine) snippet = snippet + '\n\n' + statsLine
        if (snippet.length() > 2000) snippet = snippet.substring(0, 2000) + '\n... (truncated)'
        failureList = snippet
    }

    // Scenario 4: HIP no device (hipErrorNoDevice)
    if (!reason && logText.contains('RuntimeError: hipError_t.hipErrorNoDevice')) {
        reason = 'HIP: no device (hipErrorNoDevice)'
    }

    // Scenario 5: One or more tests failed (Failed Tests (N): ...)
    if (!reason && logText.contains('Failed Tests (')) {
        reason = 'One or more tests failed'
        def failedStart = logText.indexOf('Failed Tests (')
        def failedEnd = logText.indexOf('Testing Time:', failedStart)
        if (failedEnd < 0) failedEnd = logText.indexOf('Total Discovered Tests:', failedStart)
        if (failedEnd < 0) failedEnd = Math.min(failedStart + 2000, logText.length())
        failedTestsSnippet = logText.substring(failedStart, failedEnd).trim()
        if (failedTestsSnippet.length() > 2000) failedTestsSnippet = failedTestsSnippet.substring(0, 2000) + '\n... (truncated)'
    }

    // Scenario 6: MIGraphX CMake configuration failed.
    // Match by context around "Configuring incomplete" (MIGraphX path or composable_kernel_host) so we don't rely on stage order in interleaved logs.
    def cmakeConfigErrorPos = logText.lastIndexOf('Configuring incomplete, errors occurred!')
    if (!reason && cmakeConfigErrorPos >= 0) {
        def ctxStart = Math.max(0, cmakeConfigErrorPos - 4000)
        def ctxAround = logText.substring(ctxStart, Math.min(logText.length(), cmakeConfigErrorPos + 500))
        if (ctxAround.contains('MIGraphX') || ctxAround.contains('composable_kernel_host') || ctxAround.contains('Findcomposable_kernel_host')) {
            reason = 'MIGraphX: CMake configuration failed'
            // Extract the last "CMake Error" block before "Configuring incomplete" as a snippet.
            def cmakeErrorStart = logText.lastIndexOf('CMake Error', cmakeConfigErrorPos)
            if (cmakeErrorStart >= 0) {
                def snippet = logText.substring(cmakeErrorStart, cmakeConfigErrorPos).trim()
                snippet = snippet.replaceAll(/\[\d{4}-\d{2}-\d{2}T[\d:.]+Z\]\s*/, '')
                if (snippet.length() > 2000) snippet = snippet.substring(0, 2000) + '\n... (truncated)'
                failureList = snippet
                failureListLabel = 'CMake error:'
            }
        }
    }

    // Scenario 7: Agent flapping (node repeatedly offline/online).
    // Checked last: agent disconnect messages often appear as a side effect of pod termination after a real build error.
    if (!reason) {
        def flappingMatch = logText =~ /(\S+)\s+seems to be removed or offline.*will wait for.*come back online/
        if (flappingMatch.find()) {
            reason = "Agent flapping: ${flappingMatch.group(1)} went offline/online repeatedly"
        }
    }

    if (!reason) reason = 'Could not match a known error pattern. See build log for details.'

    // Failure anchor: position in log where this failure was detected (used to extract stage/CODEPATH from the failing branch, not from later branches).
    def failureAnchor = -1

    // Prefer detecting the anchor directly from log patterns instead of the human-facing reason text.
    def scmAnchor = Math.max(logText.lastIndexOf('Maximum checkout retry attempts reached'),
                             logText.lastIndexOf('[SCM] Checkout failed on'))
    if (scmAnchor < 0) scmAnchor = logText.lastIndexOf("ERROR: Error cloning remote repo")
    if (scmAnchor < 0) scmAnchor = logText.lastIndexOf('ERROR: Checkout failed')

    if (scmAnchor >= 0) {
        failureAnchor = scmAnchor
    } else {
        def tuneAnchor = logText.lastIndexOf('Tuning failed: Detected errors in tuning log')
        if (tuneAnchor >= 0) {
            failureAnchor = tuneAnchor
        } else {
            def sweepsAnchor = logText.indexOf('*** Summary of failures ***')
            if (sweepsAnchor < 0) sweepsAnchor = logText.indexOf('Failing Configurations')
            if (sweepsAnchor >= 0) {
                failureAnchor = sweepsAnchor
            } else {
                def hipNoDeviceAnchor = logText.lastIndexOf('hipErrorNoDevice')
                if (hipNoDeviceAnchor >= 0) {
                    failureAnchor = hipNoDeviceAnchor
                } else {
                    def testsFailedAnchor = logText.lastIndexOf('Failed Tests (')
                    if (testsFailedAnchor >= 0) {
                        failureAnchor = testsFailedAnchor
                    } else {
                        def migraphxAnchor = logText.lastIndexOf('Configuring incomplete, errors occurred!')
                        if (migraphxAnchor >= 0) {
                            failureAnchor = migraphxAnchor
                        } else {
                            def agentFlappingAnchor = logText.lastIndexOf('seems to be removed or offline')
                            if (agentFlappingAnchor >= 0) {
                                failureAnchor = agentFlappingAnchor
                            }
                        }
                    }
                }
            }
        }
    }

    def searchStart = (failureAnchor >= 0) ? Math.max(0, failureAnchor - 8000) : 0
    def searchEnd = (failureAnchor >= 0) ? Math.min(logText.length(), failureAnchor + 500) : logText.length()
    def contextWindow = (failureAnchor >= 0) ? logText.substring(searchStart, searchEnd) : logText
    def logBeforeAnchor = (failureAnchor > 0) ? logText.substring(0, failureAnchor) : ''

    // CODEPATH: prefer "Failed in branch Matrix - CODEPATH = 'X'" near the failure; else any CODEPATH in context window; else global.
    def branchMatch = contextWindow =~ /Failed in branch Matrix - CODEPATH = ['"](\w+)['"]/
    if (branchMatch.find()) {
        codepath = branchMatch.group(1)
    } else {
        def cpMatch = contextWindow =~ /CODEPATH\s*=\s*['"]?(\w+)['"]?|Running\s+(\w+)\s+on\s+\S+/
        if (cpMatch.find()) codepath = cpMatch[0][1] ?: cpMatch[0][2] ?: ''
    }
    if (!codepath) {
        def cpMatch = logText =~ /CODEPATH\s*=\s*['"]?(\w+)['"]?|Running\s+(\w+)\s+on\s+\S+/
        if (cpMatch.find()) codepath = cpMatch[0][1] ?: cpMatch[0][2] ?: ''
    }

    // Stage: last stage name that appears *before* the failure anchor (so we report the stage that was running when it failed).
    def stageNames = ['SCM Checkout', 'Build and Test', 'Parameter sweeps', 'Parameter Sweep', 'Tune MLIR kernels', 'Tune rocMLIR', 'Code coverage', 'Archive performance DB', 'MIGraphX', 'Build and Verify MIGraphX with MLIR']
    def stageSearchText = (logBeforeAnchor.length() > 0) ? logBeforeAnchor : logText
    def stageIdx = -1
    for (def name in stageNames) {
        def idx = stageSearchText.lastIndexOf(name)
        if (idx >= 0 && idx > stageIdx) { stage = name; stageIdx = idx }
    }

    return [reason: reason, codepath: codepath, stage: stage, failureList: failureList, failureListLabel: failureListLabel, failedTestsSnippet: failedTestsSnippet]
}

// Parse "Aborted by USERNAME" from console log (Jenkins writes this when a user aborts the build).
def parseAbortedByFromLog(String logText) {
    if (!logText) return ''
    def m = logText =~ /Aborted by ([^\r\n]+)/
    return m.find() ? m.group(1).trim() : ''
}

// Sends a Teams adaptive card for build result (webhook URL from Jenkins credential 'CI_MONITORING_TEAMS').
// statusMessage: full phrase e.g. "Build 42 completed successfully". color: Adaptive Card color ("good"=green, "warning"=yellow, "attention"=red).
// runType: "nightly" or "weekly" (subtitle line). blueOceanUrl: Blue Ocean pipeline URL. jobUrl: classic Jenkins job/build URL.
// failureDetails: optional Map [reason:, codepath:, stage:, failureList:] — when set, adds Stage/CODEPATH/Details and optionally a code block for failureList.
void sendTeamsBuildNotification(String buildNumber, String statusMessage, String color, String runType, String blueOceanUrl, String jobUrl, Map failureDetails = null) {
    try {
        def subtitle = (runType == 'nightly') ? 'MLIR Nightly 🌙' : 'MLIR Weekly 📅'
        def timestamp = new Date().format('yyyy-MM-dd HH:mm z')
        def escapeJson = { String s -> (s ?: '').replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ') }
        def escapeJsonMultiline = { String s -> (s ?: '').replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '') }
        def detailBlocks = ''
        if (failureDetails) {
            def abortedByBlock = ''
            if (failureDetails.abortedBy != null) {
                def ab = escapeJson(failureDetails.abortedBy)
                abortedByBlock = ",{\"type\":\"RichTextBlock\",\"inlines\":[{\"type\":\"TextRun\",\"text\":\"Aborted by: \",\"weight\":\"bolder\"},{\"type\":\"TextRun\",\"text\":\"${ab}\"}]}"
            }
            def r = escapeJson(failureDetails.reason ?: '')
            def c = failureDetails.codepath ? escapeJson(failureDetails.codepath) : '—'
            def t = failureDetails.stage ? escapeJson(failureDetails.stage) : '—'
            detailBlocks = "${abortedByBlock},{\"type\":\"RichTextBlock\",\"inlines\":[{\"type\":\"TextRun\",\"text\":\"Stage: \",\"weight\":\"bolder\"},{\"type\":\"TextRun\",\"text\":\"${t}\"}]},{\"type\":\"RichTextBlock\",\"inlines\":[{\"type\":\"TextRun\",\"text\":\"CODEPATH: \",\"weight\":\"bolder\"},{\"type\":\"TextRun\",\"text\":\"${c}\"}]},{\"type\":\"RichTextBlock\",\"inlines\":[{\"type\":\"TextRun\",\"text\":\"Details: \",\"weight\":\"bolder\"},{\"type\":\"TextRun\",\"text\":\"${r}\"}],\"wrap\":true}"
            if (failureDetails.failureList) {
                def flLabel = failureDetails.failureListLabel ?: 'Failing configs:'
                def fl = escapeJsonMultiline(failureDetails.failureList)
                detailBlocks += ",{\"type\":\"RichTextBlock\",\"inlines\":[{\"type\":\"TextRun\",\"text\":\"${escapeJson(flLabel)}\",\"weight\":\"bolder\"}]},{\"type\":\"TextBlock\",\"text\":\"${fl}\",\"wrap\":true,\"fontType\":\"monospace\",\"size\":\"small\",\"separator\":true}"
            }
            if (failureDetails.failedTestsSnippet) {
                def fts = escapeJsonMultiline(failureDetails.failedTestsSnippet)
                detailBlocks += ",{\"type\":\"RichTextBlock\",\"inlines\":[{\"type\":\"TextRun\",\"text\":\"Failed tests:\",\"weight\":\"bolder\"}]},{\"type\":\"TextBlock\",\"text\":\"${fts}\",\"wrap\":true,\"fontType\":\"monospace\",\"size\":\"small\",\"separator\":true}"
            }
        }
        def payload = """
{"attachments":[{"contentType":"application/vnd.microsoft.card.adaptive","content":{"type":"AdaptiveCard","\$schema":"http://adaptivecards.io/schemas/adaptive-card.json","version":"1.4","body":[{"type":"TextBlock","text":"CI Update","weight":"bolder","size":"extraLarge","separator":true},{"type":"TextBlock","text":"${subtitle}"},{"type":"TextBlock","text":"${statusMessage}","color":"${color}"}${detailBlocks},{"type":"TextBlock","text":"Finished: ${timestamp}","size":"small","isSubtle":true}],"actions":[{"type":"Action.OpenUrl","url":"${blueOceanUrl}","title":"Open Blue Ocean 🌊"},{"type":"Action.OpenUrl","url":"${jobUrl}","title":"Open Job 🏗️"}]}}]}
"""
        writeFile file: 'teams-payload.json', text: payload.trim(), encoding: 'UTF-8'
        withCredentials([
            string(credentialsId: 'CI_MONITORING_TEAMS', variable: 'WEBHOOK_URL'),
            string(credentialsId: 'MLIR_CI_CHANNEL', variable: 'WEBHOOK_URL_MLIR')
        ]) {
            ['CI_MONITORING_TEAMS': 'WEBHOOK_URL', 'MLIR_CI_CHANNEL': 'WEBHOOK_URL_MLIR'].each { name, envVar ->
                def resp = sh(script: "curl -s -w '\\n%{http_code}' -X POST \"\$${envVar}\" -H 'Content-Type: application/json; charset=utf-8' -d @teams-payload.json", returnStdout: true).trim()
                def lines = resp.split('\n')
                def code = lines[-1]
                def body = lines.length > 1 ? lines[0..-2].join('\n') : ''
                echo "Teams webhook (${name}) response: HTTP ${code}${body ? ' body=' + body : ''}"
                if (code != '200' && code != '202') {
                    echo "Teams notification (${name}) may have failed (expected 200/202, got ${code})"
                }
            }
        }
    } catch (e) {
        echo "Teams notification skipped or failed: ${e}"
    }
}

return this
