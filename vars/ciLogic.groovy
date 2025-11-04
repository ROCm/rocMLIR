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
        if (params.weekly || params.nightly) {
            label = 'mlir && linux-mi355-8'
        } else {
            label = 'mlir && linux-mi355-1'
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
            if (params.nightly || params.weekly) {
                return "mlir && linux-mi355-8"
            } else {
                return "mlir && linux-mi355-1"
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
