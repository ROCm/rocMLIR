void postProcessPerfRes(String chip) {
    publishHTML (target: [
        allowMissing: false,
        alwaysLinkToLastBuild: false,
        keepAll: true,
        reportDir: 'build/reports',
        reportFiles: "${chip}_MLIR_Performance_Changes.html,${chip}_MLIR_vs_MIOpen.html,${chip}_MLIR_Performance_Changes_Gemm.html,${chip}_MLIR_vs_CK.html,${chip}_conv_fusion.html,${chip}_gemm_fusion.html",
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
    // Save results for future comparison
    archiveArtifacts artifacts: 'build/*_mlir_*.csv,build/perf-run-date', allowEmptyArchive: true, onlyIfSuccessful: true
}

void archivePerfDB() {
    // Note: add additional architectures here
    dir ('build/perfDB') {
        def architectures = params.canXdlops ? ['gfx908', 'gfx90a', 'gfx942', 'gfx950', 'gfx1100', 'gfx1201'] : ['vanilla']
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
