def call(script) {
    def codepaths = ['vanilla', 'mfma', 'navi21', 'navi3x', 'navi4x']

    parallel codepaths.collectEntries { cp ->

        [(cp): {

            if (!rocmlir.shouldRunBuildAndTest(cp)) {
                echo "Skipping $cp branch - filtered by params/labels"
                return
            }

            node(rocmlir.getLabelFromCodepath(cp)) {
                try {
                    stage("$cp  Prepare node") {
                        retry(3) {
                            rocmlir.resetGPUs()
                            rocmlir.checkNodeHealth()
                            env.DOCKER_ARGS = rocmlir.dockerArgs()
                        }
                        echo "DOCKER_ARGS = ${env.DOCKER_ARGS}"
                    }

                    stage("$cp  Fixed E2E (shared-lib)") {
                        if (params.sharedLib) {
                            runFixedE2E(cp)
                        }
                    }

                    stage("$cp  Random E2E (shared-lib)") {
                        if (params.sharedLib && params.nightly) {
                            runRandomE2E(cp)
                        }
                    }

                    stage("$cp  Tune selected cfgs") {
                        if (params.sharedLib && !params.nightly) {
                            tuneSelectedConfigs(cp)
                        }
                    }

                    stage("$cp  Static lib") {
                        if (params.staticLib && !params.nightly) {
                            buildStaticPackage(cp)
                        }
                    }

                } finally {
                    if (currentBuild.currentResult == 'FAILURE') {
                        rebootNode()
                    }
                }
            }
        }]
    }
}
