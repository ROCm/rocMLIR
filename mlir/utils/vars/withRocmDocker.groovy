 /* Usage examples
 *
 *   // just PATH + HOME
 *   withRocmDocker {
 *       rocmlir.build_fixedE2ETests(CODEPATH)
 *   }
 *
 *   // add coverage/profiling vars
 *   withRocmDocker(
 *       "LLVM_PROFILE_FILE=${env.WORKSPACE}/build/%m-%p.profraw",
 *       "LLVM_PROFDATA=/opt/rocm/llvm/bin/llvm-profdata",
 *       "LLVM_COV=/opt/rocm/llvm/bin/llvm-cov"
 *   ) {
 *       rocmlir.collectCoverageData(LLVM_PROFDATA, LLVM_COV, CODEPATH)
 *   }
 */

// Used to reduce scripted blocks count in pipelines that use docker containers inside scripted blocks
def call(String... extraEnv, Closure body) {
    script {
        // Get docker run arguments
        env.DOCKER_ARGS = env.DOCKER_ARGS ?: rocmlir.dockerArgs()

        def baseEnv = [
            "PATH=/opt/rocm/llvm/bin:${env.PATH}",
            "HOME=${env.WORKSPACE}"
        ]
        def fullEnv = baseEnv + extraEnv
        def img = docker.image(rocmlir.dockerImage())
        def pulled = img?.pull()
        img.inside(env.DOCKER_ARGS) {
            withEnv(fullEnv) {
                body()
            }
        }
    }
}
