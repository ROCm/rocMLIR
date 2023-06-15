#!/bin/bash

printUsage() {
    echo "./tune_MLIR_kernels.sh [-h] [options]"
    echo "Reproduce MLIR kernel tuning process"
    echo "Otions:"
    echo "  s: build MLIR static library, i.e. librockCompiler"
    echo "  m: checkout MIOpen project and build MIOpen with librockCompiler"
    echo "  -c <\"config\">: Run MIOpenDriver on a single config"
    echo "                 Otherwise, on all configs in resnet50-miopen-configs"
    echo "  -u: turn off tuning (tuning is on be default)"
    echo "  -b: don't run any tuning, just the compilation actions"
    echo "  -d <direction>: choose one direction"
    echo "                  Otherwise, run config(s) with all directions (1,2,4)"
    echo "  -l <layout>: choose one layout"
    echo "               Otherwise, run config(s) with all layouts (NCHW, NHWC)"
    echo "  -t <dtype>: choose one data type"
    echo "              Otherwise, run config(s) with all dtypes (fp16, fp32, int8)"
    echo "  -v: turn on MIOpenDriver logging"
    echo "Examples:"
    echo "./tune_MLIR_kernels.sh: run all configs with all direction, all layouts, and all dtypes with tuning on and logging off"
    echo "./tune_MLIR_kernels.sh -c \"-n 256 -c 64 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1\" -d 1 -t fp16 -l NHWC -v"
    echo "  Run a single config with direction=fwd, dtype=fp16, layout=NHWC with tuning and logging on"
}

ROCMLIR_DIR=$(git rev-parse --show-toplevel || echo "${HOME}/rocMLIR/")
WORKSPACE=${ROCMLIR_DIR}
MIOPEN_DIR=${WORKSPACE}/MIOpen
DRIVER="${MIOPEN_DIR}/build/bin/MIOpenDriver"
CONFIG_POOL=${ROCMLIR_DIR}/mlir/utils/jenkins/miopen-tests/resnet50-miopen-configs
JENKINSFILE=${WORKSPACE}/mlir/utils/jenkins/Jenkinsfile

getMIOpenBranchName() {
    # get MIOpen branch name from the defaultValue of
    # parameter MIOpenBranch
    result=$(grep -e "MIOpenBranch.*defaultValue" ${JENKINSFILE} | head -n 1)
    result=${result#*defaultValue:\ *\'}
    result=${result%%\',*}
    echo "$result"
}

gitCheckoutMIOpen() {
    cd ${ROCMLIR_DIR}
    miopen_branch=$(getMIOpenBranchName)
    if [[ -d ${MIOPEN_DIR} ]]; then
        echo ">>> MIOpen exists, update branch ${miopen_branch}"
        pushd ${MIOPEN_DIR}
        git fetch origin
        git checkout ${miopen_branch}
        git pull
        popd
    else
        echo ">>> git clone MIOpen@${miopen_branch}"
        git clone -b ${miopen_branch} https://github.com/ROCmSoftwarePlatform/MIOpen.git
    fi
}

buildlibrockCompiler() {
    echo ">>> build librockCompiler / MIXR target"
    cd ${WORKSPACE}
    cmake . -G Ninja -B build-static \
          -DBUILD_MIXR_TARGET=ON \
          -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
          -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
          -DCMAKE_BUILD_TYPE=Release # or RelWithDebInfo
    cd build-static
    ninja
    rm -rf ${WORKSPACE}/MIOpenDeps
    cmake --install . --prefix ${WORKSPACE}/MIOpenDeps
}

buildMIOpenWithlibrockCompiler() {
    echo ">>> build MIOpenDriver with librockCompiler"
    cd ${MIOPEN_DIR}
    cmake . -G "Unix Makefiles" -B build -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
          -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
          -DMIOPEN_USE_MLIR=On \
          -DMIOPEN_USE_COMPOSABLEKERNEL=Off \
          -DMIOPEN_BACKEND=HIP \
          -DCMAKE_PREFIX_PATH=${WORKSPACE}/MIOpenDeps \
          "-DCMAKE_CXX_FLAGS=-isystem ${WORKSPACE}/MIOpenDeps/include" \
          -DMIOPEN_USER_DB_PATH=${MIOPEN_DIR}/build/MIOpenUserDB \
          "-DMIOPEN_TEST_FLAGS=--verbose --disable-verification-cache"
    cd build
    make -j $(nproc) MIOpenDriver
}

run_tests_for_a_config() {
    ## $1: config
    for t in ${ALL_DTYPES[@]}; do
        if [[ ${got_dtype} -eq 1 ]] && [[ "$t" != "$DTYPE" ]]; then
            continue
        fi
        for d in ${ALL_DIRECTIONS[@]}; do
            if [[ ${got_direction} -eq 1 ]] && [[ "$d" != "$DIRECTION" ]]; then
                continue
            fi
            if [[ "$t" == "int8" ]] && [[ $d -ne 1 ]]; then
                continue
            fi
            for l in ${ALL_LAYOUTS[@]}; do
                if [[ ${got_layout} -eq 1 ]] && [[ "$l" != "$LAYOUT" ]]; then
                    continue
                fi
                run_a_single_test $t $d $l "$1"
            done
        done
    done
}

run_a_single_test() {
    ## $1: dtype
    ## $2: direction
    ## $3: layout
    ## $4: config
    case "$1" in
        fp16 ) FIXED_CONFIG_ARGS=("convfp16") ;;
        fp32 ) FIXED_CONFIG_ARGS=("conv") ;;
        int8 ) FIXED_CONFIG_ARGS=("convint8") ;;
        * ) echo "Can't happen"; exit 3 ;;
    esac

    FIXED_CONFIG_ARGS+=("--forw" "$2" "--in_layout" "$3"
                        "--out_layout" "$3" "--fil_layout" "$3")

    # Set env variables
    export MIOPEN_FIND_MODE=1
    export MIOPEN_DRIVER_USE_GPU_REFERENCE=1

    if [[ $TUNING == 1 ]]; then
        export MIOPEN_FIND_ENFORCE=4
    fi

    declare -xg MIOPEN_DEBUG_FIND_ONLY_SOLVER
    case "$2" in
        1) MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwd ;;
        2) MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwd ;;
        4) MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrW ;;
        *) echo "$0: Unsupported direction flag $DIRECTION"; exit 2
    esac

    case $(/opt/rocm/bin/rocm_agent_enumerator) in
        *gfx908*|*gfx90a*)           XDLOPS=1 ;;
        *gfx906*|*gfx900*|*gfx1030*) XDLOPS=0 ;;
        *)                           echo "No useful GPU found." ; exit 2 ;;
    esac

    if [[ $XDLOPS == 1 ]]; then
        MIOPEN_DEBUG_FIND_ONLY_SOLVER+="Xdlops"
    fi
    export MIOPEN_DEBUG_FIND_ONLY_SOLVER

    if [[ $VERBOSE -eq 1 ]]; then
        export MIOPEN_ENABLE_LOGGING=1
        export MIOPEN_ENABLE_LOGGING_CMD=1
        export MIOPEN_LOG_LEVEL=6
    fi

    ${DRIVER} ${FIXED_CONFIG_ARGS[@]} $4
}

# b (build only)
# d <direction>
# l <layout>
# t <datatype>
# u: tuning
# m: checkout MIOpen and build miopen with MLIR
# s: build librockCompiler
# c <"config">
# v: verbose
build_only=0
got_direction=0
got_layout=0
got_dtype=0
TUNING=1
checkoutMIOpen=0
buildStaticLib=0
SINGLE_CONFIG=""
VERBOSE=0
while getopts "hd:l:t:umsbc:v" opt; do
    case "$opt" in
        h)
            printUsage
            exit 0
            ;;
        b)
            build_only=1
            ;;
        d)
            got_direction=1
            DIRECTION=$OPTARG
            ;;
        l)
            got_layout=1
            LAYOUT=$OPTARG
            ;;
        t)
            got_dtype=1
            DTYPE=$OPTARG
            ;;
        u)
            TUNING=0
            ;;
        m)
            checkoutMIOpen=1
            ;;
        s)
            buildStaticLib=1
            ;;
        c)
            SINGLE_CONFIG=$OPTARG
            ;;
        v)
            VERBOSE=1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
        \?)
            printUsage
            exit 1
            ;;
    esac
done

# Step 1: build librockCompiler if needed
if [[ $buildStaticLib -eq 1 ]]; then
    buildlibrockCompiler
fi

# Step 2: checkout MIOpen if needed
if [[ $checkoutMIOpen -eq 1 ]]; then
    gitCheckoutMIOpen
    buildMIOpenWithlibrockCompiler
fi

declare -a ALL_DTYPES=(fp16 fp32 int8)
declare -a ALL_DIRECTIONS=(1 2 4)
declare -a ALL_LAYOUTS=(NCHW NHWC)

# Step 3: Run MIOpenDriver with MLIR solver
if [[ $build_only -eq 0 ]]; then
# Step 3.1: Clean out stale tuning databases
    if [[ -d ${MIOPEN_DIR}/build/MIOpenUserDB ]]; then
        rm -rf ${MIOPEN_DIR}/build/MIOpenUserDB
    fi
    if [[ -d ${HOME}/.cache/miopen ]]; then
        rm -rf ${HOME}/.cache/miopen
    fi


    if [[ "${SINGLE_CONFIG}" == "" ]]; then
        while read -r config
        do
            run_tests_for_a_config "${config}"
        done < ${CONFIG_POOL}
    else
        run_tests_for_a_config "${SINGLE_CONFIG}"
    fi
fi

