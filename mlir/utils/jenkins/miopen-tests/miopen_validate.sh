#!/bin/bash

declare -a FIXED_CONFIG_ARGS
declare -g TMPFILE
declare -ig XDLOPS=-1
declare -ig TUNING=0
declare -ig TESTALL=0
declare -g DTYPE=""
declare -g DIRECTION=""
declare -g LAYOUT=""
declare -g DRIVER="./bin/MIOpenDriver"

declare -a ALL_DTYPES=(fp16 fp32)
declare -a ALL_DIRECTIONS=(1 2 4)
declare -a ALL_LAYOUTS=(NCHW NHWC)
declare -a ALL_CONFIGS=()

function usage() {
    cat <<END
$0: [-d | --direction] DIR [-t | --dtype] [fp16 | fp32] [-l | --layout] LAYOUT
[-x | --xdlops] [-X | --no-xdlops (default)] [--tuning | --no-tuning (default)]
[--driver DRIVER (default bin/MIOpenDriver)] [--test-all]

DIR is either 1 (forward (fwd)) 2 (backward data (bwd)), or
4 (backward weights (wrw)), other values are currently unsupported
for testing.

LAYOUT is a four-letter string containing the letters N, C, H, and W
that specifies the memory layout to test.

Configs (lists of arguments to the driver) should be sent on
standard input.
END
    exit 2
}

function parse_options() {
    local -i got_layout=0
    local -i got_dtype=0
    local -i got_direction=0

    local parsed_args
    parsed_args=$(getopt -n "$0" -o d:t:l:xXh \
                         --long direction:,dtype:,layout:,xdlops,no-xdlops,driver:,driver:,tuning,no-tuning,test-all,help -- "$@")
    local -i valid_args=$?
    if [[ $valid_args -ne 0 ]]; then
        usage
    fi
    eval set -- "$parsed_args"
    while :
    do
        case "$1" in
            -h | --help ) usage ;;
            -x | --xdlops ) XDLOPS=1; shift; ;;
            -X | --no-xdlops ) XDLOPS=0; shift ;;
            --tuning ) TUNING=1; shift; ;;
            --no-tuning ) TUNING=0; shift; ;;
            --test-all ) TESTALL=1; shift; ;;
            -d | --direction ) got_direction=1; DIRECTION="$2"; shift 2 ;;
            -l | --layout ) got_layout=1; LAYOUT="$2"; shift 2 ;;
            -t | --dtype ) got_dtype=1; DTYPE="$2"; shift 2 ;;
            --driver ) DRIVER="$2"; shift 2 ;;
            --) shift; break ;;
            *) echo "$0: Unexpected option $1"; usage ;;
        esac
    done

    # Check required args
    [[ $got_layout == 1 && $got_direction == 1 && $got_dtype == 1 ]] || [[ $TESTALL == 1 ]] || usage

    # Validate options
    if [[ $got_dtype == 1 ]]; then
        case "$DTYPE" in
            fp16 | fp32 ) ;;
            * ) echo "$0: Invalid dtype $DTYPE"; usage ;;
        esac
    fi

    # Detect XDLOPS if not specified
    if [[ $XDLOPS == -1 ]]; then
      case $(/opt/rocm/bin/rocm_agent_enumerator) in
          *gfx908*)          XDLOPS=1 ;;
          *gfx906*|*gfx900*) XDLOPS=0 ;;
          *)                 echo "No useful GPU found." ; exit 2 ;;
      esac
    fi
}

function get_configs() {
    local line="#"
    while IFS= read -r line; do
        if [[ $line == \#* || $line == "" ]]; then
            continue
        fi
        ALL_CONFIGS+=("$line")
    done
}

function construct_fixed_args() {
    if [[ -z "$LAYOUT" ]]; then
        echo "Internal error: failed to parse arguments"
        exit 3
    fi
    if [[ ! -x "$DRIVER" ]]; then
       echo "$0: Driver command $DRIVER not found or not executable"
       exit 2
    fi
    case "$DTYPE" in
        fp16 ) FIXED_CONFIG_ARGS=("convfp16") ;;
        fp32 ) FIXED_CONFIG_ARGS=("conv") ;;
        * ) echo "Can't happen"; exit 3 ;;
    esac

    FIXED_CONFIG_ARGS+=("--forw" "$DIRECTION" "--in_layout" "$LAYOUT"
                       "--out_layout" "$LAYOUT" "--fil_layout" "$LAYOUT")
}

function create_configs() {
    TMPFILE=$(mktemp /tmp/test-config-cmds-miopen.XXXXXX)
    if [[ $? != 0 ]]; then
       echo "$0: Couldn't make tempfile: $TMPFILE"
       exit 1
    fi

    for line in "${ALL_CONFIGS[@]}"; do
        echo "$DRIVER" "${FIXED_CONFIG_ARGS[@]}" "$line" >>"$TMPFILE"
    done
}

function clean_miopen_caches() {
    rm -rf /tmp/miopen-*
    rm -rf ~/.cache/miopen
    rm -rf ~/.config/miopen/
}

function setup_environment() {
    export MIOPEN_FIND_MODE=1
    export MIOPEN_DRIVER_USE_GPU_REFERENCE=1

    if [[ $TUNING == 1 ]]; then
        export MIOPEN_FIND_ENFORCE=4
    fi

    declare -xg MIOPEN_DEBUG_FIND_ONLY_SOLVER
    case "$DIRECTION" in
        1) MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwd ;;
        2) MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmBwd ;;
        4) MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmWrW ;;
        *) echo "$0: Unsupported direction flag $DIRECTION"; exit 2
    esac

    if [[ $XDLOPS == 1 ]]; then
       MIOPEN_DEBUG_FIND_ONLY_SOLVER+="Xdlops"
    fi
    export MIOPEN_DEBUG_FIND_ONLY_SOLVER
}

function run_tests() {
    if [[ -z "$MIOPEN_DEBUG_FIND_ONLY_SOLVER" || ! -f "$TMPFILE" ]]
    then
        echo "Test execution preconditions not met"
        exit 3
    fi

    local exit_status
    # The extra bash -c accounts for parallel trying to pass each line as one
    # command-line argument
    # The 'awk' invocation prints all lines of output while ensuring indicators of
    # success are present
    parallel -j 1 bash -c {} '|' awk \
    "'BEGIN { status=2 } /Verifies OK/ { status=status-1 } /ConvMlirIgemm/ { status=status-1 } 1; END { exit(status) }'" \
    <"$TMPFILE"
    exit_status=$?

    rm "$TMPFILE"
    echo -n "$DTYPE $LAYOUT XDLOPS=$XDLOPS DIR=$DIRECTION: "
    if [[ $exit_status != 0 ]]; then
        echo "$exit_status tests failed on $(hostname)"
        exit $exit_status
    else
        echo "Success!"
    fi
}

function run_tests_for_a_config() {
    construct_fixed_args
    create_configs
    setup_environment
    run_tests
}

function run_tests_for_all_configs() {
    for t in ${ALL_DTYPES[@]}; do
        for d in ${ALL_DIRECTIONS[@]}; do
            for l in ${ALL_LAYOUTS[@]}; do
                DTYPE=$t
                DIRECTION=$d
                LAYOUT=$l
                run_tests_for_a_config
            done
        done
    done
}

function main() {
    clean_miopen_caches
    parse_options "$@"
    get_configs
    if [[ $TESTALL == 1 ]]; then
        run_tests_for_all_configs
    else
        run_tests_for_a_config
    fi
}

main "$@"
