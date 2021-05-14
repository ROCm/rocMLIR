#!/bin/bash

declare -a FIXED_CONFIG_ARGS=()
declare -g TMPFILE
declare -ig XDLOPS=0
declare -g DTYPE=""
declare -g DIRECTION=""
declare -g LAYOUT=""
declare -g DRIVER="./bin/MIOpenDriver"

function usage() {
    echo "$0: [-d | --direction] DIR [-t | --dtype] [fp16 | fp32] \
[-l | --layout] LAYOUT [-x | --xdlops] [-X | --no-xdlops (default)]\
[--driver DRIVER]\
then pass in configs on standard input"
    exit 2
}

function parse_options() {
    local -i got_layout=0
    local -i got_dtype=0
    local -i got_direction=0

    local parsed_args
    parsed_args=$(getopt -n "$0" -o d:t:l:xX \
                         --long direction:,dtype:,layout:,xdlops,no-xdlops,driver: -- "$@")
    local -i valid_args=$?
    if [[ $valid_args -ne 0 ]]; then
        usage
    fi
    eval set -- "$parsed_args"
    while :
    do
        case "$1" in
            -x | --xdlops ) XDLOPS=1; shift; ;;
            -X | --no-xdlops ) XDLOPS=0; shift ;;
            -d | --direction ) got_direction=1; DIRECTION="$2"; shift 2 ;;
            -l | --layout ) got_layout=1; LAYOUT="$2"; shift 2 ;;
            -t | --dtype ) got_dtype=1; DTYPE="$2"; shift 2 ;;
            --driver ) DRIVER="$2"; shift 2 ;;
            --) shift; break ;;
            *) echo "$0: Unexpected option $1"; usage ;;
        esac
    done

    # Check required args
    [[ $got_layout == 1 && $got_direction == 1 && $got_dtype == 1 ]] || usage

    # Validate options
    case "$DTYPE" in
        fp16 | fp32 ) ;;
        * ) echo "$0: Invalid dtype $DTYPE"; usage ;;
    esac
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
        fp16 ) FIXED_CONFIG_ARGS+=("convfp16") ;;
        fp32 ) FIXED_CONFIG_ARGS+=("conv") ;;
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
    # Clever thing that makes sure the file doesn't hang around, per
    # https://stackoverflow.com/questions/55435352

    local line="#"
    while IFS= read -r line; do
        if [[ $line == \#* || $line == "" ]]; then
            continue
        fi
        echo "\"$DRIVER\"" "${FIXED_CONFIG_ARGS[@]}" "$line" >>"$TMPFILE"
    done
}

function clean_miopen_caches() {
    rm -rf /tmp/miopen-*
    rm -rf ~/.cache/miopen
    rm -rf ~/.config/miopen/
}

function setup_environment() {
    export MIOPEN_FIND_MODE=1
    export MIOPEN_DEBUG_CONV_GEMM=0
    export MIOPEN_DRIVER_USE_GPU_REFERENCE=1

    declare -xg MIOPEN_DEBUG_FIND_ONLY_SOLVER
    case "$DIRECTION" in
        1) MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvMlirIgemmFwd ;;
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

    clean_miopen_caches

    local exit_status
    parallel <"$TMPFILE"
    exit_status=$?

    rm "$TMPFILE"
    echo -n "$DTYPE $LAYOUT XDLOPS=$XDLOPS DIR=$DIRECTION: "
    if [[ $exit_status != 0 ]]; then
        echo "$exit_status tests failed"
        exit $exit_status
    else
        echo "Success!"
    fi
}

function main() {
    parse_options "$@"
    construct_fixed_args
    create_configs
    setup_environment
    run_tests
}

main "$@"
